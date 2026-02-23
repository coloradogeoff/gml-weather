#! /usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from met_api import ArchivedMetAPI
import flask_data

api = ArchivedMetAPI("noaa-sites.yaml")
sites = ["alt", "brw", "cgo", "nwr", "hfm"]
RESULTS_DIR = "results"
gml_df = None
day_cache = {}

class MetComparison:
    def __init__(self, source_url, min_date="2020-01-01"):
        self.source_url = source_url
        min_ts = pd.Timestamp(min_date)
        if min_ts.tzinfo is None:
            min_ts = min_ts.tz_localize("UTC")
        else:
            min_ts = min_ts.tz_convert("UTC")
        self.min_date = min_ts

    def load_gml_data(self):
        df = flask_data.load_flask_data(self.source_url)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
        df = (
            df.assign(
                wind_spd=lambda d: pd.to_numeric(d["wind_spd"], errors="coerce"),
                wind_dir=lambda d: pd.to_numeric(d["wind_dir"], errors="coerce"),
            )
            .dropna(subset=["datetime_utc", "wind_spd", "wind_dir"])
            .loc[lambda d: d["wind_spd"].between(0, 100)]
        )
        return df.loc[df["datetime_utc"] > self.min_date]


def api_wind_for_ts(site, ts):
    if pd.isna(ts):
        return (pd.NA, pd.NA)

    lat, lon, _, _ = api.resolve_location(site, None, None)

    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    dt = ts.to_pydatetime()
    day_key = (site, ts.date().isoformat())

    if day_key not in day_cache:
        day_cache[day_key] = api.fetch_hourly_wind(lat, lon, dt)

    times, ws, wd = day_cache[day_key]
    interpolated, nearest = api.interpolate_wind(times, ws, wd, dt)

    if interpolated is not None:
        _, _, spd, direction = interpolated
        print(f"{site} at {ts.isoformat()}: using interpolated data for API wind ({spd} m/s, {direction} deg)")
        return spd, direction

    # Fallback if target isn't bracketed.
    _, spd, direction = nearest
    return spd, direction


def enrich_site_with_api(site, site_df, progress_every=250):
    total = len(site_df)
    start = time.monotonic()
    api_data = []

    for i, ts in enumerate(site_df["datetime_utc"], start=1):
        api_data.append(api_wind_for_ts(site, ts))
        if i == 1 or i % progress_every == 0 or i == total:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = total - i
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "unknown"
            print(
                f"[{site}] processed {i}/{total} ({i/total:.1%}) "
                f"elapsed={elapsed:.1f}s eta={eta_text}"
            )

    api_cols = ["api_wind_spd", "api_wind_dir"]
    site_df[api_cols] = pd.DataFrame(api_data, index=site_df.index)
    return site_df


def save_comparison_csv(path, df):
    out = df.copy()
    if "dec_date" in out.columns:
        dec = pd.to_numeric(out["dec_date"], errors="coerce")
        out["dec_date"] = dec.map(lambda v: f"{v:.6f}" if pd.notna(v) else "")
    out.to_csv(path, index=False, float_format="%.3f")


def plot_wind_spd_comparison(site, site_df):
    api_spd_col = "api_wind_spd"
    api_dir_col = "api_wind_dir"
    subset = (
        site_df.loc[:, ["datetime_utc", "wind_spd", api_spd_col, "wind_dir", api_dir_col]]
        .dropna()
        .copy()
    )
    subset["spd_diff"] = subset["wind_spd"] - subset[api_spd_col]
    subset["dir_diff"] = subset["wind_dir"] - subset[api_dir_col]

    fig, (ax_top, ax_main) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    # Top (1/4): difference
    ax_top.scatter(subset["datetime_utc"], subset["spd_diff"], s=20, alpha=0.8, color="tab:red")
    ax_top.axhline(0, color="black", lw=1)
    ax_top.set_ylabel("obs-api")
    ax_top.set_title(f"Wind Speed Difference ({site.upper()})")

    # Bottom (3/4): main time series
    ax_main.scatter(subset["datetime_utc"], subset["wind_spd"], s=15, alpha=0.7, label="Observed")
    ax_main.scatter(subset["datetime_utc"], subset[api_spd_col], s=15, alpha=0.7, label="Reanalysis")
    ax_main.set_ylabel("wind speed (m/s)")
    ax_main.set_xlabel("datetime_utc")
    ax_main.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"wind_speed_comparison_{site}.png"))


def plot_wind_dir_comparison(site, site_df):
    api_dir_col = "api_wind_dir"
    subset = (
        site_df.loc[:, ["datetime_utc", "wind_dir", api_dir_col]]
        .dropna()
        .copy()
    )

    # Circular signed difference: obs - api, in degrees, range [-180, 180]
    obs_rad = np.deg2rad(subset["wind_dir"].to_numpy())
    api_rad = np.deg2rad(subset[api_dir_col].to_numpy())
    subset["dir_diff"] = np.rad2deg(np.arctan2(np.sin(obs_rad - api_rad), np.cos(obs_rad - api_rad)))

    fig, (ax_top, ax_main) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    # Top (1/4): circular difference
    ax_top.scatter(subset["datetime_utc"], subset["dir_diff"], s=20, alpha=0.8, color="tab:red")
    ax_top.axhline(0, color="black", lw=1)
    ax_top.set_ylabel("obs-api (deg)")
    ax_top.set_title(f"Wind Direction Difference (circular, {site.upper()})")

    # Bottom (3/4): main time series
    ax_main.scatter(subset["datetime_utc"], subset["wind_dir"], s=15, alpha=0.7, label="Observed")
    ax_main.scatter(subset["datetime_utc"], subset[api_dir_col], s=15, alpha=0.7, label="Reanalysis")
    ax_main.set_ylabel("wind direction (deg)")
    ax_main.set_xlabel("datetime_utc")
    ax_main.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"wind_direction_comparison_{site}.png"))


def main():
    global gml_df
    os.makedirs(RESULTS_DIR, exist_ok=True)
    day_cache.clear()
    comparison = MetComparison(
        "https://gml.noaa.gov/aftp/data/hats/hcfcs/hcfc142b/flasks/HCFC142B_GCMS_flask.txt"
    )
    gml_df = comparison.load_gml_data()

    for site in sites:
        csv_path = os.path.join(RESULTS_DIR, f"{site}_gml_comparison.csv")
        if os.path.exists(csv_path):
            print(f"[{site}] loading cached comparison data from {csv_path}")
            site_df = pd.read_csv(csv_path, parse_dates=["datetime_utc"])
            # Backward compatibility with older per-site API column names.
            legacy_spd_col = f"{site}_wind_spd"
            legacy_dir_col = f"{site}_wind_dir"
            if "api_wind_spd" not in site_df.columns and legacy_spd_col in site_df.columns:
                site_df = site_df.rename(columns={legacy_spd_col: "api_wind_spd"})
            if "api_wind_dir" not in site_df.columns and legacy_dir_col in site_df.columns:
                site_df = site_df.rename(columns={legacy_dir_col: "api_wind_dir"})
            if "api_wind_spd" in site_df.columns and "api_wind_dir" in site_df.columns:
                save_comparison_csv(csv_path, site_df)
        else:
            print(f"[{site}] cache miss; fetching API data and writing {csv_path}")
            site_df = gml_df.loc[gml_df["site"] == site].copy()
            if site_df.empty:
                continue
            site_df = enrich_site_with_api(site, site_df)
            save_comparison_csv(csv_path, site_df)

        if site_df.empty:
            continue
        plot_wind_spd_comparison(site, site_df)
        plot_wind_dir_comparison(site, site_df)


if __name__ == "__main__":
    main()
