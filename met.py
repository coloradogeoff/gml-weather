#!/usr/bin/env python3

import argparse
import json
import math
import re
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError



def interp_direction_deg(dir1, dir2, frac):
    """Circular interpolation for wind direction in degrees."""
    a1 = math.radians(dir1)
    a2 = math.radians(dir2)

    x = (1 - frac) * math.cos(a1) + frac * math.cos(a2)
    y = (1 - frac) * math.sin(a1) + frac * math.sin(a2)

    ang = math.degrees(math.atan2(y, x))
    return ang % 360


def parse_target_datetime(value):
    if re.fullmatch(r"\d{14}", value):
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    if re.fullmatch(r"\d{12}", value):
        return datetime.strptime(value, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value):
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", value):
        return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    raise argparse.ArgumentTypeError(
        "Invalid datetime. Use 'YYYY-MM-DD HH:MM[:SS]' or 'YYYYMMDDHHMM[SS]'."
    )


def load_sites(path):
    sites = {}
    current = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue

                if line.startswith("- "):
                    key, value = line[2:].split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "code":
                        current = {"code": value, "name": value}
                        sites[value.upper()] = current
                    continue

                if current is None:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                current[key] = value
    except FileNotFoundError as exc:
        raise SystemExit(f"Sites file not found: {path}") from exc

    for code, site in sites.items():
        try:
            site["latitude"] = float(site["latitude"])
            site["longitude"] = float(site["longitude"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid lat/lon for site '{code}' in {path}") from exc

    return sites


def resolve_location(args):
    if args.site:
        if args.lon is not None:
            raise SystemExit("Do not pass --lon with --site. Use only --site.")
        sites = load_sites(args.sites_file)
        site_code = args.site.upper()
        if site_code not in sites:
            raise SystemExit(
                f"Unknown site '{args.site}'. Check code in {args.sites_file}."
            )
        site = sites[site_code]
        return site["latitude"], site["longitude"], site_code, site.get("name", site_code)

    if args.lat is None or args.lon is None:
        raise SystemExit("When not using --site, both --lat and --lon are required.")

    return args.lat, args.lon, None, "custom"


def fetch_hourly_wind(lat, lon, target_time):
    date_str = target_time.strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "models": "era5",
    }

    query = urlencode(params)
    request_url = f"{url}?{query}"

    try:
        with urlopen(request_url, timeout=30) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise RuntimeError(f"Open-Meteo request failed with HTTP {status}")
            data = json.load(response)
    except URLError as exc:
        raise SystemExit(f"Failed to reach Open-Meteo API: {exc.reason}") from exc

    return (
        data["hourly"]["time"],
        data["hourly"]["wind_speed_10m"],
        data["hourly"]["wind_direction_10m"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate wind speed/direction at a target UTC datetime."
    )
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument("--site", help="Site code from noaa-sites.yaml")
    location_group.add_argument("--lat", type=float, help="Latitude in decimal degrees")

    parser.add_argument("--lon", type=float, help="Longitude in decimal degrees")
    parser.add_argument(
        "--datetime",
        required=True,
        type=parse_target_datetime,
        help="UTC datetime: 'YYYY-MM-DD HH:MM[:SS]' or 'YYYYMMDDHHMM[SS]'",
    )
    parser.add_argument(
        "--sites-file",
        default="noaa-sites.yaml",
        help="Path to NOAA site list (default: noaa-sites.yaml)",
    )

    args = parser.parse_args()

    lat, lon, site_code, site_name = resolve_location(args)
    target_time = args.datetime

    times, ws, wd = fetch_hourly_wind(lat, lon, target_time)
    hourly_dt = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]

    print(f"Location: {site_name} ({lat}, {lon})")
    if site_code:
        print(f"Site: {site_code}")
    print("Target (UTC):", target_time.isoformat())

    valid_points = [
        (hourly_dt[i], ws[i], wd[i])
        for i in range(len(hourly_dt))
        if ws[i] is not None and wd[i] is not None
    ]
    if len(valid_points) < 2:
        raise SystemExit("Not enough valid wind data points returned for interpolation.")

    for i in range(len(valid_points) - 1):
        t0, ws0, wd0 = valid_points[i]
        t1, ws1, wd1 = valid_points[i + 1]
        if t0 <= target_time <= t1:
            frac = (target_time - t0).total_seconds() / (t1 - t0).total_seconds()

            speed_interp = (1 - frac) * ws0 + frac * ws1
            dir_interp = interp_direction_deg(wd0, wd1, frac)

            print("Bounding hours:", t0.isoformat(), "to", t1.isoformat())
            print(f"10m wind speed (interp): {speed_interp:.2f} m/s")
            print(f"10m wind direction (interp): {dir_interp:.1f} deg")
            break
    else:
        print("Target time not found within valid returned data range.")

    nearest_idx = min(
        range(len(valid_points)),
        key=lambda i: abs((valid_points[i][0] - target_time).total_seconds()),
    )
    print("\nNearest hour:")
    nearest_time, nearest_ws, nearest_wd = valid_points[nearest_idx]
    print(nearest_time.isoformat(), nearest_ws, "m/s", nearest_wd, "deg")


if __name__ == "__main__":
    main()
