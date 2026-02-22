#!/usr/bin/env python3

import argparse
import sys

import pandas as pd

DEFAULT_URL = "https://gml.noaa.gov/aftp/data/hats/hcfcs/hcfc142b/flasks/HCFC142B_GCMS_flask.txt"


def load_flask_data(source: str) -> pd.DataFrame:
    """Load a NOAA flask text file into a pandas DataFrame."""
    # NOAA GML flask files use:
    # - line 1: metadata
    # - line 2: tab-delimited column names
    df = pd.read_csv(
        source, sep="\t", skiprows=1, engine="python", na_values=["nd", "-99"]
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    lower_cols = {c.lower(): c for c in df.columns}
    datetime_col = lower_cols.get("yyyymmdd hhmmss")
    if datetime_col:
        df["datetime_utc"] = pd.to_datetime(
            df[datetime_col],
            format="%Y%m%d %H%M",
            utc=True,
            errors="coerce",
        )

    needed = {"year", "month", "day", "hour", "minute", "second"}
    if needed.issubset(lower_cols):
        df["datetime_utc"] = pd.to_datetime(
            {
                "year": df[lower_cols["year"]],
                "month": df[lower_cols["month"]],
                "day": df[lower_cols["day"]],
                "hour": df[lower_cols["hour"]],
                "minute": df[lower_cols["minute"]],
                "second": df[lower_cols["second"]],
            },
            utc=True,
            errors="coerce",
        )

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load NOAA flask text data into a pandas DataFrame."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_URL,
        help="URL or local file path (default: NOAA HCFC142B flask URL).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of preview rows to print (default: 5).",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional output path to save parsed data as CSV.",
    )

    args = parser.parse_args()

    try:
        df = load_flask_data(args.source)
    except Exception as exc:
        print(f"Failed to load data from {args.source}: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    print("\nColumns:")
    print(", ".join(df.columns.astype(str)))
    print("\nPreview:")
    print(df.head(args.head).to_string(index=False))

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved CSV to {args.save_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
