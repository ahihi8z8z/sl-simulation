#!/usr/bin/env python3
"""Calculate per-day and overall stats (mean, std, CoV) for Azure traffic CSVs."""

import argparse
import sys
from pathlib import Path

import pandas as pd


MINUTES_PER_DAY = 24 * 60


def stats(series: pd.Series) -> dict:
    s = series[series > 0]
    mean = s.mean()
    std = s.std()
    cov = std / mean if mean > 0 else float("nan")
    return {"mean": mean, "std": std, "cov": cov, "peak": s.max(), "n_minutes": len(s)}


def analyze(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("minute")
    df["day"] = (df["minute"] // MINUTES_PER_DAY).astype(int)

    rows = []

    for day, group in df.groupby("day"):
        s = stats(group["count"])
        rows.append({"file": path.stem, "day": day, **s})

    s = stats(df["count"])
    rows.append({"file": path.stem, "day": "total", **s})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="CSV files or directories; defaults to datasets/traffic_pattern/azure/",
    )
    parser.add_argument("--csv", metavar="FILE", help="also write results to CSV")
    parser.add_argument("--day", type=int, metavar="N", help="show only this day number")
    parser.add_argument("--total", action="store_true", help="show only total row per file")
    args = parser.parse_args()

    paths: list[Path] = []
    if not args.paths:
        args.paths = [Path("datasets/traffic_pattern/azure")]
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
        elif p.suffix == ".csv":
            paths.append(p)
        else:
            print(f"Skipping {p}", file=sys.stderr)

    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    frames = [analyze(p) for p in paths]
    result = pd.concat(frames, ignore_index=True)

    if args.day is not None:
        result = result[result["day"] == args.day]
    elif args.total:
        result = result[result["day"] == "total"]

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(result.to_string(index=False))

    if args.csv:
        result.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
