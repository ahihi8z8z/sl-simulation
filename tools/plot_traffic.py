"""Plot traffic patterns from traffic_pattern CSVs.

Usage:
    python tools/plot_traffic.py datasets/traffic_pattern/*.csv
    python tools/plot_traffic.py datasets/traffic_pattern/*.csv --output-dir plots
    python tools/plot_traffic.py datasets/traffic_pattern/*.csv --bucket-minutes 5
    python tools/plot_traffic.py datasets/traffic_pattern/Java_APIG-S_*.csv --plot-style line
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAFFIC_COLOR = "#FF9800"


def _bucket_label(bucket_minutes: int) -> str:
    if bucket_minutes == 60:
        return "hour"
    if bucket_minutes % 60 == 0:
        return f"{bucket_minutes // 60} h"
    return f"{bucket_minutes} min"


def load_and_aggregate(csv_path: str, bucket_minutes: int = 60) -> pd.DataFrame:
    """Load traffic CSV and aggregate into N-minute buckets.

    Output columns: bucket (bucket index), count (sum per bucket), day
    (bucket_start_minute / (24*60)).
    """
    df = pd.read_csv(csv_path)
    minutes = np.arange(df["minute"].min(), df["minute"].max() + 1)
    full = pd.DataFrame({"minute": minutes}).merge(df, on="minute", how="left")
    full["count"] = full["count"].fillna(0).astype(float)
    full["bucket"] = full["minute"] // bucket_minutes
    bucketed = full.groupby("bucket")["count"].sum().reset_index()
    bucketed["day"] = (bucketed["bucket"] * bucket_minutes) / (24.0 * 60.0)
    return bucketed


def plot_traffic(csv_path: str, output_dir: str, plot_style: str = "scatter",
                 plot_range: tuple[float, float] | None = None,
                 bucket_minutes: int = 60) -> str:
    """Plot traffic for a single CSV. Returns output path."""
    name = Path(csv_path).stem
    label = re.sub(r"_\d+---.*$", "", name)

    bucketed = load_and_aggregate(csv_path, bucket_minutes=bucket_minutes)

    if plot_range is not None:
        bucketed = bucketed[(bucketed["day"] >= plot_range[0]) & (bucketed["day"] <= plot_range[1])]

    fig, ax = plt.subplots(figsize=(10, 4))

    unit = _bucket_label(bucket_minutes)
    series_label = f"Traffic ({unit})"

    nonzero = bucketed["count"] > 0
    if plot_style == "scatter":
        ax.scatter(bucketed.loc[nonzero, "day"], bucketed.loc[nonzero, "count"],
                   color=TRAFFIC_COLOR, s=8, alpha=0.6, label=series_label)
    else:
        ax.plot(bucketed["day"], bucketed["count"],
                color=TRAFFIC_COLOR, linewidth=0.8, alpha=0.7, label=series_label)

    mean_val = bucketed["count"].mean()
    ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_val:.2f}")

    ax.set_xlabel("Day")
    ax.set_ylabel(f"Requests / {unit}")
    ax.set_title(f"Traffic — {label}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    suffix = "" if bucket_minutes == 60 else f"_{bucket_minutes}m"
    output_path = os.path.join(output_dir, f"{label}_traffic{suffix}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot traffic patterns from CSV files")
    parser.add_argument("csv_files", nargs="+", help="Traffic pattern CSV file(s)")
    parser.add_argument("--output-dir", default="plots",
                        help="Output directory (default: plots)")
    parser.add_argument("--plot-style", choices=["scatter", "line"], default="scatter",
                        help="Plot style (default: scatter)")
    parser.add_argument("--plot-range", default=None, metavar="RANGE",
                        help="Day range, e.g. '5-10'")
    parser.add_argument("--bucket-minutes", type=int, default=60,
                        help="Aggregate into N-minute buckets (default: 60 = hourly, 5 = 5-min, etc.)")
    args = parser.parse_args()

    if args.bucket_minutes < 1:
        parser.error("--bucket-minutes must be >= 1")

    plot_range = None
    if args.plot_range:
        parts = args.plot_range.split("-", 1)
        plot_range = (float(parts[0]), float(parts[1]))

    for csv_path in args.csv_files:
        path = plot_traffic(csv_path, args.output_dir, args.plot_style, plot_range,
                            bucket_minutes=args.bucket_minutes)
        print(f"  {path}")

    print(f"\nDone. Plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
