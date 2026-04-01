"""Plot traffic patterns from traffic_pattern CSVs.

Usage:
    python tools/plot_traffic.py datasets/traffic_pattern/*.csv
    python tools/plot_traffic.py datasets/traffic_pattern/*.csv --output-dir plots
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


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """Load traffic CSV and aggregate to hourly."""
    df = pd.read_csv(csv_path)
    minutes = np.arange(df["minute"].min(), df["minute"].max() + 1)
    full = pd.DataFrame({"minute": minutes}).merge(df, on="minute", how="left")
    full["count"] = full["count"].fillna(0).astype(float)
    full["hour"] = full["minute"] // 60
    hourly = full.groupby("hour")["count"].sum().reset_index()
    hourly["day"] = hourly["hour"] / 24.0
    return hourly


def plot_traffic(csv_path: str, output_dir: str, plot_style: str = "scatter",
                 plot_range: tuple[float, float] | None = None) -> str:
    """Plot traffic for a single CSV. Returns output path."""
    name = Path(csv_path).stem
    label = re.sub(r"_\d+---.*$", "", name)

    hourly = load_and_aggregate(csv_path)

    if plot_range is not None:
        hourly = hourly[(hourly["day"] >= plot_range[0]) & (hourly["day"] <= plot_range[1])]

    fig, ax = plt.subplots(figsize=(10, 4))

    nonzero = hourly["count"] > 0
    if plot_style == "scatter":
        ax.scatter(hourly.loc[nonzero, "day"], hourly.loc[nonzero, "count"],
                   color=TRAFFIC_COLOR, s=8, alpha=0.6, label="Traffic (hourly)")
    else:
        ax.plot(hourly["day"], hourly["count"],
                color=TRAFFIC_COLOR, linewidth=0.8, alpha=0.7, label="Traffic (hourly)")

    mean_val = hourly["count"].mean()
    ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_val:.2f}")

    ax.set_xlabel("Day")
    ax.set_ylabel("Requests / hour")
    ax.set_title(f"Traffic — {label}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{label}_traffic.png")
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
    args = parser.parse_args()

    plot_range = None
    if args.plot_range:
        parts = args.plot_range.split("-", 1)
        plot_range = (float(parts[0]), float(parts[1]))

    for csv_path in args.csv_files:
        path = plot_traffic(csv_path, args.output_dir, args.plot_style, plot_range)
        print(f"  {path}")

    print(f"\nDone. Plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
