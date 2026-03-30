"""Plot request arrival rate from request_trace.csv files.

Usage:
    python tools/plot_trace.py logs/run_*/request_trace.csv
    python tools/plot_trace.py logs/run_*/request_trace.csv --bin 60 --plot 2-5
    python tools/plot_trace.py trace1.csv trace2.csv --labels "baseline,lstm" --bin 3600
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_trace(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "completed"].copy()
    return df


def plot_traces(
    csv_paths: list[str],
    labels: list[str] | None,
    bin_seconds: float,
    output_path: str,
    plot_range: tuple[float, float] | None,
    plot_style: str,
    show_cold_starts: bool,
) -> None:
    if labels is None:
        labels = [Path(p).parent.name for p in csv_paths]

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    n_plots = 2 if show_cold_starts else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots), squeeze=False)

    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        df = load_trace(csv_path)
        color = colors[i % len(colors)]

        # Bin arrivals by time
        df["bin"] = (df["arrival_time"] // bin_seconds).astype(int)
        arrival_rate = df.groupby("bin").size()

        # Convert bin index to days
        x = arrival_rate.index * bin_seconds / 86400.0
        y = arrival_rate.values

        # Filter to plot range
        if plot_range is not None:
            mask = (x >= plot_range[0]) & (x <= plot_range[1])
            x = x[mask]
            y = y[mask]

        ax = axes[0, 0]
        if plot_style == "line":
            ax.plot(x, y, color=color, alpha=0.7, linewidth=0.8, label=label)
        else:
            ax.scatter(x, y, color=color, alpha=0.5, s=4, label=label)

        # Cold starts subplot
        if show_cold_starts:
            cold = df[df["cold_start"] == True]
            cold_rate = cold.groupby("bin").size()
            cx = cold_rate.index * bin_seconds / 86400.0
            cy = cold_rate.values

            if plot_range is not None:
                mask = (cx >= plot_range[0]) & (cx <= plot_range[1])
                cx = cx[mask]
                cy = cy[mask]

            ax2 = axes[1, 0]
            if plot_style == "line":
                ax2.plot(cx, cy, color=color, alpha=0.7, linewidth=0.8, label=label)
            else:
                ax2.scatter(cx, cy, color=color, alpha=0.5, s=4, label=label)

    # Format top plot
    ax = axes[0, 0]
    bin_label = _bin_label(bin_seconds)
    ax.set_ylabel(f"Requests / {bin_label}")
    ax.set_title("Request Arrival Rate")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    if show_cold_starts:
        ax2 = axes[1, 0]
        ax2.set_xlabel("Day")
        ax2.set_ylabel(f"Cold Starts / {bin_label}")
        ax2.set_title("Cold Start Rate")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax.set_xlabel("Day")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def _bin_label(bin_seconds: float) -> str:
    if bin_seconds >= 3600:
        return f"{bin_seconds / 3600:.0f}h"
    elif bin_seconds >= 60:
        return f"{bin_seconds / 60:.0f}min"
    else:
        return f"{bin_seconds:.0f}s"


def main():
    parser = argparse.ArgumentParser(description="Plot request arrival rate from trace CSVs")
    parser.add_argument("csv_files", nargs="+", help="request_trace.csv file(s)")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated labels for each trace")
    parser.add_argument("--bin", type=float, default=60.0,
                        help="Time bin in seconds (default: 60)")
    parser.add_argument("--plot", default=None, metavar="RANGE",
                        help="Day range to plot, e.g. '2-5' or '0-14'")
    parser.add_argument("--plot-style", choices=["scatter", "line"], default="line",
                        help="Plot style (default: line)")
    parser.add_argument("--cold-starts", action="store_true",
                        help="Also plot cold start rate")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: auto)")
    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else None

    plot_range = None
    if args.plot:
        parts = args.plot.split("-", 1)
        plot_range = (float(parts[0]), float(parts[1]))

    output = args.output
    if output is None:
        first_dir = str(Path(args.csv_files[0]).parent)
        output = os.path.join(first_dir, "trace_plot.png")

    plot_traces(
        args.csv_files, labels, args.bin, output,
        plot_range, args.plot_style, args.cold_starts,
    )


if __name__ == "__main__":
    main()
