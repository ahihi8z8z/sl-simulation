"""Plot cold-start cost analysis from *_costs.csv files.

Generates:
  - 6 individual figures: smoothed daily request rate per function
  - 1 combined figure: stacked bar chart of mean cost proportions across all functions

Usage:
    python tools/plot_costs.py [--input-dir datasets/request_per_min] [--output-dir plots]
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Style constants
# ------------------------------------------------------------------ #

COST_COLS = ["podAllocationCost", "deployCodeCost", "deployDependencyCost"]
COST_LABELS = ["Pod Allocation", "Deploy Code", "Deploy Dependency"]

TRAFFIC_COLOR = "#FF9800"


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def find_function_files(input_dir: str) -> list[dict]:
    """Find paired traffic + costs CSVs for each function."""
    cost_files = sorted(glob.glob(os.path.join(input_dir, "*_costs.csv")))
    functions = []
    for cost_path in cost_files:
        filename = os.path.basename(cost_path)
        base = filename.replace("_costs.csv", "")
        label = re.sub(r"_\d+---.*$", "", base)
        traffic_path = os.path.join(input_dir, base + ".csv")
        functions.append({
            "label": label,
            "base": base,
            "traffic_path": traffic_path if os.path.exists(traffic_path) else None,
            "costs_path": cost_path,
        })
    return functions


def load_traffic(path: str) -> pd.DataFrame:
    """Load traffic CSV. Returns DataFrame with time (seconds) and mean_rq."""
    df = pd.read_csv(path)
    df["mean_rq"] = pd.to_numeric(df["mean_rq"], errors="coerce")
    return df


def load_costs(path: str) -> pd.DataFrame:
    """Load costs CSV. Returns DataFrame with 3 cost columns + total."""
    df = pd.read_csv(path)
    for col in COST_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["total_cost"] = df[COST_COLS].sum(axis=1)
    return df


# ------------------------------------------------------------------ #
# Per-function plot: smoothed daily traffic
# ------------------------------------------------------------------ #

def parse_smooth_window(s: str) -> int:
    """Parse smoothing window string like '1h', '6h', '1d' into minutes."""
    s = s.strip().lower()
    if s.endswith("d"):
        return int(s[:-1]) * 1440
    elif s.endswith("h"):
        return int(s[:-1]) * 60
    elif s.endswith("m"):
        return int(s[:-1])
    else:
        return int(s)


RATE_UNITS = {
    "min": {"factor": 1.0, "ylabel": "Requests / min"},
    "hour": {"factor": 60.0, "ylabel": "Requests / hour"},
    "day": {"factor": 1440.0, "ylabel": "Requests / day"},
}


def plot_traffic_daily(ax, traffic_df: pd.DataFrame, label: str,
                       smooth_minutes: int = 60, rate_unit: str = "min") -> None:
    """Plot request rate smoothed by moving average. X-axis in days."""
    df = traffic_df.copy()
    df["day"] = df["time"] / 86400.0

    # Fill NaN with 0 (no requests), then moving average
    df["filled"] = df["mean_rq"].fillna(0.0)
    window = max(1, smooth_minutes)
    df["smooth"] = df["filled"].rolling(window=window, center=True, min_periods=window).mean()

    # Scale to requested unit
    unit_cfg = RATE_UNITS.get(rate_unit, RATE_UNITS["min"])
    df["smooth"] = df["smooth"] * unit_cfg["factor"]

    ax.plot(df["day"], df["smooth"], color=TRAFFIC_COLOR, linewidth=1.5)

    ax.set_xlabel("Day", fontsize=10)
    ax.set_ylabel(unit_cfg["ylabel"], fontsize=10)
    ax.set_title(f"Traffic (daily avg) — {label}", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_function_traffic(func_info: dict, output_dir: str,
                          smooth_minutes: int = 60, rate_unit: str = "min") -> str | None:
    """Generate traffic figure for one function. Returns output path or None."""
    if func_info["traffic_path"] is None:
        return None

    label = func_info["label"]
    traffic_df = load_traffic(func_info["traffic_path"])

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_traffic_daily(ax, traffic_df, label, smooth_minutes=smooth_minutes, rate_unit=rate_unit)
    fig.suptitle(label, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    safe_name = label.replace(" ", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f"{safe_name}_traffic.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ------------------------------------------------------------------ #
# Combined stacked bar chart: mean cost proportions
# ------------------------------------------------------------------ #

def plot_combined_costs(functions: list[dict], output_dir: str) -> str:
    """Plot stacked bar chart of mean cost proportions for all functions."""
    rows = []
    counts = []
    for func_info in functions:
        label = func_info["base"]  # Use full base name to avoid duplicates
        display = func_info["label"]
        costs_df = load_costs(func_info["costs_path"])
        means = {col: costs_df[col].mean() for col in COST_COLS}
        means["runtime"] = display
        means["_count"] = len(costs_df)
        rows.append(means)

    df = pd.DataFrame(rows)

    # Sort by total mean descending
    df["_total"] = df[COST_COLS].sum(axis=1)
    df = df.sort_values("_total", ascending=False).reset_index(drop=True)

    mean_vals = df[COST_COLS].values  # numpy array
    row_totals = mean_vals.sum(axis=1, keepdims=True)
    pct_vals = mean_vals / np.where(row_totals > 0, row_totals, 1.0)

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(df))
    bottoms = np.zeros(len(df))
    colors = ["#2f5aa8", "#f2b500", "#8b0000"]

    for j, (col, clabel, color) in enumerate(zip(COST_COLS, COST_LABELS, colors)):
        axes.bar(x, pct_vals[:, j], bottom=bottoms, label=clabel, color=color)
        bottoms += pct_vals[:, j]

    # Collect legend handles from bar patches
    legend_handles = {}
    for j, clabel in enumerate(COST_LABELS):
        legend_handles[clabel] = axes.patches[j * len(df)]

    # Remove default legend, use formatted one
    top_pad = 0.12
    fig.subplots_adjust(top=1.0 - top_pad)
    legend_cols = len(COST_LABELS)
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0 - top_pad + 0.004),
        ncol=legend_cols,
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        borderpad=0.28,
        handlelength=2.5,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    for i in range(len(df)):
        cumulative = 0.0
        for j, col in enumerate(COST_COLS):
            pct = pct_vals[i, j]
            val = mean_vals[i, j]
            y = cumulative + pct / 2
            if pct > 0.05:
                axes.text(i, y, f"{val:.3f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            cumulative += pct
        axes.text(i, 1.02, f"n={df.iloc[i]['_count']:.0f}", ha="center", fontsize=8)

    axes.set_xticks(x)
    axes.set_xticklabels(df["runtime"].tolist(), rotation=45, ha="right")
    axes.set_ylabel("Proportion (mean)")
    axes.set_ylim(0, 1.15)
    axes.set_title("Mean Cold-Start Cost Breakdown by Runtime", fontsize=13, fontweight="bold")
    axes.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "costs_combined.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Plot cold-start cost analysis")
    parser.add_argument(
        "--input-dir",
        default="datasets/request_per_min",
        help="Directory containing *_costs.csv and matching traffic CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--smooth",
        default="1h",
        help="Smoothing window for traffic (e.g. 30m, 1h, 6h, 1d). Default: 1h",
    )
    parser.add_argument(
        "--rate-unit",
        default="min",
        choices=["min", "hour", "day"],
        help="Rate unit for traffic y-axis (min, hour, day). Default: min",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    functions = find_function_files(args.input_dir)
    if not functions:
        print(f"No *_costs.csv files found in {args.input_dir}")
        return

    print(f"Found {len(functions)} function types:")
    for f in functions:
        print(f"  {f['label']} ({os.path.basename(f['costs_path'])})")
    print()

    smooth_minutes = parse_smooth_window(args.smooth)
    rate_unit = args.rate_unit
    print(f"Smoothing window: {args.smooth} ({smooth_minutes} minutes), rate unit: {rate_unit}\n")

    # Per-function traffic plots
    for func_info in functions:
        path = plot_function_traffic(func_info, args.output_dir,
                                     smooth_minutes=smooth_minutes, rate_unit=rate_unit)
        if path:
            print(f"  Saved: {path}")

    # Combined cost breakdown
    combined_path = plot_combined_costs(functions, args.output_dir)
    print(f"  Saved: {combined_path}")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
