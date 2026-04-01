"""Plot cold-start cost breakdown from *_costs.csv files.

Generates a stacked bar chart of mean cost proportions across all functions.

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


COST_COLS = ["podAllocationCost", "deployCodeCost", "deployDependencyCost"]
COST_LABELS = ["Pod Allocation", "Deploy Code", "Deploy Dependency"]


def find_cost_files(input_dir: str) -> list[dict]:
    """Find *_costs.csv files."""
    cost_files = sorted(glob.glob(os.path.join(input_dir, "*_costs.csv")))
    functions = []
    for cost_path in cost_files:
        filename = os.path.basename(cost_path)
        base = filename.replace("_costs.csv", "")
        label = re.sub(r"_\d+---.*$", "", base)
        functions.append({
            "label": label,
            "base": base,
            "costs_path": cost_path,
        })
    return functions


def load_costs(path: str) -> pd.DataFrame:
    """Load costs CSV. Returns DataFrame with 3 cost columns + total."""
    df = pd.read_csv(path)
    for col in COST_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["total_cost"] = df[COST_COLS].sum(axis=1)
    return df


def plot_combined_costs(functions: list[dict], output_dir: str) -> str:
    """Plot stacked bar chart of mean cost proportions for all functions."""
    rows = []
    for func_info in functions:
        costs_df = load_costs(func_info["costs_path"])
        means = {col: costs_df[col].mean() for col in COST_COLS}
        means["runtime"] = func_info["label"]
        means["_count"] = len(costs_df)
        rows.append(means)

    df = pd.DataFrame(rows)
    df["_total"] = df[COST_COLS].sum(axis=1)
    df = df.sort_values("_total", ascending=False).reset_index(drop=True)

    mean_vals = df[COST_COLS].values
    row_totals = mean_vals.sum(axis=1, keepdims=True)
    pct_vals = mean_vals / np.where(row_totals > 0, row_totals, 1.0)

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(df))
    bottoms = np.zeros(len(df))
    colors = ["#2f5aa8", "#f2b500", "#8b0000"]

    for j, (col, clabel, color) in enumerate(zip(COST_COLS, COST_LABELS, colors)):
        axes.bar(x, pct_vals[:, j], bottom=bottoms, label=clabel, color=color)
        bottoms += pct_vals[:, j]

    legend_handles = {}
    for j, clabel in enumerate(COST_LABELS):
        legend_handles[clabel] = axes.patches[j * len(df)]

    top_pad = 0.12
    fig.subplots_adjust(top=1.0 - top_pad)
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0 - top_pad + 0.004),
        ncol=len(COST_LABELS),
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        borderpad=0.28,
    )

    for i in range(len(df)):
        cumulative = 0.0
        for j, col in enumerate(COST_COLS):
            pct = pct_vals[i, j]
            val = mean_vals[i, j]
            y = cumulative + pct / 2
            if pct > 0.05:
                axes.text(i, y, f"{val:.3f}", ha="center", va="center",
                          fontsize=8, color="white", fontweight="bold")
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


def main():
    parser = argparse.ArgumentParser(description="Plot cold-start cost breakdown")
    parser.add_argument("--input-dir", default="datasets/request_per_min",
                        help="Directory containing *_costs.csv files")
    parser.add_argument("--output-dir", default="plots",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    functions = find_cost_files(args.input_dir)
    if not functions:
        print(f"No *_costs.csv files found in {args.input_dir}")
        return

    print(f"Found {len(functions)} functions:")
    for f in functions:
        print(f"  {f['label']}")

    path = plot_combined_costs(functions, args.output_dir)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
