"""Plot sweep results from CSV.

Auto-detects sweep parameter columns (non-metric columns) and plots.

Usage:
    python tools/plot_sweep.py logs/sweep_*/results.csv
    python tools/plot_sweep.py logs/sweep_*/results.csv --output-dir plots/sweep
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B",
          "#E91E63", "#3F51B5", "#009688", "#FFC107", "#8BC34A", "#FF5722", "#673AB7", "#CDDC39"]

METRIC_COLS = {"total", "completed", "dropped", "cold_starts", "drop_pct", "cold_start_pct",
               "latency_mean", "mem_per_req_pct", "wall_seconds", "name"}


def _detect_params(df: pd.DataFrame) -> list[str]:
    """Detect sweep parameter columns (non-metric, non-name)."""
    return [c for c in df.columns if c not in METRIC_COLS]


def plot_metric(df: pd.DataFrame, metric: str, param_cols: list[str], output_dir: str) -> None:
    """Plot metric with 4 subplots based on detected parameters."""
    # Need at least 2 param columns for meaningful plots
    # Find columns that look like pool sizes and rate
    pool_cols = [c for c in param_cols if any(k in c.lower() for k in ["prewarm", "warm", "pool"])]
    rate_cols = [c for c in param_cols if any(k in c.lower() for k in ["rate", "scale", "beta", "arrival"])]

    if not rate_cols:
        rate_cols = [c for c in param_cols if c not in pool_cols]
    if not pool_cols and len(param_cols) >= 2:
        pool_cols = [c for c in param_cols if c not in rate_cols]

    # Assign roles
    prewarm_col = next((c for c in pool_cols if "prewarm" in c.lower()), pool_cols[0] if pool_cols else None)
    warm_col = next((c for c in pool_cols if "warm" in c.lower() and "prewarm" not in c.lower()),
                     pool_cols[1] if len(pool_cols) > 1 else None)
    rate_col = rate_cols[0] if rate_cols else None

    if not all([prewarm_col, warm_col, rate_col]):
        print(f"  WARNING: need prewarm, warm, rate columns. Found: {param_cols}")
        return

    max_rate = df[rate_col].max()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(metric, fontsize=14)

    # (0,0) x=rate, lines=prewarm, warm=0
    ax = axes[0][0]
    sub = df[df[warm_col] == 0]
    for i, (pw, g) in enumerate(sub.groupby(prewarm_col)):
        g = g.sort_values(rate_col)
        ax.plot(g[rate_col], g[metric], label=f"{prewarm_col}={int(pw)}",
                color=COLORS[i % len(COLORS)], marker="o", markersize=3)
    ax.set_xlabel(rate_col)
    ax.set_ylabel(metric)
    ax.set_title(f"Prewarm sweep ({warm_col}=0)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (0,1) x=rate, lines=warm, prewarm=0
    ax = axes[0][1]
    sub = df[df[prewarm_col] == 0]
    for i, (w, g) in enumerate(sub.groupby(warm_col)):
        g = g.sort_values(rate_col)
        ax.plot(g[rate_col], g[metric], label=f"{warm_col}={int(w)}",
                color=COLORS[i % len(COLORS)], marker="o", markersize=3)
    ax.set_xlabel(rate_col)
    ax.set_ylabel(metric)
    ax.set_title(f"Warm sweep ({prewarm_col}=0)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (1,0) x=prewarm, lines=warm, rate=max
    ax = axes[1][0]
    sub = df[df[rate_col] == max_rate]
    for i, (w, g) in enumerate(sub.groupby(warm_col)):
        g = g.sort_values(prewarm_col)
        ax.plot(g[prewarm_col], g[metric], label=f"{warm_col}={int(w)}",
                color=COLORS[i % len(COLORS)], marker="o", markersize=3)
    ax.set_xlabel(prewarm_col)
    ax.set_ylabel(metric)
    ax.set_title(f"Prewarm vs Warm ({rate_col}={max_rate})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (1,1) x=warm, lines=prewarm, rate=max
    ax = axes[1][1]
    sub = df[df[rate_col] == max_rate]
    for i, (pw, g) in enumerate(sub.groupby(prewarm_col)):
        g = g.sort_values(warm_col)
        ax.plot(g[warm_col], g[metric], label=f"{prewarm_col}={int(pw)}",
                color=COLORS[i % len(COLORS)], marker="o", markersize=3)
    ax.set_xlabel(warm_col)
    ax.set_ylabel(metric)
    ax.set_title(f"Warm vs Prewarm ({rate_col}={max_rate})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = metric.replace(" ", "_").replace("%", "pct").replace("/", "_")
    fig.savefig(os.path.join(output_dir, f"sweep_{safe_name}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: sweep_{safe_name}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument("csv_path", help="Path to sweep results.csv")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--metrics", default="drop_pct,cold_start_pct,mem_per_req_pct",
                        help="Comma-separated metrics to plot")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.csv_path)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    param_cols = _detect_params(df)
    metrics = [m.strip() for m in args.metrics.split(",")]

    print(f"Plotting {len(metrics)} metrics from {len(df)} rows")
    print(f"  params: {param_cols}")
    print(f"  metrics: {metrics}")
    print()

    for metric in metrics:
        if metric in df.columns:
            plot_metric(df, metric, param_cols, output_dir)
        else:
            print(f"  WARNING: metric '{metric}' not in CSV")

    print(f"\nAll plots in {output_dir}/")


if __name__ == "__main__":
    main()
