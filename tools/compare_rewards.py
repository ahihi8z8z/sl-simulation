"""Compare per-step reward breakdown across infer runs.

Each positional arg is a run dir containing episode_*/reward_breakdown.csv
(produced by rl_agent/infer.py). For a single-episode run, the file lives
directly under the run dir.

Generates two figures in --output-dir:
  1. reward_components_bar.png — per-group mean ±std per component (across
     all steps of all episodes).
  2. reward_components_curves.png — mean component value per step,
     averaged across episodes within each group; one subplot per component.

Usage:
    python tools/compare_rewards.py \\
        logs/rppo_full/infer logs/rppo_prewarm/infer logs/rppo_warm/infer \\
        --labels "full,prewarm,warm" --output-dir plots/rewards
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def _load_group(run_dir: str) -> list[pd.DataFrame]:
    """Return list of per-episode DataFrames found under run_dir."""
    paths = sorted(glob.glob(os.path.join(run_dir, "episode_*", "reward_breakdown.csv")))
    if not paths:
        # Single-episode run
        single = os.path.join(run_dir, "reward_breakdown.csv")
        if os.path.exists(single):
            paths = [single]
    return [pd.read_csv(p) for p in paths]


def _component_cols(dfs: list[pd.DataFrame]) -> list[str]:
    cols: set[str] = set()
    for df in dfs:
        cols.update(c for c in df.columns if c.startswith("rc_") or c == "reward")
    return sorted(cols)


def plot_bar(groups: list[list[pd.DataFrame]], labels: list[str],
             cols: list[str], output_dir: str) -> None:
    """Mean ± std per component, per group (across all steps × episodes)."""
    n_cols = len(cols)
    n_groups = len(groups)

    means = np.zeros((n_groups, n_cols))
    stds = np.zeros((n_groups, n_cols))
    for gi, dfs in enumerate(groups):
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        for ci, c in enumerate(cols):
            if c in combined.columns:
                vals = combined[c].astype(float).to_numpy()
                means[gi, ci] = float(np.mean(vals))
                stds[gi, ci] = float(np.std(vals))

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.5), 5))
    x = np.arange(n_cols)
    w = 0.8 / max(n_groups, 1)
    for gi, label in enumerate(labels):
        offset = (gi - (n_groups - 1) / 2) * w
        ax.bar(x + offset, means[gi], w, yerr=stds[gi],
               label=label, color=COLORS[gi % len(COLORS)],
               alpha=0.85, capsize=3, error_kw={"elinewidth": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean per-step value")
    ax.set_title("Reward components — mean ± std across all steps")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "reward_components_bar.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def _per_step_mean(dfs: list[pd.DataFrame], col: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean & std across episodes at each step index for one component."""
    if not dfs:
        return np.array([]), np.array([])
    max_len = max(len(df) for df in dfs)
    stacked = np.full((len(dfs), max_len), np.nan)
    for i, df in enumerate(dfs):
        if col in df.columns:
            vals = df[col].astype(float).to_numpy()
            stacked[i, :len(vals)] = vals
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return mean, std


def plot_curves(groups: list[list[pd.DataFrame]], labels: list[str],
                cols: list[str], output_dir: str, smooth: int = 1) -> None:
    """Per-step mean curves (averaged across episodes within group)."""
    n = len(cols)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.2 * nrows),
                             squeeze=False)

    for ci, col in enumerate(cols):
        ax = axes[ci // ncols][ci % ncols]
        for gi, (dfs, label) in enumerate(zip(groups, labels)):
            mean, _ = _per_step_mean(dfs, col)
            if mean.size == 0:
                continue
            if smooth > 1 and mean.size >= smooth:
                kernel = np.ones(smooth) / smooth
                mean = np.convolve(mean, kernel, mode="same")
            ax.plot(mean, label=label, color=COLORS[gi % len(COLORS)],
                    linewidth=1.2, alpha=0.9)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("step")
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
        ax.grid(alpha=0.3)
        if ci == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle("Reward components over an episode (mean across episodes)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "reward_components_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="Inference run dirs")
    ap.add_argument("--labels", default=None,
                    help="Comma-separated labels (default: dir basenames)")
    ap.add_argument("--output-dir", default="plots/rewards")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Moving-average window for curves (default 1 = none)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
    else:
        labels = [os.path.basename(d.rstrip("/")) for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise SystemExit(f"--labels count ({len(labels)}) != run_dirs count ({len(args.run_dirs)})")

    groups = [_load_group(d) for d in args.run_dirs]
    for label, dfs in zip(labels, groups):
        print(f"  {label}: {len(dfs)} episode(s)")
        if not dfs:
            print(f"    WARNING: no reward_breakdown.csv found in this run dir")

    cols = _component_cols([df for dfs in groups for df in dfs])
    if not cols:
        raise SystemExit("No reward components found across the given runs.")

    print(f"Components: {cols}")
    plot_bar(groups, labels, cols, args.output_dir)
    plot_curves(groups, labels, cols, args.output_dir, smooth=args.smooth)


if __name__ == "__main__":
    main()
