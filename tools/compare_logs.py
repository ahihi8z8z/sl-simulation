"""Compare simulation logs across multiple runs.

Generates:
  1. Grouped bar chart (groups = metrics): cold start %, mem util %, cpu util %
  2. Latency bar chart with error bars (mean ± std)
  3. Container count line chart (smoothed)

Usage:
    python tools/compare_logs.py logs/infer_our_sac_llm_9b logs/infer_our_sac_prewarm_llm_9b
    python tools/compare_logs.py logs/infer_* --labels "SAC,SAC-prewarm,SAC-warm"
    python tools/compare_logs.py logs/infer_* --output-dir plots/comparison --smooth 10
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]


def load_log(log_dir: str) -> dict:
    """Load summary.json, request_trace.csv, system_metrics.csv from a log dir."""
    data = {"name": os.path.basename(log_dir)}

    summary_path = os.path.join(log_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data["summary"] = json.load(f)

    trace_path = os.path.join(log_dir, "request_trace.csv")
    if os.path.exists(trace_path):
        data["trace"] = pd.read_csv(trace_path)

    metrics_path = os.path.join(log_dir, "system_metrics.csv")
    if os.path.exists(metrics_path):
        data["metrics"] = pd.read_csv(metrics_path)

    return data


def plot_metrics_bar(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """Grouped bar chart — groups are metrics, bars are runs."""
    metrics_data = {"Cold Start %": [], "Avg Mem Util %": [], "Avg CPU Util %": []}

    for log in logs:
        s = log.get("summary", {})
        r = s.get("requests", {})
        cu = s.get("cluster_utilization", {})
        completed = max(r.get("completed", 1), 1)
        metrics_data["Cold Start %"].append(r.get("cold_starts", 0) / completed * 100)
        metrics_data["Avg Mem Util %"].append(cu.get("avg_memory_utilization", 0) * 100)
        metrics_data["Avg CPU Util %"].append(cu.get("avg_cpu_utilization", 0) * 100)

    metric_names = list(metrics_data.keys())
    x = np.arange(len(metric_names))
    n_runs = len(labels)
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 3), 5))
    for i, (label, color) in enumerate(zip(labels, COLORS)):
        values = [metrics_data[m][i] for m in metric_names]
        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", fontsize=7)

    ax.set_ylabel("%")
    ax.set_title("Performance Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_metrics.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_metrics.png")


def plot_latency_bar(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """Bar chart with error bars: cold-start latency mean ± std."""
    cold_means = []
    cold_stds = []

    for log in logs:
        trace = log.get("trace")
        if trace is not None and len(trace) > 0:
            completed = trace[trace["status"] == "completed"]
            cold = completed[completed["cold_start"] == True]
            if len(cold) > 0:
                cold_lat = cold["execution_start_time"] - cold["arrival_time"]
                cold_means.append(cold_lat.mean())
                cold_stds.append(cold_lat.std())
            else:
                cold_means.append(0)
                cold_stds.append(0)
        else:
            cold_means.append(0)
            cold_stds.append(0)

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    bars = ax.bar(x, cold_means, width, yerr=cold_stds,
                  color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
                  alpha=0.85, capsize=3)
    ax.bar_label(bars, fmt="%.2f", fontsize=7)

    ax.set_ylabel("Latency (s)")
    ax.set_title("Cold Start Latency (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_latency.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_latency.png")


def plot_container_comparison(logs: list[dict], labels: list[str], output_dir: str,
                              smooth: int = 5) -> None:
    """Stacked area chart per run: prewarm vs warm pool instances."""
    n_runs = len(logs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 3.5 * n_runs), sharex=True)
    if n_runs == 1:
        axes = [axes]

    state_colors = {"prewarm": "#2196F3", "warm": "#F44336", "running": "#FF9800"}
    _legend_handles, _legend_labels = [], []

    for ax, log, label in zip(axes, logs, labels):
        metrics = log.get("metrics")
        if metrics is None or len(metrics) == 0:
            continue

        # Resample to hourly averages
        metrics = metrics.copy()
        metrics["_hour"] = (metrics["time"] / 3600.0).astype(int)
        hourly = metrics.groupby("_hour").mean(numeric_only=True)
        time_hours = hourly.index.values.astype(float)

        states = []
        values_list = []
        colors = []
        for state in ("prewarm", "warm", "running"):
            col = f"lifecycle.instances_{state}"
            if col in hourly.columns:
                states.append(state)
                values_list.append(hourly[col].fillna(0).values)
                colors.append(state_colors.get(state, "#999999"))

        if values_list:
            ax.stackplot(time_hours, *values_list, labels=states, colors=colors, alpha=0.85)

        # Idle window on right y-axis
        idle_col = None
        for col in hourly.columns:
            if "idle_timeout" in col:
                idle_col = col
                break
        if idle_col is not None:
            ax2 = ax.twinx()
            ax2.plot(time_hours, hourly[idle_col].values / 60.0,
                     color="#9C27B0", linewidth=1.2, alpha=0.8, linestyle="--", label="idle window")
            ax2.set_ylabel("Idle window (min)", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7)
            states.append("idle window")

        ax.set_ylabel("Instances")
        ax.set_title(label, fontsize=10)
        # Collect handles for shared legend (only from first subplot)
        if ax == axes[0]:
            lines1, labels1 = ax.get_legend_handles_labels()
            if idle_col is not None:
                lines2, labels2 = ax2.get_legend_handles_labels()
                _legend_handles = lines1 + lines2
                _legend_labels = labels1 + labels2
            else:
                _legend_handles = lines1
                _legend_labels = labels1
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (hours)")

    if _legend_handles:
        fig.legend(_legend_handles, _legend_labels, loc="upper center",
                   ncol=len(_legend_labels), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(os.path.join(output_dir, "comparison_containers.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_containers.png")


def plot_idle_window(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """Line chart comparing idle timeout window over time (hourly average)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for log, label, color in zip(logs, labels, COLORS):
        metrics = log.get("metrics")
        if metrics is None or len(metrics) == 0:
            continue

        # Find idle_timeout column (may have service prefix)
        idle_col = None
        for col in metrics.columns:
            if "idle_timeout" in col:
                idle_col = col
                break
        if idle_col is None:
            continue

        m = metrics.copy()
        m["_hour"] = (m["time"] / 3600.0).astype(int)
        hourly = m.groupby("_hour")[idle_col].mean()

        ax.plot(hourly.index, hourly.values / 60.0, label=label, color=color,
                linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Idle Timeout (minutes)")
    ax.set_title("Idle Window Comparison (hourly avg)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_idle_window.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_idle_window.png")


def plot_latency_cdf(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """CDF of cold-start request latency (execution_start - arrival)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for log, label, color in zip(logs, labels, COLORS):
        trace = log.get("trace")
        if trace is None or len(trace) == 0:
            continue
        completed = trace[trace["status"] == "completed"]
        cold = completed[completed["cold_start"] == True]
        if len(cold) == 0:
            continue
        lat = (cold["execution_start_time"] - cold["arrival_time"]).sort_values()
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat.values, cdf, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Cold Start Latency (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Cold Start Latency CDF")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_latency_cdf.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_latency_cdf.png")


def main():
    parser = argparse.ArgumentParser(description="Compare simulation logs")
    parser.add_argument("log_dirs", nargs="+", help="Log directories to compare")
    parser.add_argument("--labels", default=None, help="Comma-separated labels")
    parser.add_argument("--output-dir", default="plots/comparison", help="Output directory")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window for container chart (default: 5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels.split(",") if args.labels else [os.path.basename(d) for d in args.log_dirs]
    logs = [load_log(d) for d in args.log_dirs]

    print(f"Comparing {len(logs)} runs: {labels}")
    plot_metrics_bar(logs, labels, args.output_dir)
    plot_latency_bar(logs, labels, args.output_dir)
    plot_container_comparison(logs, labels, args.output_dir, smooth=args.smooth)
    plot_idle_window(logs, labels, args.output_dir)
    plot_latency_cdf(logs, labels, args.output_dir)
    print(f"\nAll plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
