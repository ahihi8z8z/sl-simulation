"""Compare simulation logs across multiple runs.

Generates:
  1. Grouped bar chart: cold start %, avg mem util, avg cpu util
  2. Box plot: cold start latency distribution
  3. CDF: request latency comparison
  4. Per-run stacked area: container states over time

Usage:
    python tools/compare_logs.py logs/infer_our_sac_llm_9b logs/infer_our_sac_prewarm_llm_9b logs/infer_our_sac_warm_llm_9b
    python tools/compare_logs.py logs/infer_* --output-dir plots/comparison
    python tools/compare_logs.py logs/infer_* --labels "SAC,SAC-prewarm,SAC-warm"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_log(log_dir: str) -> dict:
    """Load summary.json, request_trace.csv, system_metrics.csv from a log dir."""
    import json
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


def plot_grouped_bar(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """Bar chart comparing cold start %, avg mem util, avg cpu util."""
    cold_pcts = []
    mem_utils = []
    cpu_utils = []

    for log in logs:
        s = log.get("summary", {})
        r = s.get("requests", {})
        cu = s.get("cluster_utilization", {})
        completed = max(r.get("completed", 1), 1)
        cold_pcts.append(r.get("cold_starts", 0) / completed * 100)
        mem_utils.append(cu.get("avg_memory_utilization", 0) * 100)
        cpu_utils.append(cu.get("avg_cpu_utilization", 0) * 100)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 5))
    ax.bar(x - width, cold_pcts, width, label="Cold Start %", color="#F44336")
    ax.bar(x, mem_utils, width, label="Avg Mem Util %", color="#2196F3")
    ax.bar(x + width, cpu_utils, width, label="Avg CPU Util %", color="#4CAF50")

    ax.set_ylabel("%")
    ax.set_title("Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_bar.png")


def plot_latency_boxplot(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """Box plot of cold start latency (execution_start - arrival for cold starts)."""
    cold_latencies = []
    for log in logs:
        trace = log.get("trace")
        if trace is not None and len(trace) > 0:
            cold = trace[trace["cold_start"] == True].copy()
            if len(cold) > 0:
                lat = cold["execution_start_time"] - cold["arrival_time"]
                cold_latencies.append(lat.values)
            else:
                cold_latencies.append(np.array([0.0]))
        else:
            cold_latencies.append(np.array([0.0]))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bp = ax.boxplot(cold_latencies, tick_labels=labels, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Cold Start Latency (s)")
    ax.set_title("Cold Start Latency Distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_boxplot.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_boxplot.png")


def plot_latency_cdf(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """CDF of request latency (execution_start - arrival) for all requests."""
    colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for log, label, color in zip(logs, labels, colors):
        trace = log.get("trace")
        if trace is None or len(trace) == 0:
            continue
        completed = trace[trace["status"] == "completed"].copy()
        if len(completed) == 0:
            continue
        lat = (completed["execution_start_time"] - completed["arrival_time"]).sort_values()
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat.values, cdf, label=label, color=color, linewidth=1.2)

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Request Latency CDF")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_cdf.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_cdf.png")


def plot_container_comparison(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """Line chart comparing container count over time across all runs."""
    colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(12, 5))

    for log, label, color in zip(logs, labels, colors):
        metrics = log.get("metrics")
        if metrics is None or len(metrics) == 0:
            continue

        time_hours = metrics["time"] / 3600.0
        total_col = "lifecycle.instances_total"
        if total_col in metrics.columns:
            ax.plot(time_hours, metrics[total_col].fillna(0),
                    label=label, color=color, linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Active Containers")
    ax.set_title("Container Count Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_containers.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_containers.png")


def main():
    parser = argparse.ArgumentParser(description="Compare simulation logs")
    parser.add_argument("log_dirs", nargs="+", help="Log directories to compare")
    parser.add_argument("--labels", default=None, help="Comma-separated labels")
    parser.add_argument("--output-dir", default="plots/comparison", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels.split(",") if args.labels else [os.path.basename(d) for d in args.log_dirs]
    logs = [load_log(d) for d in args.log_dirs]

    print(f"Comparing {len(logs)} runs: {labels}")
    plot_grouped_bar(logs, labels, args.output_dir)
    plot_latency_boxplot(logs, labels, args.output_dir)
    plot_latency_cdf(logs, labels, args.output_dir)
    plot_container_comparison(logs, labels, args.output_dir)
    print(f"\nAll plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
