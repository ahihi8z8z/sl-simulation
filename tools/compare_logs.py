"""Compare simulation logs across multiple runs.

Generates:
  1. comparison_metrics.png    — 2x2: cold+drop grouped bar, cold-start
                                 latency, RAM per request, avg power
  2. comparison_containers.png — stacked instances (prewarm/warm/running)
                                 per run, shared top legend
  3. comparison_pool_targets.png — pool_target lines (prewarm/warm) + idle
                                   window on twin axis, shared top legend
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
    """2x2 subplot:
       (0,0) cold-starts & drops — grouped bar, one group per algorithm
       (0,1) cold-start latency (s)
       (1,0) RAM per request (mem cost as % of cluster memory-seconds)
       (1,1) avg power (W)
    """
    cold_starts: list[float] = []
    drops: list[float] = []
    cold_lat: list[float] = []
    ram_per_req: list[float] = []
    power: list[float] = []

    for log in logs:
        s = log.get("summary", {})
        r = s.get("requests", {})
        cu = s.get("cluster_utilization", {})
        sim = s.get("simulation", {})
        eff = s.get("effective_resource_ratio", {})

        cold_starts.append(r.get("cold_starts", 0))
        drops.append(r.get("dropped", 0))

        trace = log.get("trace")
        if trace is not None and len(trace) > 0:
            completed = trace[trace["status"] == "completed"]
            cold = completed[completed["cold_start"] == True]
            n_completed = len(completed)
            if len(cold) > 0 and n_completed > 0:
                lat_sum = (cold["execution_start_time"] - cold["arrival_time"]).sum()
                cold_lat.append(lat_sum / n_completed * 1000.0)
            else:
                cold_lat.append(0)
        else:
            cold_lat.append(0)

        total_mem_sec = eff.get("total_memory_seconds", 0)
        cluster_mem_sec = sim.get("sim_end_time", 1) * cu.get("memory_total", 1)
        ram_per_req.append(total_mem_sec / cluster_mem_sec * 100 if cluster_mem_sec > 0 else 0)

        metrics_csv = log.get("metrics")
        if metrics_csv is not None and "cluster.power" in metrics_csv.columns:
            power.append(metrics_csv["cluster.power"].mean())
        else:
            power.append(0)

    n_runs = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(max(10, n_runs * 1.5 + 6), 8))
    colors = [COLORS[i % len(COLORS)] for i in range(n_runs)]
    x = np.arange(n_runs)

    # (0,0) grouped bar: cold starts vs drops
    ax = axes[0, 0]
    w = 0.38
    bars_c = ax.bar(x - w / 2, cold_starts, w, label="Cold Starts",
                    color="#F44336", alpha=0.85)
    bars_d = ax.bar(x + w / 2, drops, w, label="Dropped",
                    color="#2196F3", alpha=0.85)
    ax.bar_label(bars_c, fmt="%.0f", fontsize=7)
    ax.bar_label(bars_d, fmt="%.0f", fontsize=7)
    ax.set_title("Cold Starts & Dropped (requests)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Remaining panels: single bar, no error bars
    for ax, values, title, fmt in [
        (axes[0, 1], cold_lat, "Cold Latency per Served Request (ms)", "%.1f"),
        (axes[1, 0], ram_per_req, "RAM per request", "%.1f"),
        (axes[1, 1], power, "Avg Power (W)", "%.1f"),
    ]:
        bars = ax.bar(x, values, width=0.7, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt=fmt, fontsize=8, label_type="edge")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Performance Metrics Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_metrics.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_metrics.png")


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

        ax.set_ylabel("Instances")
        ax.set_title(label, fontsize=10)
        if ax == axes[0]:
            _legend_handles, _legend_labels = ax.get_legend_handles_labels()
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


def plot_pool_targets(logs: list[dict], labels: list[str], output_dir: str) -> None:
    """Line chart comparing pool_target decisions (prewarm + warm) over time.
    Idle window overlaid on twin y-axis; shared legend on top."""
    n_runs = len(logs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 3.5 * n_runs), sharex=True)
    if n_runs == 1:
        axes = [axes]

    _legend_handles, _legend_labels = [], []

    for ax, log, label in zip(axes, logs, labels):
        metrics = log.get("metrics")
        if metrics is None or len(metrics) == 0:
            continue

        m = metrics.copy()
        m["_hour"] = (m["time"] / 3600.0).astype(int)
        hourly = m.groupby("_hour").mean(numeric_only=True)
        time_hours = hourly.index.values.astype(float)

        s = log.get("summary", {})
        autoscaling = s.get("autoscaling", {})
        min_inst = 0
        for svc_data in autoscaling.values():
            min_inst = max(min_inst, svc_data.get("min_instances", 0))

        for state, color, ls in [("prewarm", "#2196F3", "-"), ("warm", "#F44336", "-")]:
            target_col = None
            for col in hourly.columns:
                if f"pool_target.{state}" in col:
                    target_col = col
                    break
            if target_col is not None:
                values = hourly[target_col].values.copy()
                if state == "warm":
                    values = values + min_inst
                ax.plot(time_hours, values, label=f"target({state})",
                        color=color, linewidth=1.5, linestyle=ls)

        # Idle window on right y-axis
        idle_col = None
        for col in hourly.columns:
            if "idle_timeout" in col:
                idle_col = col
                break
        ax2 = None
        if idle_col is not None:
            ax2 = ax.twinx()
            ax2.plot(time_hours, hourly[idle_col].values / 60.0,
                     color="#9C27B0", linewidth=1.2, alpha=0.8,
                     linestyle="--", label="idle window")
            ax2.set_ylabel("Idle window (min)", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7)

        ax.set_ylabel("Pool Target")
        ax.set_title(label, fontsize=10)
        if ax == axes[0]:
            lines1, labels1 = ax.get_legend_handles_labels()
            if ax2 is not None:
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
    fig.savefig(os.path.join(output_dir, "comparison_pool_targets.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_pool_targets.png")


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
    plot_container_comparison(logs, labels, args.output_dir, smooth=args.smooth)
    plot_pool_targets(logs, labels, args.output_dir)
    print(f"\nAll plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
