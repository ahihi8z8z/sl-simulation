"""Compare simulation logs across multiple runs.

Each positional arg is a log "group". A group is either:
  - a plain log dir (one run) — loaded as a 1-seed group, or
  - a parent dir containing seed_*/ subdirs (multi-seed infer output from
    infer_all.py) — aggregated as mean (±std on bar plots) across seeds.

Generates:
  1. comparison_metrics.png    — 2x2: cold/drop grouped bar, avg latency
                                 (over all completed), memory per request
                                 (MB·s), energy per request (J). Multi-seed
                                 groups get std error bars.
  2. comparison_containers.png — stacked instances (prewarm/warm/running)
                                 per group (per-hour mean across seeds).
  3. comparison_pool_targets.png — pool_target lines + idle window on twin
                                   axis (per-hour mean across seeds).
  4. comparison_latency_cdf.png — latency CDF per group (latencies concat
                                   across seeds, completed & > 0).
Usage:
    python tools/compare_logs.py logs/our_sac/infer logs/our_ppo/infer
    python tools/compare_logs.py logs/*/infer --labels "SAC,SAC-prewarm,SAC-warm"
    python tools/compare_logs.py logs/infer_one_shot logs/our_sac/infer --output-dir plots/comparison
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]


def _load_seed(seed_dir: str) -> dict:
    data = {}
    summary_path = os.path.join(seed_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data["summary"] = json.load(f)
    trace_path = os.path.join(seed_dir, "request_trace.csv")
    if os.path.exists(trace_path):
        data["trace"] = pd.read_csv(trace_path)
    metrics_path = os.path.join(seed_dir, "system_metrics.csv")
    if os.path.exists(metrics_path):
        data["metrics"] = pd.read_csv(metrics_path)
    return data


def load_log(log_dir: str) -> dict:
    """Load a log "group" (possibly multi-seed).

    If log_dir contains seed_*/ subdirs, each is loaded and returned in
    .seeds as a list. Otherwise log_dir itself is treated as a single
    seed (list of length 1). Callers always work with .seeds.
    """
    seed_subdirs = sorted(
        glob.glob(os.path.join(log_dir, "seed_*"))
        + glob.glob(os.path.join(log_dir, "episode_*"))
    )
    seed_subdirs = [d for d in seed_subdirs if os.path.isdir(d)]

    if seed_subdirs:
        seeds = [_load_seed(d) for d in seed_subdirs]
    else:
        seeds = [_load_seed(log_dir)]

    return {
        "name": os.path.basename(log_dir.rstrip("/")),
        "seeds": seeds,
        "n_seeds": len(seeds),
    }


def _seed_scalar_metrics(seed: dict, service: str | None = None) -> dict:
    """Extract per-seed scalar metrics for the bar plot.

    If ``service`` is given, cold/drop/latency are derived from trace filtered
    to that service. mem_per_req and power_per_req stay cluster-wide (not
    available per-service in summary/system_metrics).
    """
    s = seed.get("summary", {})
    r = s.get("requests", {})
    eff = s.get("effective_resource_ratio", {})

    trace = seed.get("trace")
    if service is not None:
        if trace is not None and len(trace) > 0:
            t = trace[trace["service_id"] == service]
            cold_starts = float(t["cold_start"].sum()) if "cold_start" in t.columns else 0.0
            drops = float((t["status"] == "dropped").sum())
            completed = t[t["status"] == "completed"]
            latencies = completed["execution_start_time"] - completed["arrival_time"]
            avg_lat = float(latencies.mean() * 1000.0) if len(latencies) > 0 else 0.0
        else:
            cold_starts, drops, avg_lat = 0.0, 0.0, 0.0
    else:
        cold_starts = float(r.get("cold_starts", 0))
        drops = float(r.get("dropped", 0))
        if trace is not None and len(trace) > 0:
            completed = trace[trace["status"] == "completed"]
            latencies = completed["execution_start_time"] - completed["arrival_time"]
            avg_lat = float(latencies.mean() * 1000.0) if len(latencies) > 0 else 0.0
        else:
            avg_lat = 0.0

    mem_per_req = float(eff.get("memory_per_request", 0.0))

    # Energy per completed request (J/req): integrate cluster.power over the
    # run, divide by completed count. Approximated as mean(power) * duration.
    metrics_csv = seed.get("metrics")
    power_per_req = 0.0
    if (metrics_csv is not None
            and "cluster.power" in metrics_csv.columns
            and "time" in metrics_csv.columns
            and len(metrics_csv) > 1):
        completed_n = float(r.get("completed", 0))
        if completed_n > 0:
            mean_w = float(pd.to_numeric(metrics_csv["cluster.power"], errors="coerce").mean())
            t = pd.to_numeric(metrics_csv["time"], errors="coerce")
            duration = float(t.max() - t.min())
            power_per_req = mean_w * duration / completed_n

    return {
        "cold_starts": cold_starts,
        "drops": drops,
        "avg_lat": avg_lat,
        "mem_per_req": mem_per_req,
        "power_per_req": power_per_req,
    }


def _aggregate_group(group: dict, keys: list[str], service: str | None = None) -> dict[str, tuple[float, float]]:
    """Return {key: (mean, std)} across seeds in this group."""
    per_seed = [_seed_scalar_metrics(s, service=service) for s in group["seeds"]]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std(ddof=0)) if len(vals) > 1 else 0.0)
    return out


def plot_metrics_bar(groups: list[dict], labels: list[str], output_dir: str,
                     service: str | None = None) -> None:
    """2x2 bar plot with mean±std across seeds per group.

    If ``service`` is set, cold/drop/latency are filtered to that service.
    The mem_per_req and power_per_req panels show cluster-wide values
    (per-service breakdown not tracked in summary / system_metrics).
    """
    keys = ["cold_starts", "drops", "avg_lat", "mem_per_req", "power_per_req"]
    agg = [_aggregate_group(g, keys, service=service) for g in groups]

    def vals(k): return [a[k][0] for a in agg], [a[k][1] for a in agg]

    cold_m, cold_s = vals("cold_starts")
    drop_m, drop_s = vals("drops")
    lat_m, lat_s   = vals("avg_lat")
    mem_m, mem_s   = vals("mem_per_req")
    pwr_m, pwr_s   = vals("power_per_req")

    n_runs = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(max(10, n_runs * 1.5 + 6), 8))
    colors = [COLORS[i % len(COLORS)] for i in range(n_runs)]
    x = np.arange(n_runs)

    # (0,0) grouped bar: cold starts / drops
    ax = axes[0, 0]
    w = 0.4
    bars_c = ax.bar(x - w / 2, cold_m, w, yerr=cold_s, color=colors, alpha=0.85,
                    edgecolor="black", linewidth=0.5, capsize=3,
                    error_kw={"elinewidth": 0.8})
    bars_d = ax.bar(x + w / 2, drop_m, w, yerr=drop_s, color=colors, alpha=0.85,
                    edgecolor="black", linewidth=0.5, hatch="//", capsize=3,
                    error_kw={"elinewidth": 0.8})
    ax.bar_label(bars_c, fmt="%.0f", fontsize=7)
    ax.bar_label(bars_d, fmt="%.0f", fontsize=7)
    ax.set_title("Cold / Dropped (requests)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="black", label="Cold Start"),
        Patch(facecolor="lightgray", edgecolor="black", hatch="//", label="Dropped"),
    ]
    ax.legend(handles=legend_handles, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Remaining panels: single bar with yerr
    for ax, means, stds, title, fmt in [
        (axes[0, 1], lat_m, lat_s, "Avg Latency (ms)", "%.1f"),
        (axes[1, 0], mem_m, mem_s, "Memory per request (MB·s)", "%.0f"),
        (axes[1, 1], pwr_m, pwr_s, "Energy per request (J)", "%.2f"),
    ]:
        bars = ax.bar(x, means, width=0.7, yerr=stds, color=colors, alpha=0.85,
                      capsize=4, error_kw={"elinewidth": 0.9})
        ax.bar_label(bars, fmt=fmt, fontsize=8, label_type="edge")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    n_seeds_str = ", ".join(f"{l}: n={g['n_seeds']}" for l, g in zip(labels, groups))
    scope = f" — service={service}" if service else ""
    fig.suptitle(f"Performance Metrics Comparison{scope} ({n_seeds_str})", fontsize=12)
    plt.tight_layout()
    fname = f"comparison_metrics_svc_{service}.png" if service else "comparison_metrics.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def _pick_bucket(seeds: list[dict]) -> tuple[float, str]:
    """Auto-pick a bucket size based on the longest seed's time span.

    Returns (bucket_seconds, x_unit_label).
    Heuristic: aim for ~30-300 buckets so plots have visible resolution.
    """
    max_span = 0.0
    for s in seeds:
        m = s.get("metrics")
        if m is None or len(m) == 0:
            continue
        max_span = max(max_span, float(m["time"].max() - m["time"].min()))
    if max_span >= 7200:        # >= 2h → bucket = 1h
        return 3600.0, "hours"
    if max_span >= 120:         # >= 2min → bucket = 1min
        return 60.0, "minutes"
    return 1.0, "seconds"        # very short sims


def _hourly_mean_across_seeds(seeds: list[dict], columns: list[str]) -> tuple[pd.DataFrame, str] | None:
    """Resample each seed's metrics into time buckets (auto-sized to the run
    length), align on shared bucket index, then mean across seeds.

    Returns (DataFrame indexed by bucket center in the chosen unit,
             x-axis unit label) — or None if no usable data.
    """
    bucket_sec, unit = _pick_bucket(seeds)
    per_seed: list[pd.DataFrame] = []
    for s in seeds:
        m = s.get("metrics")
        if m is None or len(m) == 0:
            continue
        m = m.copy()
        m["_bucket"] = (m["time"] / bucket_sec).astype(int)
        binned = m.groupby("_bucket").mean(numeric_only=True)
        keep = [c for c in columns if c in binned.columns]
        if keep:
            per_seed.append(binned[keep])
    if not per_seed:
        return None
    # Align on union of buckets, mean across seeds (NaN-safe)
    stacked = pd.concat(per_seed, axis=0)
    return stacked.groupby(stacked.index).mean(), unit


def plot_container_comparison(groups: list[dict], labels: list[str], output_dir: str,
                              smooth: int = 5, service: str | None = None) -> None:
    """Container chart per group, per-bucket mean across seeds.

    Cluster-wide (service=None): stacked area of prewarm/warm/running.
    Per-service: line plot of running and total (state breakdown not
    available per-service in system_metrics).
    """
    n_runs = len(groups)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 3.5 * n_runs), sharex=True)
    if n_runs == 1:
        axes = [axes]

    if service is None:
        state_colors = {"prewarm": "#2196F3", "warm": "#F44336", "running": "#FF9800"}
        wanted_cols = [f"lifecycle.instances_{s}" for s in ("prewarm", "warm", "running")]
    else:
        wanted_cols = [f"lifecycle.{service}.instances_running",
                       f"lifecycle.{service}.instances_total"]

    _legend_handles, _legend_labels = [], []
    x_unit = "hours"

    for ax, group, label in zip(axes, groups, labels):
        result = _hourly_mean_across_seeds(group["seeds"], wanted_cols)
        if result is None:
            continue
        hourly, x_unit = result
        time_hours = hourly.index.values.astype(float)

        if service is None:
            states, values_list, colors = [], [], []
            for state in ("prewarm", "warm", "running"):
                col = f"lifecycle.instances_{state}"
                if col in hourly.columns:
                    states.append(state)
                    values_list.append(hourly[col].fillna(0).values)
                    colors.append(state_colors.get(state, "#999999"))
            if values_list:
                ax.stackplot(time_hours, *values_list, labels=states, colors=colors, alpha=0.85)
        else:
            for col, name, color in [
                (f"lifecycle.{service}.instances_total", "total", "#2196F3"),
                (f"lifecycle.{service}.instances_running", "running", "#FF9800"),
            ]:
                if col in hourly.columns:
                    ax.plot(time_hours, hourly[col].fillna(0).values,
                            label=name, color=color, linewidth=1.5)

        title = f"{label} (n={group['n_seeds']})" if group["n_seeds"] > 1 else label
        ax.set_ylabel("Instances")
        ax.set_title(title, fontsize=10)
        if ax == axes[0]:
            _legend_handles, _legend_labels = ax.get_legend_handles_labels()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel(f"Time ({x_unit})")

    if _legend_handles:
        fig.legend(_legend_handles, _legend_labels, loc="upper center",
                   ncol=len(_legend_labels), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
    if service:
        fig.suptitle(f"Containers — service={service}", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fname = f"comparison_containers_svc_{service}.png" if service else "comparison_containers.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_pool_targets(groups: list[dict], labels: list[str], output_dir: str,
                      service: str | None = None) -> None:
    """Line chart per group: pool_target lines + idle window (twin axis).

    Cluster-wide (service=None): finds first matching pool_target.<state>
    column (any service). Per-service: filters to ``autoscaling.<service>.*``.
    """
    n_runs = len(groups)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 3.5 * n_runs), sharex=True)
    if n_runs == 1:
        axes = [axes]

    _legend_handles, _legend_labels = [], []
    x_unit = "hours"

    for ax, group, label in zip(axes, groups, labels):
        # Discover available columns from any seed's metrics
        sample_metrics = next((s.get("metrics") for s in group["seeds"]
                               if s.get("metrics") is not None), None)
        if sample_metrics is None:
            continue
        all_cols = list(sample_metrics.columns)

        if service is not None:
            svc_prefix = f"autoscaling.{service}."
            cols_in_scope = [c for c in all_cols if c.startswith(svc_prefix)]
        else:
            cols_in_scope = all_cols

        target_cols = {}
        for state in ("prewarm", "warm"):
            for col in cols_in_scope:
                if f"pool_target.{state}" in col:
                    target_cols[state] = col
                    break
        idle_col = next((c for c in cols_in_scope if "idle_timeout" in c), None)

        wanted = list(target_cols.values()) + ([idle_col] if idle_col else [])
        result = _hourly_mean_across_seeds(group["seeds"], wanted)
        if result is None:
            continue
        hourly, x_unit = result
        time_hours = hourly.index.values.astype(float)

        # min_instances from first seed's summary (configuration, not seed-dependent)
        min_inst = 0
        first_summary = next((s.get("summary", {}) for s in group["seeds"]
                              if s.get("summary")), {})
        for svc_data in first_summary.get("autoscaling", {}).values():
            min_inst = max(min_inst, svc_data.get("min_instances", 0))

        for state, color in [("prewarm", "#2196F3"), ("warm", "#F44336")]:
            col = target_cols.get(state)
            if col is not None and col in hourly.columns:
                values = hourly[col].values.copy()
                if state == "warm":
                    values = values + min_inst
                ax.plot(time_hours, values, label=f"target({state})",
                        color=color, linewidth=1.5)

        ax2 = None
        if idle_col is not None and idle_col in hourly.columns:
            ax2 = ax.twinx()
            ax2.plot(time_hours, hourly[idle_col].values / 60.0,
                     color="#9C27B0", linewidth=1.2, alpha=0.8,
                     linestyle="--", label="idle window")
            ax2.set_ylabel("Idle window (min)", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7)

        title = f"{label} (n={group['n_seeds']})" if group["n_seeds"] > 1 else label
        ax.set_ylabel("Pool Target")
        ax.set_title(title, fontsize=10)
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

    axes[-1].set_xlabel(f"Time ({x_unit})")

    if _legend_handles:
        fig.legend(_legend_handles, _legend_labels, loc="upper center",
                   ncol=len(_legend_labels), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
    if service:
        fig.suptitle(f"Pool targets — service={service}", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    fname = f"comparison_pool_targets_svc_{service}.png" if service else "comparison_pool_targets.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_latency_cdf(groups: list[dict], labels: list[str], output_dir: str,
                     service: str | None = None) -> None:
    """CDF of request latency per group, optionally filtered by service."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    for group, label, color in zip(groups, labels, colors):
        all_lat: list[np.ndarray] = []
        for seed in group["seeds"]:
            trace = seed.get("trace")
            if trace is None or len(trace) == 0:
                continue
            t = trace if service is None else trace[trace["service_id"] == service]
            completed = t[t["status"] == "completed"]
            latencies = (completed["execution_start_time"] - completed["arrival_time"]) * 1000.0
            latencies = latencies[latencies > 0].to_numpy()
            if len(latencies) > 0:
                all_lat.append(latencies)
        if not all_lat:
            continue
        combined = np.concatenate(all_lat)
        sorted_lat = np.sort(combined)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
        legend = (f"{label} (n={len(sorted_lat)}, seeds={group['n_seeds']})"
                  if group["n_seeds"] > 1 else f"{label} (n={len(sorted_lat)})")
        ax.plot(sorted_lat, cdf, color=color, linewidth=1.5, label=legend)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    title = "Distribution of Cold Start Latency"
    if service:
        title += f" — service={service}"
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fname = f"comparison_latency_cdf_svc_{service}.png" if service else "comparison_latency_cdf.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def discover_services(groups: list[dict]) -> list[str]:
    """Union of service IDs found in any seed's trace or system_metrics."""
    found: set[str] = set()
    import re
    pat = re.compile(r"^lifecycle\.([^.]+)\.instances_total$")
    for g in groups:
        for seed in g["seeds"]:
            trace = seed.get("trace")
            if trace is not None and "service_id" in trace.columns:
                found.update(map(str, trace["service_id"].dropna().unique()))
            metrics = seed.get("metrics")
            if metrics is not None:
                for c in metrics.columns:
                    m = pat.match(c)
                    if m:
                        found.add(m.group(1))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description="Compare simulation logs")
    parser.add_argument("log_dirs", nargs="+",
                        help="Log dirs (plain dir or dir containing seed_*/)")
    parser.add_argument("--labels", default=None, help="Comma-separated labels")
    parser.add_argument("--output-dir", default="plots/comparison", help="Output directory")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Smoothing window for container chart (default: 5)")
    parser.add_argument("--services", default=None,
                        help="Comma-separated services for per-service plots "
                             "(default: auto-detect; 'none' to skip)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels.split(",") if args.labels else [os.path.basename(d.rstrip("/")) for d in args.log_dirs]
    groups = [load_log(d) for d in args.log_dirs]

    seeds_summary = ", ".join(f"{l}={g['n_seeds']}" for l, g in zip(labels, groups))
    print(f"Comparing {len(groups)} groups: {labels} (seeds: {seeds_summary})")

    # Cluster-wide
    plot_metrics_bar(groups, labels, args.output_dir)
    plot_container_comparison(groups, labels, args.output_dir, smooth=args.smooth)
    plot_pool_targets(groups, labels, args.output_dir)
    plot_latency_cdf(groups, labels, args.output_dir)

    # Per-service
    if args.services == "none":
        services: list[str] = []
    elif args.services:
        services = [s.strip() for s in args.services.split(",") if s.strip()]
    else:
        services = discover_services(groups)
    if services:
        print(f"\nPer-service plots: {services}")
    for svc in services:
        plot_metrics_bar(groups, labels, args.output_dir, service=svc)
        plot_container_comparison(groups, labels, args.output_dir,
                                  smooth=args.smooth, service=svc)
        plot_pool_targets(groups, labels, args.output_dir, service=svc)
        plot_latency_cdf(groups, labels, args.output_dir, service=svc)

    print(f"\nAll plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
