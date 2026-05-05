"""Run default_baseline simulation for every trace in datasets/traffic_pattern/azure/
and plot metrics per trace.

Supports sweeping over services[0].min_instances to produce line charts
showing how each metric changes as min_instances varies.

Uses experimental/azure_3b4942e69dd3 as the config template.

Usage:
    # bar chart across traces (single min_instances value):
    python tools/benchmark_datanew.py

    # line chart: sweep min_instances=[5,10,20,50] across all traces:
    python tools/benchmark_datanew.py --min-instances 5,10,20,50

    # also accepts a range: start:stop:step
    python tools/benchmark_datanew.py --min-instances 5:55:10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATANEW_DIR = "datasets/traffic_pattern/dre_scale"
TEMPLATE_DIR = "experimental/rppo_test"
DEFAULT_OUT   = "plots/rppo_benchmark"
TYPE = "trace"
SCALE = 20
START_MINUTE = 0
PATTERN = "*dre_scale_replay_seed29.csv"

BASELINE_OVERRIDES = {
    # "services[0].min_instances": 20,
    "services[0].autoscaling_defaults.idle_timeout": 60,
    "cluster.nodes[0].count": 10,
    "simulation.duration": 10500,
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config(trace_path: str, scale: int, overrides: dict | None = None) -> dict:
    from tools.config_merge import load_merged_config, apply_overrides

    base = os.path.join(TEMPLATE_DIR, "base_config.json")
    cfg = load_merged_config(base)

    import pandas as pd
    df = pd.read_csv(trace_path)
    if "minute" in df.columns:
        span = float(df["minute"].max() - df["minute"].min()) * 60.0
    elif "timestamp" in df.columns:
        span = float(df["timestamp"].max() - df["timestamp"].min())
    else:
        span = 3600.0
    span = max(span, 60.0)
    duration = span * 1.05 + 120

    cfg["simulation"]["duration"] = round(duration)
    cfg["services"][0]["workload"] = {
        "generator": TYPE,
        "trace_path": trace_path,
        "scale": SCALE,
        "start_minute": START_MINUTE,
    }

    merged = {**BASELINE_OVERRIDES, **(overrides or {})}
    apply_overrides(cfg, merged)
    return cfg


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(trace_path: str, run_dir: str, scale: int, overrides: dict) -> tuple[str, dict, float]:
    t0 = time.monotonic()
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from serverless_sim.core.config.loader import load_config_from_dict
        from serverless_sim.core.logging.logger_factory import create_logger
        from serverless_sim.core.simulation.sim_builder import SimulationBuilder
        from serverless_sim.core.simulation.sim_engine import SimulationEngine

        cfg = _build_config(trace_path, scale, overrides)
        os.makedirs(run_dir, exist_ok=True)

        validated = load_config_from_dict(cfg)
        logger = create_logger(
            module_name=f"sim.{Path(trace_path).stem}",
            run_dir=run_dir,
            mode="file",
            level="WARNING",
        )
        builder = SimulationBuilder()
        ctx = builder.build(validated, run_dir, logger)
        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        summary_path = os.path.join(run_dir, "summary.json")
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        return trace_path, summary, time.monotonic() - t0
    except Exception as e:
        return trace_path, {"error": str(e)}, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def _extract(trace_path: str, run_dir: str, summary: dict) -> dict:
    req  = summary.get("requests", {})
    eff  = summary.get("effective_resource_ratio", {})
    label = Path(trace_path).stem

    power = 0.0
    metrics_path = os.path.join(run_dir, "system_metrics.csv")
    if os.path.exists(metrics_path):
        import pandas as pd
        m = pd.read_csv(metrics_path)
        if "cluster.power" in m.columns:
            power = float(m["cluster.power"].mean())

    return {
        "label":       label,
        "cold_starts": float(req.get("cold_starts", 0)),
        "drops":       float(req.get("dropped", 0)),
        "total":       float(req.get("total", 0)),
        "avg_lat":     float(summary.get("latency", {}).get("mean", 0.0)) * 1000,
        "mem_per_req": float(eff.get("memory_per_request", 0.0)),
        "power":       power,
        "drop_pct":    float(summary.get("rates", {}).get("drop_rate_pct", 0.0)),
        "cold_pct":    float(summary.get("rates", {}).get("cold_start_rate_pct", 0.0)),
    }


# ---------------------------------------------------------------------------
# Plot: bar chart (single min_instances run)
# ---------------------------------------------------------------------------

def _plot_bar(records: list[dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = sorted(records, key=lambda r: -r["cold_pct"])
    labels  = [r["label"] for r in records]
    n = len(labels)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(max(14, n * 0.35 + 4), 10))

    def bar(ax, vals, title, ylabel, color):
        ax.bar(x, vals, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    bar(axes[0, 0], [r["drop_pct"]    for r in records], "Drop rate",          "%",     "#e05c5c")
    bar(axes[0, 1], [r["cold_pct"]    for r in records], "Cold start rate",    "%",     "#5c8ee0")
    bar(axes[0, 2], [r["total"]       for r in records], "Total requests",     "count", "#7f8c8d")
    bar(axes[1, 0], [r["avg_lat"]     for r in records], "Avg latency",        "ms",    "#8e5ce0")
    bar(axes[1, 1], [r["mem_per_req"] for r in records], "Memory per request", "MB·s",  "#27ae60")
    bar(axes[1, 2], [r["power"]       for r in records], "Avg power",          "W",     "#e08c5c")

    fig.suptitle(
        f"default_baseline — azure ({n} traces, sorted by cold start rate)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "benchmark_datanew.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    import csv
    csv_path = os.path.join(out_dir, "benchmark_datanew.csv")
    with open(csv_path, "w", newline="") as f:
        fields = ["label","total","drops","drop_pct","cold_starts","cold_pct","avg_lat","mem_per_req","power"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: round(r[k], 4) if isinstance(r[k], float) else r[k] for k in fields})
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Plot: line chart (min_instances sweep)
# ---------------------------------------------------------------------------

METRICS = [
    ("drop_pct",    "Drop rate (%)",          "#e05c5c"),
    ("cold_pct",    "Cold start rate (%)",     "#5c8ee0"),
    ("avg_lat",     "Avg latency (ms)",        "#8e5ce0"),
    ("mem_per_req", "Memory per request (MB·s)","#27ae60"),
    ("power",       "Avg power (W)",           "#e08c5c"),
    ("total",       "Total requests",          "#7f8c8d"),
]


def _plot_sweep(
    sweep_results: dict[int, list[dict]],
    out_dir: str,
) -> None:
    """
    sweep_results: {min_instances_value -> list of per-trace record dicts}
    Produces one figure per metric: x=min_instances, one line per trace.
    Also produces a combined 2×3 figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    min_vals = sorted(sweep_results.keys())

    # Collect all trace labels
    all_labels: list[str] = []
    for records in sweep_results.values():
        for r in records:
            if r["label"] not in all_labels:
                all_labels.append(r["label"])
    all_labels = sorted(all_labels)

    # Build per-label, per-metric time series
    # data[label][metric] = list of values aligned with min_vals
    data: dict[str, dict[str, list[float]]] = {
        lbl: {m: [] for m, *_ in METRICS} for lbl in all_labels
    }
    for mv in min_vals:
        records_by_label = {r["label"]: r for r in sweep_results[mv]}
        for lbl in all_labels:
            r = records_by_label.get(lbl)
            for metric, *_ in METRICS:
                data[lbl][metric].append(r[metric] if r else float("nan"))

    os.makedirs(out_dir, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(all_labels))]

    n_metrics = len(METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, (metric, ylabel, _) in zip(axes_flat, METRICS):
        for i, lbl in enumerate(all_labels):
            ax.plot(
                min_vals,
                data[lbl][metric],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[i],
                label=lbl,
            )
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("min_instances", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(min_vals)
        ax.grid(True, alpha=0.3)

    # hide unused subplot(s)
    for ax in list(axes_flat)[n_metrics:]:
        ax.set_visible(False)

    # shared legend below
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, lbls,
        loc="lower center",
        ncol=min(len(all_labels), 4),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Metrics vs services[0].min_instances — {len(all_labels)} traces",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "benchmark_min_instances_sweep.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Per-metric individual figures
    for metric, ylabel, color in METRICS:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        for i, lbl in enumerate(all_labels):
            ax2.plot(
                min_vals,
                data[lbl][metric],
                marker="o",
                linewidth=1.8,
                markersize=5,
                color=colors[i],
                label=lbl,
            )
        ax2.set_title(f"{ylabel} vs min_instances", fontsize=11)
        ax2.set_xlabel("min_instances", fontsize=9)
        ax2.set_ylabel(ylabel, fontsize=9)
        ax2.set_xticks(min_vals)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7, loc="best")
        plt.tight_layout()
        out2 = os.path.join(out_dir, f"sweep_{metric}.png")
        fig2.savefig(out2, dpi=130, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out2}")

    # CSV
    import csv
    csv_path = os.path.join(out_dir, "benchmark_min_instances_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        fields = ["label", "min_instances"] + [m for m, *_ in METRICS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mv in min_vals:
            for r in sweep_results[mv]:
                row = {"label": r["label"], "min_instances": mv}
                for metric, *_ in METRICS:
                    row[metric] = round(r[metric], 4)
                w.writerow(row)
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_min_instances(s: str) -> list[int]:
    """Accept '5,10,20,50' or '5:55:10' (start:stop:step, inclusive)."""
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            step = 1
        else:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
        return list(range(start, stop + 1, step))
    return [int(v) for v in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parallel",      type=int, default=4,       help="Parallel workers (default 4)")
    parser.add_argument("--out",           default=DEFAULT_OUT,        help="Output directory")
    parser.add_argument("--log-dir",       default="logs/benchmark_datanew", help="Sim log root dir")
    parser.add_argument(
        "--min-instances",
        metavar="RANGE",
        default=None,
        help=(
            "Sweep services[0].min_instances over these values. "
            "Comma-separated list (e.g. '5,10,20,50') or start:stop:step range (e.g. '5:55:10'). "
            "When omitted, uses the single value in BASELINE_OVERRIDES and produces a bar chart."
        ),
    )
    args = parser.parse_args()

    traces = sorted(str(p) for p in Path(DATANEW_DIR).glob(PATTERN))
    print(f"Found {len(traces)} traces in {DATANEW_DIR}")

    import pandas as pd
    record_counts = {}
    for tp in traces:
        df = pd.read_csv(tp)
        if "count" in df.columns:
            record_counts[tp] = int(df["count"].sum())
        elif "timestamp" in df.columns:
            record_counts[tp] = len(df)
        else:
            record_counts[tp] = len(df)
    max_count = max(record_counts.values())
    trace_scales = {tp: max(1, round(max_count / record_counts[tp])) for tp in traces}

    print(f"Max trace size: {max_count:,} requests")
    print(f"Scale range: {min(trace_scales.values())}–{max(trace_scales.values())}")

    if args.min_instances is None:
        # ---- single run: bar chart ----
        min_vals = [int(BASELINE_OVERRIDES["services[0].min_instances"])]
    else:
        min_vals = _parse_min_instances(args.min_instances)

    total_jobs = len(traces) * len(min_vals)
    print(f"min_instances sweep: {min_vals}")
    print(f"Total jobs: {len(traces)} traces × {len(min_vals)} values = {total_jobs}")
    print(f"Running default_baseline, parallel={args.parallel}\n")

    # Flatten all (trace, mv) pairs into one pool
    futures: dict = {}
    raw_results: dict[tuple[str, int], tuple[str, dict]] = {}  # (tp, mv) -> (run_dir, summary)

    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        for mv in min_vals:
            overrides = {"services[0].min_instances": mv}
            for tp in traces:
                run_dir = os.path.join(args.log_dir, f"min{mv}", Path(tp).stem)
                f = pool.submit(_worker, tp, run_dir, trace_scales[tp], overrides)
                futures[f] = (tp, run_dir, mv)

        done = 0
        for f in as_completed(futures):
            tp, run_dir, mv = futures[f]
            _, summary, elapsed = f.result()
            done += 1
            if "error" in summary:
                status = f"FAILED — {summary['error']}"
            else:
                req   = summary.get("requests", {})
                rates = summary.get("rates", {})
                status = (f"OK {elapsed:.1f}s — "
                          f"total={req.get('total',0):,}  "
                          f"drop={rates.get('drop_rate_pct',0):.2f}%  "
                          f"cold={rates.get('cold_start_rate_pct',0):.2f}%")
            print(f"[{done:>3}/{total_jobs}][min={mv:>3}] {Path(tp).stem[:38]:<38} {status}")
            raw_results[(tp, mv)] = (run_dir, summary)

    # Group into sweep_results[mv] -> list of records
    sweep_results: dict[int, list[dict]] = {mv: [] for mv in min_vals}
    for mv in min_vals:
        for tp in traces:
            run_dir, summary = raw_results.get((tp, mv), ("", {}))
            if "error" not in summary and summary:
                sweep_results[mv].append(_extract(tp, run_dir, summary))
        n = len(sweep_results[mv])
        print(f"min_instances={mv}: {n}/{len(traces)} succeeded")

    if len(min_vals) == 1:
        records = sweep_results[min_vals[0]]
        if records:
            _plot_bar(records, args.out)
    else:
        _plot_sweep(sweep_results, args.out)


if __name__ == "__main__":
    main()
