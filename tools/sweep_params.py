"""Parameter sweep: apply sweep config on top of base simulation config.

Sweep config format:
{
    "base_config": "experimental/toy/config.json",
    "sweep": {
        "workload.gamma_alpha": [0.3, 1.0, 5.0],
        "workload.gamma_beta": [1.0, 2.0],
        "services[0].autoscaling_defaults.pool_targets.prewarm": [0, 5, 10],
        "services[0].autoscaling_defaults.idle_timeout": [10, 60]
    }
}

Usage:
    python tools/sweep_params.py experimental/toy/sweep.json
    python tools/sweep_params.py experimental/toy/sweep.json --parallel 8
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _set_nested(cfg: dict, path: str, value) -> None:
    """Set a value at a dotted path, supporting [N] indexing."""
    parts = path.replace("[", ".[").split(".")
    obj = cfg
    for part in parts[:-1]:
        if part.startswith("[") and part.endswith("]"):
            obj = obj[int(part[1:-1])]
        else:
            obj = obj.setdefault(part, {})
    last = parts[-1]
    if last.startswith("[") and last.endswith("]"):
        obj[int(last[1:-1])] = value
    else:
        obj[last] = value


def _short_key(path: str) -> str:
    """Extract short key from dotted path for naming."""
    return path.split(".")[-1].replace("[0]", "")


def _make_name(params: list[tuple[str, any]]) -> str:
    """Create short name from param values."""
    parts = []
    for path, value in params:
        key = _short_key(path)
        if isinstance(value, float) and value == int(value):
            parts.append(f"{key}{int(value)}")
        else:
            parts.append(f"{key}{value}")
    return "_".join(parts)


def _run_one(args: tuple) -> dict:
    """Worker: run one simulation."""
    config_dict, name, run_dir = args

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from serverless_sim.core.config.loader import load_config_from_dict
    from serverless_sim.core.simulation.sim_builder import SimulationBuilder
    from serverless_sim.core.simulation.sim_engine import SimulationEngine
    import logging

    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(f"sweep.{name}")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)

    t0 = time.monotonic()
    try:
        config = load_config_from_dict(config_dict)
        builder = SimulationBuilder()
        ctx = builder.build(config=config, run_dir=run_dir, logger=logger,
                            export_mode_override=0)
        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run(progress=False)
        engine.shutdown()

        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        elapsed = time.monotonic() - t0
        return {"name": name, "summary": summary, "elapsed": elapsed}
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {"name": name, "summary": {"error": str(e)}, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep")
    parser.add_argument("sweep_config", help="Sweep config JSON")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        sweep_cfg = json.load(f)

    base_path = sweep_cfg["base_config"]
    sweep_params = sweep_cfg["sweep"]

    # Resolve base_config path relative to sweep config
    if not os.path.isabs(base_path):
        sweep_dir = os.path.dirname(os.path.abspath(args.sweep_config))
        candidate = os.path.join(sweep_dir, base_path)
        if os.path.exists(candidate):
            base_path = candidate

    with open(base_path) as f:
        base_cfg = json.load(f)

    paths = list(sweep_params.keys())
    all_values = [sweep_params[p] for p in paths]
    combos = list(itertools.product(*all_values))
    total = len(combos)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("logs", f"sweep_{timestamp}_params")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base config: {base_path}")
    print(f"Sweep parameters:")
    for path in paths:
        print(f"  {path}: {sweep_params[path]}")
    print(f"\n{total} combinations, output: {output_dir}, parallel: {args.parallel}\n")

    # Prepare jobs
    jobs = []
    for combo in combos:
        param_values = list(zip(paths, combo))
        name = _make_name(param_values)
        cfg = copy.deepcopy(base_cfg)
        for path, value in param_values:
            _set_nested(cfg, path, value)
        run_dir = os.path.join(output_dir, name)
        jobs.append((cfg, name, run_dir))

    # Run
    results = []
    t_total = time.monotonic()

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_run_one, job): job[1] for job in jobs}
            done = 0
            for future in as_completed(futures):
                done += 1
                result = future.result()
                results.append(result)
                s = result["summary"]
                if "error" in s:
                    print(f"  [{done}/{total}] {result['name']}: ERROR {s['error']}")
                else:
                    r = s.get("requests", {})
                    cu = s.get("cluster_utilization", {})
                    comp = max(r.get("completed", 1), 1)
                    print(f"  [{done}/{total}] {result['name']}: "
                          f"cold={r.get('cold_starts', 0)/comp*100:.1f}% "
                          f"drop={r.get('dropped', 0)} "
                          f"mem={cu.get('avg_memory_utilization', 0)*100:.1f}% "
                          f"({result['elapsed']:.1f}s)")
    else:
        for i, job in enumerate(jobs, 1):
            result = _run_one(job)
            results.append(result)
            s = result["summary"]
            if "error" in s:
                print(f"  [{i}/{total}] {result['name']}: ERROR {s['error']}")
            else:
                r = s.get("requests", {})
                cu = s.get("cluster_utilization", {})
                comp = max(r.get("completed", 1), 1)
                print(f"  [{i}/{total}] {result['name']}: "
                      f"cold={r.get('cold_starts', 0)/comp*100:.1f}% "
                      f"drop={r.get('dropped', 0)} "
                      f"mem={cu.get('avg_memory_utilization', 0)*100:.1f}% "
                      f"({result['elapsed']:.1f}s)")

    total_elapsed = time.monotonic() - t_total

    # Save CSV
    csv_path = os.path.join(output_dir, "results.csv")
    short_keys = [_short_key(p) for p in paths]
    metric_cols = ["total", "completed", "dropped", "cold_starts",
                   "drop_pct", "cold_start_pct", "latency_mean",
                   "mem_per_req_pct", "wall_seconds"]
    header = ["name"] + short_keys + metric_cols

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for result in sorted(results, key=lambda r: r["name"]):
            s = result["summary"]
            row = {"name": result["name"], "wall_seconds": round(result["elapsed"], 2)}

            # Find matching job to get param values
            for job in jobs:
                if job[1] == result["name"]:
                    cfg = job[0]
                    for path, key in zip(paths, short_keys):
                        parts = path.replace("[", ".[").split(".")
                        obj = cfg
                        for part in parts:
                            if part.startswith("[") and part.endswith("]"):
                                obj = obj[int(part[1:-1])]
                            else:
                                obj = obj[part]
                        row[key] = obj
                    break

            if "error" in s:
                row["cold_start_pct"] = "ERROR"
                writer.writerow(row)
                continue

            r = s.get("requests", {})
            cu = s.get("cluster_utilization", {})
            eff = s.get("effective_resource_ratio", {})
            lat = s.get("latency", {})
            comp = max(r.get("completed", 1), 1)
            cluster_mem = cu.get("memory_total", 1)

            row.update({
                "total": r.get("total", 0),
                "completed": r.get("completed", 0),
                "dropped": r.get("dropped", 0),
                "cold_starts": r.get("cold_starts", 0),
                "drop_pct": round(r.get("dropped", 0) / max(r.get("total", 1), 1) * 100, 2),
                "cold_start_pct": round(r.get("cold_starts", 0) / comp * 100, 2),
                "latency_mean": round(lat.get("mean", 0), 6),
                "mem_per_req_pct": round(eff.get("memory_per_request", 0) / cluster_mem * 100, 4) if cluster_mem > 0 else 0,
            })
            writer.writerow(row)

    # Save sweep config copy
    with open(os.path.join(output_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_cfg, f, indent=2)

    print(f"\nResults saved to {csv_path}")
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
