"""Parameter sweep using config merge.

Sweep config format:
{
    "base": "experimental/base_config.json",
    "overrides": {"services[0].min_instances": 2},
    "sweep": {
        "services[0].autoscaling_defaults.idle_timeout": [10, 30, 60, 120, 300]
    }
}

Usage:
    python tools/sweep_params.py configs/sweep/sweep_idle.json
    python tools/sweep_params.py configs/sweep/sweep_idle.json --parallel 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _short_key(path: str) -> str:
    return path.split(".")[-1].replace("[0]", "")


def _make_name(params: list[tuple[str, object]]) -> str:
    parts = []
    for path, value in params:
        key = _short_key(path)
        if isinstance(value, float) and value == int(value):
            parts.append(f"{key}{int(value)}")
        else:
            parts.append(f"{key}{value}")
    return "_".join(parts)


def _run_one(args: tuple) -> dict:
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

    from tools.config_merge import expand_sweep

    with open(args.sweep_config) as f:
        sweep_cfg = json.load(f)

    # Resolve base path relative to sweep config file
    base_path = sweep_cfg["base"]
    if not os.path.isabs(base_path):
        sweep_dir = os.path.dirname(os.path.abspath(args.sweep_config))
        candidate = os.path.join(sweep_dir, base_path)
        if os.path.exists(candidate):
            base_path = candidate

    overrides = sweep_cfg.get("overrides", {})
    sweep_params = sweep_cfg.get("sweep", {})

    configs = expand_sweep(base_path, overrides=overrides, sweep=sweep_params)
    total = len(configs)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("logs", f"sweep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base config: {base_path}")
    if overrides:
        print(f"Overrides: {overrides}")
    print(f"Sweep parameters:")
    for path, values in sweep_params.items():
        print(f"  {path}: {values}")
    print(f"\n{total} combinations, output: {output_dir}, parallel: {args.parallel}\n")

    # Prepare jobs
    jobs = []
    for config, sweep_point in configs:
        if sweep_point:
            name = _make_name(list(sweep_point.items()))
        else:
            name = "baseline"
        run_dir = os.path.join(output_dir, name)
        jobs.append((config, name, run_dir))

    # Run
    results = []
    t_total = time.monotonic()

    if args.parallel > 1 and total > 1:
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

    # Save results CSV
    import csv
    csv_path = os.path.join(output_dir, "results.csv")
    header = ["name"] + list(sweep_params.keys()) + [
        "total", "completed", "dropped", "cold_starts",
        "drop_rate_pct", "cold_start_rate_pct", "latency_mean",
        "cpu_eff", "mem_eff", "elapsed",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for result in sorted(results, key=lambda r: r["name"]):
            s = result["summary"]
            if "error" in s:
                writer.writerow({"name": result["name"], "elapsed": result["elapsed"]})
                continue
            req = s.get("requests", {})
            rates = s.get("rates", {})
            lat = s.get("latency", {})
            ratio = s.get("effective_resource_ratio", {})
            row = {
                "name": result["name"],
                "total": req.get("total", 0),
                "completed": req.get("completed", 0),
                "dropped": req.get("dropped", 0),
                "cold_starts": req.get("cold_starts", 0),
                "drop_rate_pct": rates.get("drop_rate_pct", ""),
                "cold_start_rate_pct": rates.get("cold_start_rate_pct", ""),
                "latency_mean": lat.get("mean", ""),
                "cpu_eff": ratio.get("cpu_effective_ratio", ""),
                "mem_eff": ratio.get("memory_effective_ratio", ""),
                "elapsed": f"{result['elapsed']:.1f}",
            }
            writer.writerow(row)

    print(f"\nResults saved to {csv_path}")
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
