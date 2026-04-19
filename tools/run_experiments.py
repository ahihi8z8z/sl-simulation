"""Run simulations for experiments defined in experiments.json.

Usage:
    python tools/run_experiments.py experimental/experiments.json
    python tools/run_experiments.py experimental/experiments.json --filter default
    python tools/run_experiments.py experimental/experiments.json --parallel 4
    python tools/run_experiments.py experimental/experiments.json --dump-config default_baseline
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _run_single(config: dict, name: str, run_dir: str, progress: bool = False) -> dict:
    """Run one simulation from a merged config dict. Returns summary."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from serverless_sim.core.config.loader import load_config_from_dict
    from serverless_sim.core.logging.logger_factory import create_logger
    from serverless_sim.core.simulation.sim_builder import SimulationBuilder
    from serverless_sim.core.simulation.sim_engine import SimulationEngine

    os.makedirs(run_dir, exist_ok=True)
    validated = load_config_from_dict(config)

    logger = create_logger(
        module_name=f"sim.{name}",
        run_dir=run_dir,
        mode="file",
        level="WARNING",
    )

    builder = SimulationBuilder()
    ctx = builder.build(validated, run_dir, logger)
    engine = SimulationEngine(ctx)
    engine.setup()
    engine.run(progress=progress)
    engine.shutdown()

    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


def _run_worker(config: dict, name: str, run_dir: str) -> tuple[str, dict, float]:
    """Worker for parallel execution."""
    t0 = time.monotonic()
    try:
        summary = _run_single(config, name, run_dir)
        return name, summary, time.monotonic() - t0
    except Exception as e:
        return name, {"error": str(e)}, time.monotonic() - t0


def _format_result(name: str, summary: dict, elapsed: float) -> str:
    if "error" in summary:
        return f"  {name}: FAILED in {elapsed:.1f}s — {summary['error']}"
    req = summary.get("requests", {})
    ratio = summary.get("effective_resource_ratio", {})
    return (
        f"  {name}: Done in {elapsed:.1f}s — "
        f"requests={req.get('total', 0)}, "
        f"completed={req.get('completed', 0)}, "
        f"dropped={req.get('dropped', 0)}, "
        f"cold_starts={req.get('cold_starts', 0)}, "
        f"cpu_eff={ratio.get('cpu_effective_ratio', 0):.4f}, "
        f"mem_eff={ratio.get('memory_effective_ratio', 0):.4f}"
    )


def _save_results_csv(results: dict, path: str) -> None:
    import csv
    header = [
        "experiment", "total", "completed", "dropped", "truncated", "cold_starts",
        "throughput", "drop_rate_pct", "cold_start_rate_pct", "latency_mean",
        "cpu_effective_ratio", "memory_effective_ratio",
        "total_cpu_seconds", "running_cpu_seconds",
        "total_memory_seconds", "running_memory_seconds",
        "wall_clock_seconds", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for name, summary in results.items():
            if "error" in summary:
                writer.writerow({"experiment": name, "error": summary["error"]})
                continue
            req = summary.get("requests", {})
            rates = summary.get("rates", {})
            lat = summary.get("latency", {})
            ratio = summary.get("effective_resource_ratio", {})
            sim = summary.get("simulation", {})
            writer.writerow({
                "experiment": name,
                "total": req.get("total", 0),
                "completed": req.get("completed", 0),
                "dropped": req.get("dropped", 0),
                "truncated": req.get("truncated", 0),
                "cold_starts": req.get("cold_starts", 0),
                "throughput": rates.get("throughput_req_per_s", ""),
                "drop_rate_pct": rates.get("drop_rate_pct", ""),
                "cold_start_rate_pct": rates.get("cold_start_rate_pct", ""),
                "latency_mean": lat.get("mean", ""),
                "cpu_effective_ratio": ratio.get("cpu_effective_ratio", ""),
                "memory_effective_ratio": ratio.get("memory_effective_ratio", ""),
                "total_cpu_seconds": ratio.get("total_cpu_seconds", ""),
                "running_cpu_seconds": ratio.get("running_cpu_seconds", ""),
                "total_memory_seconds": ratio.get("total_memory_seconds", ""),
                "running_memory_seconds": ratio.get("running_memory_seconds", ""),
                "wall_clock_seconds": sim.get("wall_clock_seconds", ""),
                "error": "",
            })


def _print_table(results: dict) -> None:
    print(f"\n{'='*120}")
    print(f"{'Experiment':<25} {'Total':>8} {'Done':>8} {'Drop':>8} {'Cold':>8} {'CPU eff':>10} {'Mem eff':>10} {'CPU/req':>10} {'Mem/req':>10}")
    print(f"{'-'*120}")
    for name, summary in sorted(results.items()):
        if "error" in summary:
            print(f"{name:<25} ERROR: {summary['error']}")
            continue
        req = summary.get("requests", {})
        ratio = summary.get("effective_resource_ratio", {})
        print(f"{name:<25} "
              f"{req.get('total', 0):>8} "
              f"{req.get('completed', 0):>8} "
              f"{req.get('dropped', 0):>8} "
              f"{req.get('cold_starts', 0):>8} "
              f"{ratio.get('cpu_effective_ratio', 0):>10.4f} "
              f"{ratio.get('memory_effective_ratio', 0):>10.4f} "
              f"{ratio.get('cpu_per_request', 0):>10.4f} "
              f"{ratio.get('memory_per_request', 0):>10.4f}")
    print(f"{'='*120}")


def main():
    parser = argparse.ArgumentParser(description="Run experiments from experiments.json")
    parser.add_argument("experiments_file", help="Path to experiments.json")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated exact experiment names to run")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--progress", action="store_true", help="Show progress bar (sequential only)")
    parser.add_argument("--dump-config", default=None, metavar="NAME",
                        help="Dump merged config for an experiment and exit")
    args = parser.parse_args()

    from tools.config_merge import load_experiments, load_merged_config

    base_path, data = load_experiments(args.experiments_file)
    experiments = data["experiments"]

    # Dump mode
    if args.dump_config:
        for exp in experiments:
            if exp["name"] == args.dump_config:
                config = load_merged_config(base_path, exp.get("overrides"))
                print(json.dumps(config, indent=2))
                return
        print(f"Experiment '{args.dump_config}' not found")
        return

    # Filter to simulation-only experiments (no rl_template)
    sim_experiments = [e for e in experiments if "rl_template" not in e]
    if args.filter:
        wanted = {f.strip() for f in args.filter.split(",")}
        sim_experiments = [e for e in sim_experiments if e["name"] in wanted]

    if not sim_experiments:
        print("No matching experiments found")
        return

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join("logs", f"experiments_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"Running {len(sim_experiments)} experiments")
    print(f"Output: {sweep_dir}\n")
    for e in sim_experiments:
        print(f"  {e['name']}")
    print()

    # Build merged configs
    jobs = []
    for exp in sim_experiments:
        config = load_merged_config(base_path, exp.get("overrides"))
        run_dir = os.path.join(sweep_dir, exp["name"])
        jobs.append((config, exp["name"], run_dir))

    results = {}
    t_total = time.monotonic()
    n_workers = min(args.parallel, len(jobs))

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_worker, cfg, name, rd): name
                for cfg, name, rd in jobs
            }
            for future in as_completed(futures):
                name, summary, elapsed = future.result()
                results[name] = summary
                print(_format_result(name, summary, elapsed))
    else:
        for i, (config, name, run_dir) in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] Running {name}...")
            t0 = time.monotonic()
            try:
                summary = _run_single(config, name, run_dir, progress=args.progress)
                elapsed = time.monotonic() - t0
                results[name] = summary
                print(_format_result(name, summary, elapsed))
            except Exception as e:
                elapsed = time.monotonic() - t0
                results[name] = {"error": str(e)}
                print(f"  FAILED in {elapsed:.1f}s: {e}")

    total_elapsed = time.monotonic() - t_total

    output_path = os.path.join(sweep_dir, "results.csv")
    _save_results_csv(results, output_path)
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {total_elapsed:.1f}s")
    _print_table(results)


if __name__ == "__main__":
    main()
