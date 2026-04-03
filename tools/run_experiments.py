"""Run simulations for all config files in an experimental directory.

Usage:
    python tools/run_experiments.py [--exp-dir experimental/min_baseline] [--parallel 4]
"""

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure serverless_sim is importable when running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _run_worker(config_path: str, sweep_dir: str) -> tuple[str, dict, float]:
    """Worker function for parallel execution. Returns (name, summary, elapsed)."""
    # Re-insert path for subprocess
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from serverless_sim.core.config.loader import load_config
    from serverless_sim.core.logging.logger_factory import create_logger
    from serverless_sim.core.simulation.sim_builder import SimulationBuilder
    from serverless_sim.core.simulation.sim_engine import SimulationEngine

    base_name = os.path.splitext(os.path.basename(config_path))[0]
    t0 = time.monotonic()

    try:
        config = load_config(config_path)
        run_dir = os.path.join(sweep_dir, base_name)
        os.makedirs(run_dir, exist_ok=True)

        logger = create_logger(
            module_name=f"sim.{base_name}",
            run_dir=run_dir,
            mode="file",
            level="WARNING",
        )

        builder = SimulationBuilder()
        ctx = builder.build(config, run_dir, logger)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run(progress=False)
        engine.shutdown()

        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {}

        elapsed = time.monotonic() - t0
        return base_name, summary, elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        return base_name, {"error": str(e)}, elapsed


def run_one(config_path: str, sweep_dir: str, progress: bool = False) -> dict:
    """Run a single simulation (sequential mode)."""
    from serverless_sim.core.config.loader import load_config
    from serverless_sim.core.logging.logger_factory import create_logger
    from serverless_sim.core.simulation.sim_builder import SimulationBuilder
    from serverless_sim.core.simulation.sim_engine import SimulationEngine

    config = load_config(config_path)

    base_name = os.path.splitext(os.path.basename(config_path))[0]
    run_dir = os.path.join(sweep_dir, base_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = create_logger(
        module_name=f"sim.{base_name}",
        run_dir=run_dir,
        mode="file",
        level="WARNING",
    )

    builder = SimulationBuilder()
    ctx = builder.build(config, run_dir, logger)

    engine = SimulationEngine(ctx)
    engine.setup()
    engine.run(progress=progress)
    engine.shutdown()

    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


def _format_result(name: str, summary: dict, elapsed: float) -> str:
    """Format one result line."""
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
    """Save results as CSV."""
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
    """Print summary table."""
    print(f"\n{'='*90}")
    print(f"{'Experiment':<25} {'Total':>8} {'Done':>8} {'Drop':>8} {'Cold':>8} {'CPU eff':>10} {'Mem eff':>10}")
    print(f"{'-'*90}")
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
              f"{ratio.get('memory_effective_ratio', 0):>10.4f}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments in a directory")
    parser.add_argument(
        "--exp-dir",
        default="experimental/min_baseline",
        help="Directory containing experiment config JSONs",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar (only in sequential mode)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run configs matching this substring (e.g. 'Java')",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    args = parser.parse_args()

    configs = sorted(glob.glob(os.path.join(args.exp_dir, "*.json")))
    if args.filter:
        configs = [c for c in configs if args.filter in os.path.basename(c)]
    configs = [c for c in configs if os.path.basename(c) not in ("results.json", "results.csv")]

    if not configs:
        print(f"No JSON configs found in {args.exp_dir}")
        return

    # Create sweep directory
    exp_name = os.path.basename(os.path.normpath(args.exp_dir))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join("logs", f"sweep_{timestamp}_{exp_name}")
    os.makedirs(sweep_dir, exist_ok=True)

    n_workers = min(args.parallel, len(configs))
    mode = f"parallel ({n_workers} workers)" if n_workers > 1 else "sequential"

    print(f"Found {len(configs)} experiments in {args.exp_dir}")
    print(f"Sweep directory: {sweep_dir}")
    print(f"Mode: {mode}\n")
    for c in configs:
        print(f"  {os.path.basename(c)}")
    print()

    results = {}
    t_total = time.monotonic()

    if n_workers > 1:
        # Parallel execution with progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_worker, cfg, sweep_dir): cfg
                for cfg in configs
            }
            if use_tqdm:
                pbar = tqdm(total=len(configs), desc="Sweep", unit="exp")
            for future in as_completed(futures):
                name, summary, elapsed = future.result()
                results[name] = summary
                if use_tqdm:
                    pbar.update(1)
                    pbar.set_postfix_str(f"{name} ({elapsed:.0f}s)", refresh=False)
                else:
                    print(_format_result(name, summary, elapsed))
            if use_tqdm:
                pbar.close()
    else:
        # Sequential execution
        for i, config_path in enumerate(configs, 1):
            name = os.path.splitext(os.path.basename(config_path))[0]
            print(f"[{i}/{len(configs)}] Running {name}...")
            t0 = time.monotonic()

            try:
                summary = run_one(config_path, sweep_dir, progress=args.progress)
                elapsed = time.monotonic() - t0
                results[name] = summary
                print(_format_result(name, summary, elapsed))
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"  FAILED in {elapsed:.1f}s: {e}")
                results[name] = {"error": str(e)}

    total_elapsed = time.monotonic() - t_total

    # Save combined results as CSV
    output_path = os.path.join(sweep_dir, "results.csv")
    _save_results_csv(results, output_path)
    print(f"\nAll results saved to {output_path}")
    print(f"Total time: {total_elapsed:.1f}s")

    _print_table(results)


if __name__ == "__main__":
    main()
