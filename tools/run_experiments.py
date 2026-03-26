"""Run simulations for all config files in an experimental directory.

Usage:
    python tools/run_experiments.py [--exp-dir experimental/min_baseline] [--progress]
"""

import argparse
import glob
import json
import os
import sys
import time

# Ensure serverless_sim is importable when running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_one(config_path: str, progress: bool = False) -> dict:
    """Run a single simulation and return summary dict."""
    from serverless_sim.core.config.loader import load_config
    from serverless_sim.core.logging.logger_factory import create_logger
    from serverless_sim.core.simulation.sim_builder import SimulationBuilder
    from serverless_sim.core.simulation.sim_engine import SimulationEngine

    config = load_config(config_path)

    # Create run dir under logs/
    base_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", f"run_{timestamp}_{base_name}")
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

    # Read summary
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


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
        help="Show progress bar for each simulation",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run configs matching this substring (e.g. 'Java')",
    )
    args = parser.parse_args()

    configs = sorted(glob.glob(os.path.join(args.exp_dir, "*.json")))
    if args.filter:
        configs = [c for c in configs if args.filter in os.path.basename(c)]

    if not configs:
        print(f"No JSON configs found in {args.exp_dir}")
        return

    print(f"Found {len(configs)} experiments in {args.exp_dir}:\n")
    for c in configs:
        print(f"  {os.path.basename(c)}")
    print()

    results = {}
    for i, config_path in enumerate(configs, 1):
        name = os.path.splitext(os.path.basename(config_path))[0]
        print(f"[{i}/{len(configs)}] Running {name}...")
        t0 = time.monotonic()

        try:
            summary = run_one(config_path, progress=args.progress)
            elapsed = time.monotonic() - t0

            req = summary.get("requests", {})
            ratio = summary.get("effective_resource_ratio", {})
            print(f"  Done in {elapsed:.1f}s — "
                  f"requests={req.get('total', 0)}, "
                  f"completed={req.get('completed', 0)}, "
                  f"dropped={req.get('dropped', 0)}, "
                  f"cold_starts={req.get('cold_starts', 0)}, "
                  f"cpu_eff={ratio.get('cpu_effective_ratio', 0):.4f}, "
                  f"mem_eff={ratio.get('memory_effective_ratio', 0):.4f}")

            results[name] = summary
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  FAILED in {elapsed:.1f}s: {e}")
            results[name] = {"error": str(e)}

    # Save combined results
    output_path = os.path.join(args.exp_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {output_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<25} {'Total':>8} {'Done':>8} {'Drop':>8} {'Cold':>8} {'CPU eff':>10} {'Mem eff':>10}")
    print(f"{'-'*80}")
    for name, summary in results.items():
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
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
