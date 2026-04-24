"""Run all experiments (baseline simulations + RL inference) across episodes.

Each experiment × episode produces output in its own directory:
  - Baselines (no rl_template): logs/<name>/sim/episode_<N>/
  - RL agents (has rl_template): logs/<name>/infer/episode_<N>/

Episodes use seed = base_seed + ep, so each episode is deterministic
but different. compare_logs.py reads episode_*/ dirs for aggregation.

Usage:
    python tools/run_all.py experimental/experiments.json
    python tools/run_all.py experimental/experiments.json --filter our_sac,default_baseline
    python tools/run_all.py experimental/experiments.json --only sim --episodes 3
    python tools/run_all.py experimental/experiments.json --only infer --parallel 4
    python tools/run_all.py experimental/experiments.json --dump-config default_baseline
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Sim runner (baseline experiments)
# ---------------------------------------------------------------------------

def _run_sim(config: dict, name: str, run_dir: str, progress: bool = False) -> dict:
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


def _sim_worker(config: dict, name: str, run_dir: str) -> tuple[str, str, dict, float]:
    """Worker for parallel sim execution."""
    t0 = time.monotonic()
    try:
        summary = _run_sim(config, name, run_dir)
        return name, run_dir, summary, time.monotonic() - t0
    except Exception as e:
        return name, run_dir, {"error": str(e)}, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Infer runner (RL experiments)
# ---------------------------------------------------------------------------

def _run_infer(exp: dict, data: dict, base_path: str,
               seed: int, ep: int, ep_dir: str, export_mode: int) -> dict:
    """Build configs and run one RL inference episode. Returns summary."""
    from tools.config_merge import load_merged_config, build_gym_config, build_infer_config
    from rl_agent.infer import run_inference

    os.makedirs(ep_dir, exist_ok=True)

    sim_config = load_merged_config(base_path, exp.get("overrides"))
    sim_config["simulation"]["seed"] = seed + ep
    sim_config["simulation"]["export_mode"] = export_mode

    gym_config = build_gym_config(exp, data)
    if gym_config is not None:
        gym_config["export_mode"] = export_mode
        gym_config["run_dir"] = ep_dir

    rl_infer = build_infer_config(exp, data, seed=seed + ep)
    rl_infer["n_episodes"] = 1  # 1 episode per call, folder already set

    sim_path = os.path.join(ep_dir, "sim_config.json")
    gym_path = os.path.join(ep_dir, "gym_config.json")
    rl_path = os.path.join(ep_dir, "rl_infer.json")
    with open(sim_path, "w") as f:
        json.dump(sim_config, f, indent=2)
    with open(gym_path, "w") as f:
        json.dump(gym_config, f, indent=2)
    with open(rl_path, "w") as f:
        json.dump(rl_infer, f, indent=2)

    run_inference(sim_path, gym_path if gym_config else None,
                  rl_path, run_dir=ep_dir)

    summary_path = os.path.join(ep_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_result(label: str, summary: dict, elapsed: float) -> str:
    if "error" in summary:
        return f"  {label}: FAILED in {elapsed:.1f}s — {summary['error']}"
    req = summary.get("requests", {})
    rates = summary.get("rates", {})
    return (
        f"  {label}: OK {elapsed:.1f}s — "
        f"total={req.get('total', 0)}, "
        f"drop={rates.get('drop_rate_pct', 0):.2f}%, "
        f"cold={rates.get('cold_start_rate_pct', 0):.2f}%"
    )


def _print_table(results: dict) -> None:
    print(f"\n{'='*100}")
    print(f"{'Experiment':<35} {'Total':>8} {'Done':>8} {'Drop':>8} {'Cold':>8} {'CPU eff':>10} {'Mem eff':>10}")
    print(f"{'-'*100}")
    for label, summary in sorted(results.items()):
        if "error" in summary:
            print(f"{label:<35} ERROR: {summary['error']}")
            continue
        req = summary.get("requests", {})
        ratio = summary.get("effective_resource_ratio", {})
        print(f"{label:<35} "
              f"{req.get('total', 0):>8} "
              f"{req.get('completed', 0):>8} "
              f"{req.get('dropped', 0):>8} "
              f"{req.get('cold_starts', 0):>8} "
              f"{ratio.get('cpu_effective_ratio', 0):>10.4f} "
              f"{ratio.get('memory_effective_ratio', 0):>10.4f}")
    print(f"{'='*100}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run all experiments (sim + infer)")
    parser.add_argument("experiments_file", help="Path to experiments.json")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated experiment names")
    parser.add_argument("--only", choices=["sim", "infer", "all"], default="all",
                        help="Run only sim (baselines) or infer (RL) experiments")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override n_episodes from run_defaults")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Parallel workers for sim jobs (default: 1)")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress bar (sequential sim only)")
    parser.add_argument("--dump-config", default=None, metavar="NAME",
                        help="Dump merged config for an experiment and exit")
    args = parser.parse_args()

    from tools.config_merge import (
        load_experiments, load_merged_config, build_infer_config,
    )

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

    # Filter
    if args.filter:
        wanted = {f.strip() for f in args.filter.split(",")}
        experiments = [e for e in experiments if e["name"] in wanted]

    if not experiments:
        print("No matching experiments found")
        return

    # Run defaults
    run_defaults = data.get("run_defaults", {})
    base_seed = run_defaults.get("seed", 42)
    n_episodes = args.episodes or run_defaults.get("n_episodes", 1)
    export_mode = run_defaults.get("export_mode", 2)

    # Split into sim vs infer
    sim_exps = [e for e in experiments if not e.get("rl_template")]
    infer_exps = [e for e in experiments if e.get("rl_template")]

    if args.only == "sim":
        infer_exps = []
    elif args.only == "infer":
        sim_exps = []

    # Check for missing RL models (fail-fast)
    missing = []
    runnable_infer = []
    for exp in infer_exps:
        probe = build_infer_config(exp, data, seed=base_seed)
        model_zip = probe["model_path"] + ".zip"
        if not os.path.exists(model_zip):
            missing.append(f"{exp['name']} → {model_zip}")
        else:
            runnable_infer.append(exp)
    if missing:
        print("Skipping experiments with no trained model:")
        for m in missing:
            print(f"  {m}")
        print()

    total_jobs = len(sim_exps) * n_episodes + len(runnable_infer) * n_episodes
    if total_jobs == 0:
        print("No runnable jobs")
        return

    print(f"Running {total_jobs} jobs ({n_episodes} episodes, seed={base_seed})")
    for e in sim_exps:
        print(f"  [sim]   {e['name']}")
    for e in runnable_infer:
        print(f"  [infer] {e['name']}")
    print()

    results = {}
    t_total = time.monotonic()
    job_i = 0

    # --- Sim jobs ---
    if sim_exps:
        sim_jobs = []
        for exp in sim_exps:
            base_config = load_merged_config(base_path, exp.get("overrides"))
            for ep in range(n_episodes):
                config = copy.deepcopy(base_config)
                config["simulation"]["seed"] = base_seed + ep
                config["simulation"]["export_mode"] = export_mode
                ep_dir = os.path.join("logs", exp["name"], "sim", f"episode_{ep}")
                sim_jobs.append((config, exp["name"], ep_dir))

        n_workers = min(args.parallel, len(sim_jobs))
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_sim_worker, cfg, name, rd): (name, rd)
                    for cfg, name, rd in sim_jobs
                }
                for future in as_completed(futures):
                    name, rd, summary, elapsed = future.result()
                    label = f"{name}/{os.path.basename(rd)}"
                    results[label] = summary
                    job_i += 1
                    print(f"[{job_i}/{total_jobs}] {_format_result(label, summary, elapsed)}")
        else:
            for config, name, ep_dir in sim_jobs:
                job_i += 1
                label = f"{name}/{os.path.basename(ep_dir)}"
                print(f"[{job_i}/{total_jobs}] {name} {os.path.basename(ep_dir)}...")
                t0 = time.monotonic()
                try:
                    summary = _run_sim(config, name, ep_dir, progress=args.progress)
                    elapsed = time.monotonic() - t0
                    results[label] = summary
                    print(_format_result(label, summary, elapsed))
                except Exception as e:
                    elapsed = time.monotonic() - t0
                    results[label] = {"error": str(e)}
                    print(f"  FAILED in {elapsed:.1f}s: {e}")

    # --- Infer jobs (sequential — GPU / model loading) ---
    for exp in runnable_infer:
        for ep in range(n_episodes):
            job_i += 1
            ep_dir = os.path.join("logs", exp["name"], "infer", f"episode_{ep}")
            label = f"{exp['name']}/{os.path.basename(ep_dir)}"
            print(f"[{job_i}/{total_jobs}] {exp['name']} episode_{ep}...")
            t0 = time.monotonic()
            try:
                summary = _run_infer(exp, data, base_path,
                                     base_seed, ep, ep_dir, export_mode)
                elapsed = time.monotonic() - t0
                results[label] = summary
                print(_format_result(label, summary, elapsed))
            except Exception as e:
                elapsed = time.monotonic() - t0
                results[label] = {"error": str(e)}
                print(f"  FAILED in {elapsed:.1f}s: {e}")

    total_elapsed = time.monotonic() - t_total
    passed = sum(1 for s in results.values() if "error" not in s)
    failed = len(results) - passed
    print(f"\nSummary: {passed} passed, {failed} failed / {len(results)} total")
    print(f"Total: {total_elapsed:.1f}s")
    _print_table(results)


if __name__ == "__main__":
    main()
