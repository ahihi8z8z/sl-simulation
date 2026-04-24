"""Run inference for trained RL experiments across multiple seeds.

For each experiment with rl_template, iterates over seeds (from exp.infer.seeds
or infer_defaults.seeds) and runs one inference episode per seed. Outputs go to
logs/<exp_name>/infer/seed_<N>/ with summary.json + request_trace.csv +
system_metrics.csv.

Usage:
    python tools/infer_all.py experimental/experiments.json
    python tools/infer_all.py experimental/experiments.json --filter our_sac
    python tools/infer_all.py experimental/experiments.json --seeds 42,43
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Run infer for RL experiments × seeds")
    parser.add_argument("experiments_file", help="Path to experiments.json")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated experiment names")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds override (e.g. 42,43)")
    args = parser.parse_args()

    from tools.config_merge import (
        load_experiments, load_merged_config,
        build_gym_config, build_infer_config, get_infer_seeds,
    )
    from rl_agent.infer import run_inference

    base_path, data = load_experiments(args.experiments_file)
    experiments = data["experiments"]

    rl_experiments = [e for e in experiments if e.get("rl_template")]
    if args.filter:
        wanted = {f.strip() for f in args.filter.split(",")}
        rl_experiments = [e for e in rl_experiments if e["name"] in wanted]

    if not rl_experiments:
        print("No matching RL experiments found")
        return

    seed_override = None
    if args.seeds:
        seed_override = [int(s.strip()) for s in args.seeds.split(",")]

    # Build job list; fail-fast on missing models
    jobs: list[tuple[dict, int]] = []
    missing: list[str] = []
    for exp in rl_experiments:
        seeds = seed_override or get_infer_seeds(exp, data)
        probe = build_infer_config(exp, data, seed=seeds[0])
        model_zip = probe["model_path"] + ".zip"
        if not os.path.exists(model_zip):
            missing.append(f"{exp['name']} → {model_zip}")
            continue
        for s in seeds:
            jobs.append((exp, s))

    if missing:
        print("Skipping experiments with no trained model:")
        for m in missing:
            print(f"  {m}")
        print()

    if not jobs:
        print("No runnable jobs")
        return

    print(f"Running {len(jobs)} infer jobs")
    for exp, s in jobs:
        print(f"  {exp['name']} seed={s}")
    print()

    results = []
    t_total = time.monotonic()
    for i, (exp, seed) in enumerate(jobs, 1):
        name = exp["name"]
        run_dir = os.path.join("logs", name, "infer", f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        sim_config = load_merged_config(base_path, exp.get("overrides"))
        sim_config["simulation"]["seed"] = seed
        # Force CSV export so compare_logs has data to aggregate
        sim_config["simulation"]["export_mode"] = 2
        gym_config = build_gym_config(exp, data)
        if gym_config is not None:
            gym_config["export_mode"] = 2
        rl_infer = build_infer_config(exp, data, seed=seed)

        sim_path = os.path.join(run_dir, "sim_config.json")
        gym_path = os.path.join(run_dir, "gym_config.json")
        rl_path = os.path.join(run_dir, "rl_infer.json")
        with open(sim_path, "w") as f: json.dump(sim_config, f, indent=2)
        with open(gym_path, "w") as f: json.dump(gym_config, f, indent=2)
        with open(rl_path, "w") as f: json.dump(rl_infer, f, indent=2)

        print(f"[{i}/{len(jobs)}] {name} seed={seed} → {run_dir}")
        t0 = time.monotonic()
        try:
            run_inference(sim_path, gym_path if gym_config else None,
                          rl_path, run_dir=run_dir)
            elapsed = time.monotonic() - t0
            # Peek at summary.json (written by simulation export)
            summary_path = os.path.join(run_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    s = json.load(f)
                req = s.get("requests", {})
                rates = s.get("rates", {})
                print(f"  OK {elapsed:.1f}s — "
                      f"total={req.get('total', 0)}, "
                      f"drop={rates.get('drop_rate_pct', 0):.2f}%, "
                      f"cold={rates.get('cold_start_rate_pct', 0):.2f}%")
                results.append((name, seed, True, s))
            else:
                print(f"  WARN {elapsed:.1f}s — no summary.json produced")
                results.append((name, seed, True, {}))
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  FAILED {elapsed:.1f}s — {e}")
            results.append((name, seed, False, {"error": str(e)}))

    total_elapsed = time.monotonic() - t_total
    passed = sum(1 for _, _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"\n{'='*60}")
    print(f"Summary: {passed} passed, {failed} failed / {len(results)} total")
    print(f"Total: {total_elapsed:.1f}s")
    if failed:
        for name, seed, ok, payload in results:
            if not ok:
                print(f"  FAILED {name} seed={seed}: {payload.get('error')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
