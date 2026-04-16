"""Train RL agents for experiments defined in experiments.json.

Only trains experiments that have rl_template defined.

Usage:
    python tools/train_all.py experimental/experiments.json
    python tools/train_all.py experimental/experiments.json --filter our_sac
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def train_experiment(name: str, sim_config: dict, rl_config: dict,
                     gym_config: dict | None = None) -> tuple[str, bool, str]:
    """Train a single experiment. Returns (name, success, message)."""
    # Write configs to temp files (CLI expects file paths)
    tmp_files = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sim_config, f)
            sim_path = f.name
            tmp_files.append(sim_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rl_config, f)
            rl_path = f.name
            tmp_files.append(rl_path)

        cmd = [sys.executable, "-m", "serverless_sim.runtime.cli", "train",
               "--sim-config", sim_path, "--rl-config", rl_path,
               "--run-name", name]

        if gym_config:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(gym_config, f)
                gym_path = f.name
                tmp_files.append(gym_path)
            cmd += ["--gym-config", gym_path]

        result = subprocess.run(cmd, timeout=None)
        if result.returncode == 0:
            return name, True, "OK"
        else:
            return name, False, f"Exit code {result.returncode}"
    except KeyboardInterrupt:
        return name, False, "Interrupted by user"
    except Exception as e:
        return name, False, str(e)
    finally:
        for f in tmp_files:
            os.unlink(f)


def main():
    parser = argparse.ArgumentParser(description="Train RL agents from experiments.json")
    parser.add_argument("experiments_file", help="Path to experiments.json")
    parser.add_argument("--filter", default=None, help="Only train experiments matching substring")
    args = parser.parse_args()

    from tools.config_merge import (
        load_experiments, load_merged_config,
        build_rl_config, build_gym_config,
    )

    base_path, data = load_experiments(args.experiments_file)
    experiments = data["experiments"]

    # Filter to RL experiments only
    rl_experiments = [e for e in experiments if e.get("rl_template")]
    if args.filter:
        rl_experiments = [e for e in rl_experiments if args.filter in e["name"]]

    if not rl_experiments:
        print("No matching RL experiments found")
        return

    print(f"Found {len(rl_experiments)} RL experiments:")
    for e in rl_experiments:
        print(f"  {e['name']} (template: {e['rl_template']})")
    print()

    results = []
    for i, exp in enumerate(rl_experiments, 1):
        sim_config = load_merged_config(base_path, exp.get("overrides"))
        rl_config = build_rl_config(exp, data)
        gym_config = build_gym_config(exp, data)

        print(f"{'='*60}")
        print(f"[{i}/{len(rl_experiments)}] Training: {exp['name']}")
        print(f"  algorithm: {rl_config['algorithm']}, env: {rl_config['env']}")
        print(f"{'='*60}")

        name, ok, msg = train_experiment(
            name=exp["name"],
            sim_config=sim_config,
            rl_config=rl_config,
            gym_config=gym_config,
        )
        status = "OK" if ok else f"FAILED: {msg}"
        print(f"\n  >> {name}: {status}\n")
        results.append((name, ok, msg))

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{'='*60}")
    print(f"Summary: {passed} passed, {failed} failed / {len(results)} total")
    if failed:
        for name, ok, msg in results:
            if not ok:
                print(f"  FAILED: {name}: {msg}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
