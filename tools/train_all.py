"""Train all services in a folder (e.g. experimental/lstm_baseline/).

Usage:
    python tools/train_all.py experimental/lstm_baseline/
    python tools/train_all.py experimental/lstm_baseline/ --parallel 2
    python tools/train_all.py experimental/lstm_baseline/Java_APIG-S/
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_services(base_dir: str) -> list[str]:
    """Find service directories containing rl_train.json."""
    # Single service dir
    if os.path.exists(os.path.join(base_dir, "rl_train.json")):
        return [base_dir]
    # Parent dir with multiple services
    services = sorted(glob.glob(os.path.join(base_dir, "*", "rl_train.json")))
    return [os.path.dirname(s) for s in services]


def train_service(svc_dir: str) -> tuple[str, bool, str]:
    """Train a single service. Returns (name, success, message)."""
    name = os.path.basename(svc_dir)
    config = os.path.join(svc_dir, "config.json")
    gym = os.path.join(svc_dir, "gym_config.json")
    rl = os.path.join(svc_dir, "rl_train.json")

    if not os.path.exists(config):
        return name, False, f"Missing {config}"
    if not os.path.exists(rl):
        return name, False, f"Missing {rl}"

    cmd = [sys.executable, "-m", "serverless_sim", "train",
           "--sim-config", config, "--rl-config", rl]
    if os.path.exists(gym):
        cmd += ["--gym-config", gym]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        if result.returncode == 0:
            return name, True, "OK"
        else:
            return name, False, result.stderr[-500:] if result.stderr else "Unknown error"
    except Exception as e:
        return name, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Train all services in a folder")
    parser.add_argument("base_dir", help="Folder with service subdirectories")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel training jobs (default: 1)")
    args = parser.parse_args()

    services = find_services(args.base_dir)
    if not services:
        print(f"No services found in {args.base_dir}")
        return

    print(f"Found {len(services)} services:")
    for s in services:
        print(f"  {os.path.basename(s)}")
    print()

    if args.parallel <= 1:
        for svc_dir in services:
            name = os.path.basename(svc_dir)
            print(f"{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")
            name, ok, msg = train_service(svc_dir)
            status = "OK" if ok else f"FAILED: {msg}"
            print(f"  {name}: {status}\n")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(train_service, s): s for s in services}
            for future in as_completed(futures):
                name, ok, msg = future.result()
                status = "OK" if ok else f"FAILED: {msg}"
                print(f"  {name}: {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
