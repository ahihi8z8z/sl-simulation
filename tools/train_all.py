"""Train all services in a folder (e.g. experimental/lstm_baseline/).

Usage:
    python tools/train_all.py experimental/lstm_baseline/
    python tools/train_all.py experimental/
    python tools/train_all.py experimental/lstm_baseline/Java_APIG-S/
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def find_services(base_dir: str) -> list[str]:
    """Find service directories containing rl_train.json (recursive)."""
    if os.path.exists(os.path.join(base_dir, "rl_train.json")):
        return [base_dir]
    services = sorted(glob.glob(os.path.join(base_dir, "**", "rl_train.json"), recursive=True))
    return [os.path.dirname(s) for s in services]


def train_service(svc_dir: str) -> tuple[str, bool, str]:
    """Train a single service. Returns (label, success, message)."""
    name = os.path.basename(svc_dir)
    group = os.path.basename(os.path.dirname(svc_dir))
    label = f"{group}/{name}"
    config = os.path.join(svc_dir, "config.json")
    gym = os.path.join(svc_dir, "gym_config.json")
    rl = os.path.join(svc_dir, "rl_train.json")

    if not os.path.exists(config):
        return label, False, f"Missing {config}"
    if not os.path.exists(rl):
        return label, False, f"Missing {rl}"

    cmd = [sys.executable, "-m", "serverless_sim", "train",
           "--sim-config", config, "--rl-config", rl]
    if os.path.exists(gym):
        cmd += ["--gym-config", gym]

    try:
        result = subprocess.run(cmd, timeout=None)
        if result.returncode == 0:
            return label, True, "OK"
        else:
            return label, False, f"Exit code {result.returncode}"
    except KeyboardInterrupt:
        return label, False, "Interrupted by user"
    except Exception as e:
        return label, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Train all services in a folder")
    parser.add_argument("base_dir", help="Folder with service subdirectories")
    args = parser.parse_args()

    services = find_services(args.base_dir)
    if not services:
        print(f"No services found in {args.base_dir}")
        return

    print(f"Found {len(services)} services:")
    for s in services:
        group = os.path.basename(os.path.dirname(s))
        print(f"  {group}/{os.path.basename(s)}")
    print()

    results = []
    for i, svc_dir in enumerate(services, 1):
        group = os.path.basename(os.path.dirname(svc_dir))
        name = os.path.basename(svc_dir)
        print(f"{'='*60}")
        print(f"[{i}/{len(services)}] Training: {group}/{name}")
        print(f"{'='*60}")
        label, ok, msg = train_service(svc_dir)
        status = "OK" if ok else f"FAILED: {msg}"
        print(f"\n  >> {label}: {status}\n")
        results.append((label, ok, msg))

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{'='*60}")
    print(f"Summary: {passed} passed, {failed} failed / {len(results)} total")
    if failed:
        for label, ok, msg in results:
            if not ok:
                print(f"  FAILED: {label}: {msg}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
