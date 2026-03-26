"""Build state resource CSVs from runtime_costs data.

Reads memoryUsage (MB) and cpuUsage (cores) from runtime_costs CSVs
and outputs all observations as running state entries.

Output format matches state_profile CSV (state,cpu,memory) with one
row per observation for the running state.

Usage:
    python tools/build_state_resources.py [--input-dir datasets/runtime_costs] [--output-dir datasets/state_resources]
"""

import argparse
import csv
import glob
import os


def build_state_resources(runtime_costs_path: str, output_path: str) -> dict:
    """Build state resource CSV from runtime_costs. Returns stats dict."""
    rows = []

    with open(runtime_costs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mem = row.get("memoryUsage", "")
            cpu = row.get("cpuUsage", "")
            if mem and cpu:
                rows.append((float(cpu), float(mem)))

    if not rows:
        return {}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "cpu", "memory"])
        for cpu, mem in rows:
            writer.writerow(["running", f"{cpu:.6f}", f"{mem:.4f}"])

    cpus = [r[0] for r in rows]
    mems = [r[1] for r in rows]
    return {
        "n_samples": len(rows),
        "cpu_mean": sum(cpus) / len(cpus),
        "cpu_min": min(cpus),
        "cpu_max": max(cpus),
        "mem_mean": sum(mems) / len(mems),
        "mem_min": min(mems),
        "mem_max": max(mems),
    }


def main():
    parser = argparse.ArgumentParser(description="Build state resource CSVs from runtime_costs")
    parser.add_argument(
        "--input-dir",
        default="datasets/runtime_costs",
        help="Directory containing runtime_costs CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/state_resources",
        help="Output directory for state resource CSVs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(files)} runtime_costs files:\n")

    for path in files:
        filename = os.path.basename(path)
        label = filename.replace(".csv", "")
        output_path = os.path.join(args.output_dir, f"{label}_state_resources.csv")

        stats = build_state_resources(path, output_path)
        if stats:
            print(f"  {label}: {stats['n_samples']} rows")
            print(f"    cpu:  mean={stats['cpu_mean']:.4f}  min={stats['cpu_min']:.4f}  max={stats['cpu_max']:.4f}")
            print(f"    mem:  mean={stats['mem_mean']:.2f}  min={stats['mem_min']:.2f}  max={stats['mem_max']:.2f} MB")
            print(f"    → {output_path}")
        else:
            print(f"  {label}: no data, skipped")

    print(f"\nAll state resources saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
