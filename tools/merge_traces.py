"""Merge request-per-min CSVs with runtime_cost CSVs into simulation-ready trace files.

For each {Runtime}_{Trigger}_{ID}.csv traffic file:
  1. Find matching {ID}_runtime_cost.csv
  2. Merge runtime cost as execution duration per minute
  3. Output: minute,function_id,count,duration

Usage:
    python tools/merge_traces.py [--input-dir datasets/request_per_min] [--output-dir datasets/merged]
"""

import argparse
import csv
import glob
import os
import re


def find_pairs(input_dir: str) -> list[dict]:
    """Find traffic CSVs and match with runtime_cost CSVs by ID."""
    traffic_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    pairs = []

    for path in traffic_files:
        filename = os.path.basename(path)
        # Skip costs, runtime_cost, filtered files
        if "_costs.csv" in filename or "_runtime_cost.csv" in filename or "filtered" in filename:
            continue

        # Extract: {Runtime}_{Trigger}_{ID}.csv
        # e.g. Java_APIG-S_1715---842---pool22-300-128.csv
        match = re.match(r"^(.+?)_(\d+---.*?)\.csv$", filename)
        if not match:
            continue

        prefix = match.group(1)  # e.g. Java_APIG-S
        trace_id = match.group(2)  # e.g. 1715---842---pool22-300-128

        runtime_cost_path = os.path.join(input_dir, f"{trace_id}_runtime_cost.csv")
        costs_path = os.path.join(input_dir, f"{prefix}_{trace_id}_costs.csv")

        pairs.append({
            "function_id": prefix,
            "trace_id": trace_id,
            "traffic_path": path,
            "runtime_cost_path": runtime_cost_path if os.path.exists(runtime_cost_path) else None,
            "costs_path": costs_path if os.path.exists(costs_path) else None,
        })

    return pairs


def load_traffic(path: str) -> dict[int, float]:
    """Load traffic CSV. Returns {minute_index: count}."""
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_sec = int(row["time"])
            minute = time_sec // 60
            count_str = row.get("mean_rq", "").strip()
            if count_str:
                result[minute] = float(count_str)
    return result


def load_runtime_costs(path: str) -> dict[int, float]:
    """Load runtime_cost CSV. Returns {minute_index: duration}.

    If multiple entries per minute, take the mean.
    """
    by_minute: dict[int, list[float]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = int(row["day"])
            time_sec = int(row["time"])
            minute = day * 1440 + time_sec // 60
            cost = float(row["runtimeCost"])
            by_minute.setdefault(minute, []).append(cost)

    return {m: sum(v) / len(v) for m, v in by_minute.items()}


def merge_and_write(pair: dict, output_dir: str) -> str:
    """Merge traffic + runtime_cost into simulation-ready CSV."""
    function_id = pair["function_id"]
    traffic = load_traffic(pair["traffic_path"])

    runtime_costs = {}
    if pair["runtime_cost_path"]:
        runtime_costs = load_runtime_costs(pair["runtime_cost_path"])

    # Compute default duration from costs CSV (mean total)
    default_duration = 0.0
    if pair["costs_path"]:
        with open(pair["costs_path"]) as f:
            reader = csv.DictReader(f)
            totals = []
            for row in reader:
                total = (float(row.get("podAllocationCost", 0) or 0)
                         + float(row.get("deployCodeCost", 0) or 0)
                         + float(row.get("deployDependencyCost", 0) or 0))
                totals.append(total)
            if totals:
                default_duration = sum(totals) / len(totals)

    # All minutes that have traffic
    all_minutes = sorted(traffic.keys())
    if not all_minutes:
        print(f"  Warning: no traffic data for {function_id}, skipping")
        return ""

    output_path = os.path.join(output_dir, f"{function_id}_{pair['trace_id']}.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["minute", "function_id", "count", "duration"])
        for minute in all_minutes:
            count = traffic[minute]
            if count <= 0:
                continue
            duration = runtime_costs.get(minute, default_duration)
            writer.writerow([minute, function_id, int(round(count)), f"{duration:.6f}"])

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge traffic + runtime_cost into simulation trace")
    parser.add_argument(
        "--input-dir",
        default="datasets/request_per_min",
        help="Directory containing traffic and runtime_cost CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/merged",
        help="Output directory for merged trace CSVs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = find_pairs(args.input_dir)
    if not pairs:
        print(f"No traffic CSVs found in {args.input_dir}")
        return

    print(f"Found {len(pairs)} functions:")
    for p in pairs:
        rc = "yes" if p["runtime_cost_path"] else "no"
        cs = "yes" if p["costs_path"] else "no"
        print(f"  {p['function_id']} — runtime_cost: {rc}, costs: {cs}")
    print()

    for pair in pairs:
        path = merge_and_write(pair, args.output_dir)
        if path:
            # Count rows
            with open(path) as f:
                n = sum(1 for _ in f) - 1
            print(f"  Saved: {path} ({n} rows)")

    print(f"\nAll merged traces saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
