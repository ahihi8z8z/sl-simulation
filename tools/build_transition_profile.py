"""Build transition_profile CSV from *_costs.csv files.

Maps cost columns to transitions:
  podAllocationCost    → null → prewarm
  deployCodeCost       → prewarm → code_loaded
  deployDependencyCost → code_loaded → warm

Transition cpu/memory = resource usage of the source state (from --state-resources CSV).

Usage:
    python tools/build_transition_profile.py [--input-dir datasets/request_per_min] [--output-dir datasets/profiles]
    python tools/build_transition_profile.py --state-resources resources.csv ...

State resources CSV format:
    state,cpu,memory
    null,0,0
    prewarm,0.1,128
    code_loaded,0.3,256
    warm,0.5,512
    running,1.0,512
"""

import argparse
import csv
import glob
import os


COST_TO_TRANSITION = {
    "podAllocationCost": ("null", "prewarm"),
    "deployCodeCost": ("prewarm", "code_loaded"),
    "deployDependencyCost": ("code_loaded", "warm"),
}


def load_state_resources(path: str) -> dict[str, tuple[float, float]]:
    """Load state resource CSV. Returns {state: (cpu, memory)}."""
    resources = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            resources[row["state"]] = (
                float(row.get("cpu", 0)),
                float(row.get("memory", 0)),
            )
    return resources


def build_profile(costs_path: str, output_path: str,
                  state_resources: dict[str, tuple[float, float]] | None = None) -> int:
    """Convert one costs CSV to transition_profile CSV. Returns row count."""
    rows = []
    with open(costs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for cost_col, (from_s, to_s) in COST_TO_TRANSITION.items():
                val = row.get(cost_col, "0")
                time_val = float(val) if val else 0.0
                if time_val > 0:
                    # Transition resource = source state resource
                    if state_resources and from_s in state_resources:
                        cpu, mem = state_resources[from_s]
                    else:
                        cpu, mem = 0.0, 0.0
                    rows.append([from_s, to_s, f"{time_val:.6f}", f"{cpu}", f"{mem}"])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from_state", "to_state", "time", "cpu", "memory"])
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Build transition profiles from costs CSVs")
    parser.add_argument(
        "--input-dir",
        default="datasets/request_per_min",
        help="Directory containing *_costs.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/profiles",
        help="Output directory for transition profile CSVs",
    )
    parser.add_argument(
        "--state-resources",
        default=None,
        help="CSV file with resource usage per state (state,cpu,memory). "
             "Transition resource = source state resource.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    state_resources = None
    if args.state_resources:
        state_resources = load_state_resources(args.state_resources)
        print(f"State resources loaded from {args.state_resources}:")
        for state, (cpu, mem) in state_resources.items():
            print(f"  {state}: cpu={cpu}, memory={mem}")
        print()

    cost_files = sorted(glob.glob(os.path.join(args.input_dir, "*_costs.csv")))
    if not cost_files:
        print(f"No *_costs.csv files found in {args.input_dir}")
        return

    print(f"Found {len(cost_files)} costs files:\n")

    for costs_path in cost_files:
        filename = os.path.basename(costs_path)
        label = filename.replace("_costs.csv", "")
        output_path = os.path.join(args.output_dir, f"{label}_transitions.csv")
        n = build_profile(costs_path, output_path, state_resources=state_resources)
        print(f"  {label}: {n} rows → {output_path}")

    print(f"\nAll profiles saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
