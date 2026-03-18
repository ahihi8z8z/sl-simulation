import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serverless-sim",
        description="SimPy-based serverless system simulator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Run standalone SimPy simulation")
    sim_parser.add_argument("--sim-config", required=True, help="Path to simulation config JSON")
    sim_parser.add_argument("--log-mode", choices=["console", "file", "both"], default="console")
    sim_parser.add_argument("--log-level", default="INFO")
    sim_parser.add_argument("--run-name", default=None, help="Optional run name suffix")
    sim_parser.add_argument("--export-mode", type=int, choices=[0, 1, 2], default=None,
                            help="Override export mode from config")

    # train
    train_parser = subparsers.add_parser("train", help="Train PPO agent")
    train_parser.add_argument("--sim-config", required=True, help="Path to simulation config JSON")
    train_parser.add_argument("--gym-config", required=True, help="Path to gym config JSON")
    train_parser.add_argument("--rl-config", required=True, help="Path to RL config JSON")
    train_parser.add_argument("--log-mode", choices=["console", "file", "both"], default="console")
    train_parser.add_argument("--log-level", default="INFO")
    train_parser.add_argument("--run-name", default=None)

    # infer
    infer_parser = subparsers.add_parser("infer", help="Run RL inference")
    infer_parser.add_argument("--sim-config", required=True, help="Path to simulation config JSON")
    infer_parser.add_argument("--gym-config", required=True, help="Path to gym config JSON")
    infer_parser.add_argument("--rl-config", required=True, help="Path to RL config JSON")
    infer_parser.add_argument("--log-mode", choices=["console", "file", "both"], default="console")
    infer_parser.add_argument("--log-level", default="INFO")
    infer_parser.add_argument("--run-name", default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "simulate":
        from serverless_sim.runtime.app import run_simulate
        run_simulate(args)
    elif args.command == "train":
        from serverless_sim.runtime.app import run_train
        run_train(args)
    elif args.command == "infer":
        from serverless_sim.runtime.app import run_infer
        run_infer(args)
