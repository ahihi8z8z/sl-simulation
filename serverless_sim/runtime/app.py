import os
from datetime import datetime

from serverless_sim.core.config.loader import load_config
from serverless_sim.core.logging.logger_factory import create_logger


def _create_run_dir(base: str = "logs", run_name: str | None = None) -> str:
    """Create a timestamped run directory under *base*."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"run_{timestamp}"
    if run_name:
        folder += f"_{run_name}"
    run_dir = os.path.join(base, folder)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_simulate(args):
    config = load_config(args.sim_config)

    run_dir = _create_run_dir(run_name=getattr(args, "run_name", None))

    logger = create_logger(
        module_name="serverless_sim",
        run_dir=run_dir,
        mode=args.log_mode,
        level=args.log_level,
    )

    logger.info("Config loaded successfully from %s", args.sim_config)
    logger.info("Run directory: %s", run_dir)
    logger.info(
        "Simulation config: duration=%.1f, seed=%d, services=%d, nodes=%d",
        config["simulation"]["duration"],
        config["simulation"]["seed"],
        len(config["services"]),
        len(config["cluster"]["nodes"]),
    )

    # TODO: Step 9 — wire SimulationBuilder + SimulationEngine here


def run_train(args):
    print(f"[train] sim-config={args.sim_config} (not yet implemented)")


def run_infer(args):
    print(f"[infer] sim-config={args.sim_config} (not yet implemented)")
