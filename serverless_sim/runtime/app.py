import os
from datetime import datetime

from serverless_sim.core.config.loader import load_config
from serverless_sim.core.logging.logger_factory import create_logger
from serverless_sim.core.simulation.sim_builder import SimulationBuilder
from serverless_sim.core.simulation.sim_engine import SimulationEngine


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

    # Build
    export_mode = getattr(args, "export_mode", None)
    builder = SimulationBuilder()
    ctx = builder.build(config, run_dir, logger, export_mode_override=export_mode)

    # Run
    engine = SimulationEngine(ctx)
    engine.setup()
    engine.run()
    engine.shutdown()

    logger.info("Done. Results in %s", run_dir)


def run_train(args):
    from serverless_sim.rl_agent.train import run_training

    run_dir = _create_run_dir(run_name=getattr(args, "run_name", None) or "train")
    run_training(
        sim_config_path=args.sim_config,
        gym_config_path=args.gym_config,
        rl_config_path=args.rl_config,
        run_dir=run_dir,
    )
    print(f"Training complete. Results in {run_dir}")


def run_infer(args):
    from serverless_sim.rl_agent.infer import run_inference

    run_dir = _create_run_dir(run_name=getattr(args, "run_name", None) or "infer")
    run_inference(
        sim_config_path=args.sim_config,
        gym_config_path=args.gym_config,
        rl_config_path=args.rl_config,
        run_dir=run_dir,
    )
    print(f"Inference complete. Results in {run_dir}")
