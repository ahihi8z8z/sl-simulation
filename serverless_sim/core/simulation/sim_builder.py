from __future__ import annotations

import logging

import numpy as np
import simpy

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import create_load_balancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.export.export_manager import ExportManager
from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler
from serverless_sim.controller.base_controller import BaseController
from serverless_sim.controller.policies.base_policy import BaseControlPolicy


class SimulationBuilder:
    """Constructs the entire simulator from config."""

    def build(
        self,
        config: dict,
        run_dir: str,
        logger: logging.Logger,
        export_mode_override: int | None = None,
    ) -> SimContext:
        """Build all components and return a wired SimContext."""
        env = simpy.Environment()
        seed = config["simulation"]["seed"]
        rng = np.random.default_rng(seed)

        ctx = SimContext(
            env=env,
            config=config,
            rng=rng,
            logger=logger,
            run_dir=run_dir,
        )

        # Service time provider
        from serverless_sim.workload.service_time import create_service_time_provider
        ctx.service_time_provider = create_service_time_provider(config)

        # Cluster
        ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)

        # Workload (each service builds its own state machine from config)
        ctx.workload_manager = WorkloadManager.from_config(ctx)

        # Lifecycle (per-service state machines are in ServiceClass)
        ctx.lifecycle_manager = LifecycleManager(ctx)

        # Load balancer (config-driven: scheduling.strategy)
        ctx.dispatcher = create_load_balancer(ctx)

        # Wire context into nodes
        ctx.cluster_manager.set_context(ctx)

        # Autoscaling
        auto_cfg = config.get("autoscaling", {})
        if auto_cfg.get("enabled", False):
            pool_mode = auto_cfg.get("pool_mode", "per_node")
            autoscaler = OpenWhiskPoolAutoscaler(
                ctx,
                pool_mode=pool_mode,
            )
            ctx.autoscaling_manager = autoscaler

            # Placement strategy (used by global pool mode)
            from serverless_sim.scheduling.placement_strategy import create_placement_strategy
            ctx.placement_strategy = create_placement_strategy(
                auto_cfg.get("placement_strategy", "best_fit")
            )

        # Monitoring
        mon_cfg = config.get("monitoring", {})
        ctx.monitor_manager = MonitorManager(
            ctx,
            interval=mon_cfg.get("interval", 1.0),
            max_history=mon_cfg.get("max_history_length", 1000),
        )

        # Controller
        ctrl_cfg = config.get("controller", {})
        if ctrl_cfg.get("enabled", False) and ctx.autoscaling_manager is not None:
            policy_type = ctrl_cfg.get("policy", "noop")

            if policy_type == "predictive":
                from serverless_sim.controller.policies.predictive_policy import PredictivePolicy
                policy = PredictivePolicy(
                    predict_path=ctrl_cfg["predict_path"],
                    predict_column=ctrl_cfg.get("predict_column", "predicted_count"),
                    predict_scale=ctrl_cfg.get("predict_scale", 1.0),
                    avg_duration=ctrl_cfg.get("avg_duration", 0.0),
                    interval=ctrl_cfg.get("interval", 3600.0),
                )
            else:
                policy = BaseControlPolicy()  # no-op

            ctx.controller = BaseController(
                ctx,
                policy=policy,
                interval=ctrl_cfg.get("interval", 5.0),
            )

        # Export
        export_mode = export_mode_override
        if export_mode is None:
            export_mode = config.get("simulation", {}).get("export_mode", 0)
        ctx.export_manager = ExportManager(ctx, mode=export_mode)

        logger.info("SimulationBuilder: all components wired")
        return ctx
