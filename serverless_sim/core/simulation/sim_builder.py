from __future__ import annotations

import logging

import numpy as np
import simpy

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.export.export_manager import ExportManager
from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler
from serverless_sim.controller.base_controller import BaseController
from serverless_sim.controller.policies.threshold_policy import ThresholdPolicy


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

        # Cluster
        ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)

        # Workload
        ctx.workload_manager = WorkloadManager.from_config(ctx)

        # Lifecycle
        sm = OpenWhiskExtendedStateMachine.from_config(config)
        ctx.lifecycle_manager = LifecycleManager(ctx, state_machine=sm)

        # Load balancer
        ctx.dispatcher = ShardingContainerPoolBalancer(ctx)

        # Wire context into nodes
        ctx.cluster_manager.set_context(ctx)

        # Autoscaling
        auto_cfg = config.get("autoscaling", {})
        if auto_cfg.get("enabled", False):
            autoscaler = OpenWhiskPoolAutoscaler(
                ctx,
                reconcile_interval=auto_cfg.get("reconcile_interval", 5.0),
            )
            ctx.autoscaling_manager = autoscaler

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
            policy = ThresholdPolicy(
                cpu_high=ctrl_cfg.get("cpu_high", 0.8),
                cpu_low=ctrl_cfg.get("cpu_low", 0.3),
                prewarm_max=ctrl_cfg.get("prewarm_max", 10),
            )
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
