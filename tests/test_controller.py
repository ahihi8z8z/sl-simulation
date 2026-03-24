"""Unit tests for Step 12: Controller + ThresholdPolicy."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.controller.base_controller import BaseController
from serverless_sim.controller.policies.threshold_policy import ThresholdPolicy


LIFECYCLE_256_1 = {
    "cold_start_chain": ["null", "prewarm", "warm"],
    "states": [
        {"name": "null", "category": "stable", "cpu": 0, "memory": 0},
        {"name": "prewarm", "category": "stable", "cpu": 0, "memory": 128},
        {"name": "warm", "category": "stable", "cpu": 0.1, "memory": 256, "service_bound": True, "reusable": True},
        {"name": "running", "category": "transient", "cpu": 1.0, "memory": 256, "service_bound": True, "reusable": False},
        {"name": "evicted", "category": "stable", "cpu": 0, "memory": 0, "reusable": False},
    ],
    "transitions": [
        {"from": "null", "to": "prewarm", "time": 0.5},
        {"from": "prewarm", "to": "warm", "time": 0.3},
        {"from": "warm", "to": "running", "time": 0.0},
        {"from": "running", "to": "warm", "time": 0.0},
        {"from": "warm", "to": "evicted", "time": 0.0},
        {"from": "prewarm", "to": "evicted", "time": 0.0},
    ],
}

CONFIG = {
    "simulation": {"duration": 20.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "arrival_rate": 5.0,
            "job_size": 0.1,
            "max_concurrency": 4,
            "lifecycle": LIFECYCLE_256_1,
        }
    ],
    "cluster": {
        "nodes": [
            {"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192},
        ]
    },
}


def _make_ctx():
    env = simpy.Environment()
    rng = np.random.default_rng(42)
    logger = logging.getLogger("test_controller")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    ctx = SimContext(env=env, config=CONFIG, rng=rng, logger=logger, run_dir="/tmp/test_run")
    ctx.cluster_manager = ClusterManager(env=env, config=CONFIG, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    ctx.monitor_manager = MonitorManager(ctx, interval=1.0)
    ctx.autoscaling_manager = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=2.0)
    return ctx


class TestThresholdPolicy:
    def test_high_cpu_increases_pool_target(self):
        ctx = _make_ctx()
        # Set initial pool target so there's something to increase from
        ctx.autoscaling_manager.set_pool_target("svc-a", "prewarm", 1)
        policy = ThresholdPolicy(cpu_high=0.5, cpu_low=0.2, prewarm_max=10)
        snapshot = {"cluster.cpu_utilization": 0.8}  # > 0.5
        actions = policy.decide(snapshot, ctx)
        pool_actions = [a for a in actions if a["action"] == "set_pool_target"]
        assert len(pool_actions) > 0
        assert pool_actions[0]["value"] > 1

    def test_low_cpu_decreases_pool_target(self):
        ctx = _make_ctx()
        # Set initial pool target so there's something to decrease from
        ctx.autoscaling_manager.set_pool_target("svc-a", "prewarm", 1)
        policy = ThresholdPolicy(cpu_high=0.8, cpu_low=0.3, prewarm_min=0)
        snapshot = {"cluster.cpu_utilization": 0.1}  # < 0.3
        actions = policy.decide(snapshot, ctx)
        pool_actions = [a for a in actions if a["action"] == "set_pool_target"]
        assert len(pool_actions) > 0
        assert pool_actions[0]["value"] == 0

    def test_normal_cpu_no_actions(self):
        ctx = _make_ctx()
        policy = ThresholdPolicy(cpu_high=0.8, cpu_low=0.2)
        snapshot = {"cluster.cpu_utilization": 0.5}
        actions = policy.decide(snapshot, ctx)
        assert len(actions) == 0


class TestBaseController:
    def test_controller_runs_steps(self):
        ctx = _make_ctx()
        policy = ThresholdPolicy()
        controller = BaseController(ctx, policy=policy, interval=2.0)
        ctx.controller = controller

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.monitor_manager.start()
        ctx.autoscaling_manager.start()
        controller.start()

        ctx.env.run(until=20.0)

        assert controller.step_count > 0
        # At least ~9 steps in 20s with interval=2
        assert controller.step_count >= 8
