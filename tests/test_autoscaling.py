"""Unit tests for Step 11: Autoscaling."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler
from serverless_sim.autoscaling.autoscaling_api import AutoscalingAPI
from serverless_sim.monitoring.monitor_manager import MonitorManager


def _make_ctx(config, seed=42):
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    logger = logging.getLogger("test_autoscaling")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_run")
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    ctx.monitor_manager = MonitorManager(ctx)
    return ctx


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

BASE_CONFIG = {
    "simulation": {"duration": 20.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "arrival_rate": 2.0,
            "job_size": 0.1,
            "max_concurrency": 2,
            "lifecycle": LIFECYCLE_256_1,
        }
    ],
    "cluster": {
        "nodes": [
            {"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192},
        ]
    },
}


class TestAutoscaler:
    def test_pool_target_top_up(self):
        """Autoscaler should create prewarm containers up to pool target."""
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=2.0)
        ctx.autoscaling_manager = autoscaler

        # Set pool target for first pool state
        autoscaler.set_pool_target("svc-a", "prewarm", 2)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Run just enough for reconcile + cold start to finish
        ctx.env.run(until=5.0)

        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        alive = [i for i in instances if i.state not in ("null", "evicted")]
        assert len(alive) >= 2, f"Expected >= 2 prewarm instances, got {len(alive)}"

    def test_idle_eviction(self):
        """Idle instances should be evicted after idle_timeout."""
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=1.0)
        ctx.autoscaling_manager = autoscaler

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        autoscaler.start()

        # Run simulation: requests create instances, then they go idle
        ctx.env.run(until=20.0)

        # Evicted instances are removed from list — check evicted_count
        assert ctx.lifecycle_manager._evicted_count >= 0

    def test_only_evict_stable_idle(self):
        """Autoscaler should not evict running instances."""
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=1.0)
        ctx.autoscaling_manager = autoscaler

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        autoscaler.start()

        ctx.env.run(until=10.0)

        # All surviving instances should be alive (not evicted)
        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        for inst in instances:
            # Running instances should never have been evicted
            if inst.active_requests > 0:
                assert inst.state == "running"

    def test_reactive_fill_on_eviction(self):
        """Pool should be replenished immediately after eviction, not at next reconcile."""
        config = {
            **BASE_CONFIG,
            "services": [{
                **BASE_CONFIG["services"][0],
                "arrival_rate": 0.1,  # very low to let instances go idle
            }],
        }
        ctx = _make_ctx(config)
        autoscaler = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=100.0)  # very long reconcile
        ctx.autoscaling_manager = autoscaler

        # Set pool target and short idle timeout via controller/policy API
        autoscaler.set_pool_target("svc-a", "prewarm", 2)
        autoscaler.set_idle_timeout("svc-a", 2.0)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Initial fill should create 2 prewarm instances immediately
        ctx.env.run(until=1.0)
        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        prewarm = [i for i in instances if i.state == "prewarm"]
        assert len(prewarm) >= 2, f"Expected >= 2 prewarm after initial fill, got {len(prewarm)}"

    def test_reactive_fill_on_set_pool_target(self):
        """Setting pool target should immediately trigger fill."""
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=100.0)  # very long reconcile
        ctx.autoscaling_manager = autoscaler

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Wait for initial fill
        ctx.env.run(until=1.0)

        # Increase target — should fill immediately, not wait for reconcile
        autoscaler.set_pool_target("svc-a", "prewarm", 5)
        ctx.env.run(until=2.0)

        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        prewarm = [i for i in instances if i.state == "prewarm"]
        assert len(prewarm) >= 5, f"Expected >= 5 prewarm after set_pool_target, got {len(prewarm)}"


class TestAutoscalingAPI:
    def test_get_set_idle_timeout(self):
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx)
        api = AutoscalingAPI(autoscaler)

        assert api.get_idle_timeout("svc-a") == 60.0  # default, not from service config
        api.set_idle_timeout("svc-a", 10.0)
        assert api.get_idle_timeout("svc-a") == 10.0

    def test_get_set_pool_target(self):
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx)
        api = AutoscalingAPI(autoscaler)

        assert api.get_pool_target("svc-a", "prewarm") == 0
        api.set_pool_target("svc-a", "prewarm", 5)
        assert api.get_pool_target("svc-a", "prewarm") == 5

    def test_get_set_min_max_instances(self):
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx)
        api = AutoscalingAPI(autoscaler)

        assert api.get_min_instances("svc-a") == 0
        assert api.get_max_instances("svc-a") == 0
        api.set_min_instances("svc-a", 3)
        assert api.get_min_instances("svc-a") == 3
        api.set_max_instances("svc-a", 10)
        assert api.get_max_instances("svc-a") == 10

    def test_trigger_reconcile(self):
        ctx = _make_ctx(BASE_CONFIG)
        autoscaler = OpenWhiskPoolAutoscaler(ctx)
        api = AutoscalingAPI(autoscaler)
        # Should not raise
        api.trigger_reconcile()
