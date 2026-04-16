"""Tests for find_reusable_instance() with running instances (max_concurrency > 1 bug fix)."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.lifecycle.container_instance import ContainerInstance


LIFECYCLE = {
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
    "simulation": {"duration": 100.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "job_size": 1.0,
            "max_concurrency": 4,
            "lifecycle": LIFECYCLE,
        }
    ],
    "cluster": {
        "nodes": [
            {"node_id": "node-0", "cpu_capacity": 16.0, "memory_capacity": 8192},
        ]
    },
}


def _make_ctx(config=None) -> SimContext:
    config = config or CONFIG
    env = simpy.Environment()
    rng = np.random.default_rng(42)
    logger = logging.getLogger("test_reusable")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_reusable")
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    return ctx


def _make_instance(ctx, state: str, max_concurrency: int = 4, active_slots: int = 0) -> ContainerInstance:
    """Create a ContainerInstance with the given state and active_requests count."""
    env = ctx.env
    inst = ContainerInstance(env, "svc-a", "node-0", max_concurrency=max_concurrency)
    inst.state = state
    inst.active_requests = active_slots  # directly set tracked concurrency
    ctx.lifecycle_manager.instances.setdefault("node-0", []).append(inst)
    return inst


class TestFindReusableInstance:
    def test_running_instance_with_free_slots_is_reusable(self):
        """A running instance with free slots must be returned (core bug fix)."""
        ctx = _make_ctx()
        node = ctx.cluster_manager.get_enabled_nodes()[0]
        inst = _make_instance(ctx, state="running", max_concurrency=4, active_slots=1)
        assert inst.active_requests == 1
        assert inst.max_concurrency - inst.active_requests == 3  # 3 free slots

        result = ctx.lifecycle_manager.find_reusable_instance(node, "svc-a")
        assert result is inst, "Should reuse running instance with free slots"

    def test_running_instance_fully_occupied_not_reusable(self):
        """A running instance with no free slots must NOT be returned."""
        ctx = _make_ctx()
        node = ctx.cluster_manager.get_enabled_nodes()[0]
        inst = _make_instance(ctx, state="running", max_concurrency=2, active_slots=2)
        assert inst.active_requests == inst.max_concurrency  # fully occupied

        result = ctx.lifecycle_manager.find_reusable_instance(node, "svc-a")
        assert result is None

    def test_warm_instance_still_found(self):
        """Warm instances are still returned (backward compatibility)."""
        ctx = _make_ctx()
        node = ctx.cluster_manager.get_enabled_nodes()[0]
        inst = _make_instance(ctx, state="warm", max_concurrency=4, active_slots=0)

        result = ctx.lifecycle_manager.find_reusable_instance(node, "svc-a")
        assert result is inst

    def test_warm_preferred_over_running(self):
        """Warm instance preferred over running instance (no transition cost)."""
        ctx = _make_ctx()
        node = ctx.cluster_manager.get_enabled_nodes()[0]
        running_inst = _make_instance(ctx, state="running", max_concurrency=4, active_slots=1)
        warm_inst = _make_instance(ctx, state="warm", max_concurrency=4, active_slots=0)

        result = ctx.lifecycle_manager.find_reusable_instance(node, "svc-a")
        assert result is warm_inst, "Warm should be preferred over running"

    def test_wrong_service_ignored(self):
        """Instances for a different service are not returned."""
        ctx = _make_ctx()
        node = ctx.cluster_manager.get_enabled_nodes()[0]
        inst = _make_instance(ctx, state="running", max_concurrency=4, active_slots=0)
        inst.service_id = "svc-other"

        result = ctx.lifecycle_manager.find_reusable_instance(node, "svc-a")
        assert result is None
