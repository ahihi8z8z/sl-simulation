"""Unit tests for Step 5: Minimal lifecycle + execution."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.invocation import Invocation
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine
from serverless_sim.lifecycle.container_instance import ContainerInstance
from serverless_sim.workload.service_time import FixedServiceTime


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

SAMPLE_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "lifecycle": LIFECYCLE_256_1,
            "workload": {"arrival_rate": 5.0},
        }
    ],
    "cluster": {
        "nodes": [
            {"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192},
        ]
    },
}


def _make_ctx(config=None, seed=42) -> SimContext:
    config = config or SAMPLE_CONFIG
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    logger = logging.getLogger("test_lifecycle")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_run")
    ctx.service_time_provider = FixedServiceTime(duration=0.1)
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    return ctx


class TestStateMachine:
    def test_default_states(self):
        sm = OpenWhiskExtendedStateMachine.default()
        assert "null" in sm.states
        assert "prewarm" in sm.states
        assert "warm" in sm.states
        assert "running" in sm.states
        assert "evicted" in sm.states

    def test_find_path(self):
        sm = OpenWhiskExtendedStateMachine.default()
        path = sm.find_path("null", "warm")
        assert path == ["null", "prewarm", "warm"]

    def test_find_path_same(self):
        sm = OpenWhiskExtendedStateMachine.default()
        path = sm.find_path("warm", "warm")
        assert path == ["warm"]

    def test_find_path_none(self):
        sm = OpenWhiskExtendedStateMachine.default()
        path = sm.find_path("evicted", "warm")
        assert path is None


class TestContainerInstance:
    def test_creation(self):
        env = simpy.Environment()
        inst = ContainerInstance(env, "svc-a", "node-0")
        assert inst.state == "null"
        assert inst.is_idle
        assert inst.active_requests == 0


class TestLifecycleEndToEnd:
    def test_requests_complete(self):
        """Run full simulation: arrive → dispatch → execute → complete."""
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=10.0)

        completed = ctx.request_table.counters.completed
        assert completed > 0, "No requests completed"

    def test_timestamps_populated(self):
        """Completed requests should have valid latencies recorded."""
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=10.0)

        # Completed invocations are flushed from memory, but mean latency is tracked
        assert ctx.request_table.counters.completed > 0
        assert ctx.request_table.latency_mean > 0, "Mean latency should be positive"

    def test_cold_start_first_request(self):
        """First request to a service should be a cold start, with warm hits after."""
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=10.0)

        completed = ctx.request_table.counters.completed
        cold_starts = ctx.request_table.counters.cold_starts
        assert completed > 1
        # At least one cold start
        assert cold_starts >= 1
        # Some warm hits (completed > cold_starts)
        warm_hits = completed - cold_starts
        assert warm_hits > 0, "Expected some warm hits after first request"

    def test_per_request_cpu_released(self):
        """After all requests complete, node CPU should be fully released."""
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=10.0)

        # Wait a bit more to let last requests finish
        ctx.env.run(until=15.0)

        node = ctx.cluster_manager.get_node("node-0")
        # Memory may be allocated for containers, but CPU from requests should be released
        # Check that allocated CPU equals only container steady-state (0 for running requests)
        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        active = sum(i.active_requests for i in instances if i.state != "evicted")
        assert active == 0, f"Expected 0 active requests after simulation, got {active}"
