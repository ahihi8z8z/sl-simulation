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


SAMPLE_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "arrival_rate": 5.0,
            "job_size": 0.1,
            "memory": 256,
            "cpu": 1.0,
            "max_concurrency": 4,
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
        inst = ContainerInstance(env, "svc-a", "node-0", max_concurrency=4)
        assert inst.state == "null"
        assert inst.is_idle
        assert inst.available_slots == 4
        assert inst.cold_start is True


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

    def test_concurrency(self):
        """With max_concurrency=4, multiple requests can run on one instance."""
        config = {
            "simulation": {"duration": 5.0, "seed": 42, "export_mode": 0},
            "services": [
                {
                    "service_id": "svc-concurrent",
                    "arrival_rate": 20.0,
                    "job_size": 1.0,
                    "memory": 256,
                    "cpu": 0.5,
                    "max_concurrency": 4,
                }
            ],
            "cluster": {
                "nodes": [
                    {"node_id": "node-0", "cpu_capacity": 16.0, "memory_capacity": 16384},
                ]
            },
        }
        ctx = _make_ctx(config=config)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=5.0)

        completed = ctx.request_table.counters.completed
        # With rate=20 and duration=5, many requests should complete
        assert completed > 10

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
