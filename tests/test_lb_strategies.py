"""Tests for pluggable load balancer strategies."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.invocation import Invocation
from serverless_sim.scheduling.load_balancer import (
    HashRingBalancer,
    RoundRobinBalancer,
    LeastLoadedBalancer,
    PowerOfTwoChoicesBalancer,
    create_load_balancer,
)


LIFECYCLE_CFG = {
    "cold_start_chain": ["null", "prewarm", "warm"],
    "states": [
        {"name": "null", "category": "stable", "cpu": 0, "memory": 0},
        {"name": "prewarm", "category": "stable", "cpu": 0, "memory": 128},
        {"name": "warm", "category": "stable", "cpu": 0.1, "memory": 256,
         "service_bound": True, "reusable": True},
        {"name": "running", "category": "transient", "cpu": 1.0, "memory": 256,
         "service_bound": True, "reusable": False},
        {"name": "evicted", "category": "stable", "cpu": 0, "memory": 0,
         "reusable": False},
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


def _make_config(strategy="hash_ring", num_nodes=2):
    nodes = [
        {"node_id": f"node-{i}", "cpu_capacity": 8.0, "memory_capacity": 8192}
        for i in range(num_nodes)
    ]
    return {
        "simulation": {"duration": 10.0, "seed": 42},
        "services": [{
            "service_id": "svc-a",
            "arrival_rate": 5.0,
            "job_size": 0.1,
            "max_concurrency": 4,
            "lifecycle": LIFECYCLE_CFG,
        }],
        "cluster": {"nodes": nodes},
        "scheduling": {"strategy": strategy},
    }


def _make_ctx(config):
    env = simpy.Environment()
    rng = np.random.default_rng(42)
    logger = logging.getLogger("test_lb_strategies")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_lb")
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    return ctx


def _dispatch_n(lb, ctx, n, service_id="svc-a"):
    """Dispatch n requests and return list of assigned node_ids."""
    node_ids = []
    for i in range(n):
        inv = Invocation(
            request_id=f"r-{i}", service_id=service_id,
            arrival_time=0.0, job_size=0.1, status="arrived",
        )
        ctx.request_table[inv.request_id] = inv
        lb.dispatch(inv)
        if inv.assigned_node_id:
            node_ids.append(inv.assigned_node_id)
    return node_ids


# ------------------------------------------------------------------
# create_load_balancer factory tests
# ------------------------------------------------------------------

class TestCreateLoadBalancer:
    def test_default_is_hash_ring(self):
        config = _make_config()
        del config["scheduling"]
        ctx = _make_ctx(config)
        lb = create_load_balancer(ctx)
        assert isinstance(lb, HashRingBalancer)

    def test_hash_ring_from_config(self):
        ctx = _make_ctx(_make_config("hash_ring"))
        lb = create_load_balancer(ctx)
        assert isinstance(lb, HashRingBalancer)

    def test_round_robin_from_config(self):
        ctx = _make_ctx(_make_config("round_robin"))
        lb = create_load_balancer(ctx)
        assert isinstance(lb, RoundRobinBalancer)

    def test_least_loaded_from_config(self):
        ctx = _make_ctx(_make_config("least_loaded"))
        lb = create_load_balancer(ctx)
        assert isinstance(lb, LeastLoadedBalancer)

    def test_power_of_two_from_config(self):
        ctx = _make_ctx(_make_config("power_of_two_choices"))
        lb = create_load_balancer(ctx)
        assert isinstance(lb, PowerOfTwoChoicesBalancer)

    def test_unknown_strategy_raises(self):
        ctx = _make_ctx(_make_config("nonexistent"))
        try:
            create_load_balancer(ctx)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)


# ------------------------------------------------------------------
# RoundRobinBalancer tests
# ------------------------------------------------------------------

class TestRoundRobinBalancer:
    def test_dispatches_requests(self):
        ctx = _make_ctx(_make_config("round_robin"))
        lb = RoundRobinBalancer(ctx)
        node_ids = _dispatch_n(lb, ctx, 10)
        assert len(node_ids) == 10

    def test_distributes_evenly(self):
        ctx = _make_ctx(_make_config("round_robin"))
        lb = RoundRobinBalancer(ctx)
        node_ids = _dispatch_n(lb, ctx, 20)
        counts = {}
        for nid in node_ids:
            counts[nid] = counts.get(nid, 0) + 1
        # Should be roughly even (10 each)
        for count in counts.values():
            assert count == 10

    def test_drops_when_all_full(self):
        ctx = _make_ctx(_make_config("round_robin"))
        lb = RoundRobinBalancer(ctx)
        for node in ctx.cluster_manager.get_enabled_nodes():
            node.flavor_memory_used = node.capacity.memory

        inv = Invocation(request_id="r-drop", service_id="svc-a",
                         arrival_time=0.0, job_size=0.1, status="arrived")
        ctx.request_table["r-drop"] = inv
        assert lb.dispatch(inv) is False
        assert inv.dropped is True

    def test_skips_full_node(self):
        ctx = _make_ctx(_make_config("round_robin"))
        lb = RoundRobinBalancer(ctx)
        # Fill node-0 flavor capacity
        node0 = ctx.cluster_manager.get_node("node-0")
        node0.flavor_memory_used = node0.capacity.memory

        node_ids = _dispatch_n(lb, ctx, 5)
        assert all(nid == "node-1" for nid in node_ids)


# ------------------------------------------------------------------
# LeastLoadedBalancer tests
# ------------------------------------------------------------------

class TestLeastLoadedBalancer:
    def test_dispatches_requests(self):
        ctx = _make_ctx(_make_config("least_loaded"))
        lb = LeastLoadedBalancer(ctx)
        node_ids = _dispatch_n(lb, ctx, 10)
        assert len(node_ids) == 10

    def test_prefers_node_with_more_memory(self):
        config = _make_config("least_loaded")
        # node-0: 8192MB, node-1: 4096MB
        config["cluster"]["nodes"][1]["memory_capacity"] = 4096
        ctx = _make_ctx(config)
        lb = LeastLoadedBalancer(ctx)

        node_ids = _dispatch_n(lb, ctx, 1)
        # Should pick node-0 (more available memory)
        assert node_ids[0] == "node-0"

    def test_drops_when_all_full(self):
        ctx = _make_ctx(_make_config("least_loaded"))
        lb = LeastLoadedBalancer(ctx)
        for node in ctx.cluster_manager.get_enabled_nodes():
            node.flavor_memory_used = node.capacity.memory

        inv = Invocation(request_id="r-drop", service_id="svc-a",
                         arrival_time=0.0, job_size=0.1, status="arrived")
        ctx.request_table["r-drop"] = inv
        assert lb.dispatch(inv) is False


# ------------------------------------------------------------------
# PowerOfTwoChoicesBalancer tests
# ------------------------------------------------------------------

class TestPowerOfTwoChoicesBalancer:
    def test_dispatches_requests(self):
        ctx = _make_ctx(_make_config("power_of_two_choices"))
        lb = PowerOfTwoChoicesBalancer(ctx)
        node_ids = _dispatch_n(lb, ctx, 20)
        assert len(node_ids) == 20

    def test_uses_multiple_nodes(self):
        ctx = _make_ctx(_make_config("power_of_two_choices"))
        lb = PowerOfTwoChoicesBalancer(ctx)
        node_ids = _dispatch_n(lb, ctx, 50)
        unique = set(node_ids)
        assert len(unique) == 2  # should use both nodes

    def test_drops_when_all_full(self):
        ctx = _make_ctx(_make_config("power_of_two_choices"))
        lb = PowerOfTwoChoicesBalancer(ctx)
        for node in ctx.cluster_manager.get_enabled_nodes():
            node.flavor_memory_used = node.capacity.memory

        inv = Invocation(request_id="r-drop", service_id="svc-a",
                         arrival_time=0.0, job_size=0.1, status="arrived")
        ctx.request_table["r-drop"] = inv
        assert lb.dispatch(inv) is False

    def test_single_node(self):
        ctx = _make_ctx(_make_config("power_of_two_choices", num_nodes=1))
        lb = PowerOfTwoChoicesBalancer(ctx)
        node_ids = _dispatch_n(lb, ctx, 5)
        assert len(node_ids) == 5
        assert all(nid == "node-0" for nid in node_ids)

    def test_prefers_shorter_queue(self):
        ctx = _make_ctx(_make_config("power_of_two_choices"))
        lb = PowerOfTwoChoicesBalancer(ctx)
        # Put some items in node-0's queue to make it "longer"
        node0 = ctx.cluster_manager.get_node("node-0")
        for i in range(10):
            node0.queue.put(f"dummy-{i}")

        # Most requests should go to node-1 (shorter queue)
        node_ids = _dispatch_n(lb, ctx, 20)
        node1_count = sum(1 for nid in node_ids if nid == "node-1")
        assert node1_count > 10  # majority should go to node-1


# ------------------------------------------------------------------
# Integration: full simulation with each strategy
# ------------------------------------------------------------------

class TestStrategyIntegration:
    def _run_sim(self, strategy):
        from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
        config = _make_config(strategy)
        ctx = _make_ctx(config)
        ctx.lifecycle_manager = LifecycleManager(ctx)
        ctx.dispatcher = create_load_balancer(ctx)
        ctx.cluster_manager.set_context(ctx)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=5.0)
        return ctx

    def test_hash_ring_completes_requests(self):
        ctx = self._run_sim("hash_ring")
        assert ctx.request_table.counters.completed > 0

    def test_round_robin_completes_requests(self):
        ctx = self._run_sim("round_robin")
        assert ctx.request_table.counters.completed > 0

    def test_least_loaded_completes_requests(self):
        ctx = self._run_sim("least_loaded")
        assert ctx.request_table.counters.completed > 0

    def test_power_of_two_completes_requests(self):
        ctx = self._run_sim("power_of_two_choices")
        assert ctx.request_table.counters.completed > 0
