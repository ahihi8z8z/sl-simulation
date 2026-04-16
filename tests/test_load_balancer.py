"""Unit tests for Step 4: Load balancer."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.invocation import Invocation
from serverless_sim.workload.service_time import FixedServiceTime
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer


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
            "max_concurrency": 2,
            "lifecycle": LIFECYCLE_256_1,
        }
    ],
    "cluster": {
        "nodes": [
            {"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192},
            {"node_id": "node-1", "cpu_capacity": 4.0, "memory_capacity": 4096},
        ]
    },
    "workload": {"arrival_rate": 10.0},
}


def _make_ctx(config=None) -> SimContext:
    config = config or SAMPLE_CONFIG
    env = simpy.Environment()
    rng = np.random.default_rng(42)
    logger = logging.getLogger("test_lb")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_run")
    ctx.service_time_provider = FixedServiceTime(duration=0.1)
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    return ctx


class TestShardingBalancer:
    def test_dispatch_to_node(self):
        ctx = _make_ctx()
        lb = ShardingContainerPoolBalancer(ctx)
        inv = Invocation(request_id="r1", service_id="svc-a", arrival_time=0.0,
                         status="arrived")
        ctx.request_table["r1"] = inv

        ok = lb.dispatch(inv)
        assert ok is True
        assert inv.assigned_node_id is not None
        assert inv.status == "dispatched"
        assert inv.dispatch_time == 0.0

    def test_affinity_same_service(self):
        """Same service_id should hash to the same primary node."""
        ctx = _make_ctx()
        lb = ShardingContainerPoolBalancer(ctx)

        node_ids = set()
        for i in range(10):
            inv = Invocation(request_id=f"r{i}", service_id="svc-a",
                             arrival_time=0.0, status="arrived")
            ctx.request_table[inv.request_id] = inv
            lb.dispatch(inv)
            node_ids.add(inv.assigned_node_id)

        # All should go to the same node (same service_id hash)
        assert len(node_ids) == 1

    def test_fallback_when_node_full(self):
        """When primary node has no memory, fallback to next node."""
        ctx = _make_ctx()
        lb = ShardingContainerPoolBalancer(ctx)

        # Figure out primary node for svc-a
        inv0 = Invocation(request_id="probe", service_id="svc-a",
                          arrival_time=0.0, status="arrived")
        ctx.request_table["probe"] = inv0
        lb.dispatch(inv0)
        primary_node_id = inv0.assigned_node_id

        # Exhaust primary node's flavor capacity
        primary_node = ctx.cluster_manager.get_node(primary_node_id)
        primary_node.flavor_memory_used = primary_node.capacity.memory

        # Next dispatch should fallback
        inv1 = Invocation(request_id="r-fallback", service_id="svc-a",
                          arrival_time=0.0, status="arrived")
        ctx.request_table["r-fallback"] = inv1
        ok = lb.dispatch(inv1)
        assert ok is True
        assert inv1.assigned_node_id != primary_node_id

    def test_drop_when_all_full(self):
        """Drop when all nodes have no flavor capacity."""
        ctx = _make_ctx()
        lb = ShardingContainerPoolBalancer(ctx)

        for node in ctx.cluster_manager.get_enabled_nodes():
            node.flavor_memory_used = node.capacity.memory

        inv = Invocation(request_id="r-drop", service_id="svc-a",
                         arrival_time=0.0, status="arrived")
        ctx.request_table["r-drop"] = inv
        ok = lb.dispatch(inv)
        assert ok is False
        assert inv.dropped is True
        assert inv.drop_reason == "no_capacity"
        assert inv.status == "dropped"

    def test_end_to_end_with_generator(self):
        """Workload → dispatcher → node queues."""
        ctx = _make_ctx()
        ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=5.0)

        dispatched = [inv for inv in ctx.request_table.values() if inv.status == "dispatched"]
        assert len(dispatched) > 0
        # All dispatched requests should have a node assigned
        for inv in dispatched:
            assert inv.assigned_node_id is not None
