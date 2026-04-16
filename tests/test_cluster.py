"""Unit tests for resource model, Node, ClusterManager."""

import simpy
import logging

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.cluster.node import Node
from serverless_sim.cluster.compute_class import ComputeClass
from serverless_sim.cluster.cluster_manager import ClusterManager


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
    "simulation": {"duration": 10.0, "seed": 42},
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
}


# ------------------------------------------------------------------ #
# ResourceProfile tests
# ------------------------------------------------------------------ #

class TestResourceProfile:
    def test_add(self):
        a = ResourceProfile(cpu=2.0, memory=1024)
        b = ResourceProfile(cpu=1.0, memory=512)
        c = a.add(b)
        assert c.cpu == 3.0
        assert c.memory == 1536

    def test_subtract(self):
        a = ResourceProfile(cpu=4.0, memory=2048)
        b = ResourceProfile(cpu=1.5, memory=512)
        c = a.subtract(b)
        assert c.cpu == 2.5
        assert c.memory == 1536

    def test_can_fit_true(self):
        available = ResourceProfile(cpu=4.0, memory=2048)
        request = ResourceProfile(cpu=2.0, memory=1024)
        assert available.can_fit(request) is True

    def test_can_fit_false_cpu(self):
        available = ResourceProfile(cpu=1.0, memory=2048)
        request = ResourceProfile(cpu=2.0, memory=1024)
        assert available.can_fit(request) is False

    def test_can_fit_false_memory(self):
        available = ResourceProfile(cpu=4.0, memory=512)
        request = ResourceProfile(cpu=2.0, memory=1024)
        assert available.can_fit(request) is False


# ------------------------------------------------------------------ #
# Node tests
# ------------------------------------------------------------------ #

class TestNode:
    def _make_node(self, env):
        return Node(
            env=env,
            node_id="test-node",
            capacity=ResourceProfile(cpu=4.0, memory=4096),
            compute_class=ComputeClass(class_id="default"),
            logger=logging.getLogger("test"),
        )

    def test_allocate_and_release(self):
        env = simpy.Environment()
        node = self._make_node(env)
        req = ResourceProfile(cpu=1.0, memory=512)

        assert node.allocate(req) is True
        assert node.available.cpu == 3.0
        assert node.available.memory == 3584
        assert node.allocated.cpu == 1.0

        node.release(req)
        assert node.available.cpu == 4.0
        assert node.available.memory == 4096

    def test_allocate_exceeds_capacity(self):
        env = simpy.Environment()
        node = self._make_node(env)
        big_req = ResourceProfile(cpu=5.0, memory=256)
        assert node.allocate(big_req) is False
        # State should be unchanged
        assert node.available.cpu == 4.0

    def test_queue_put_and_pull(self):
        """Put a request into queue, verify the pull loop picks it up."""
        env = simpy.Environment()
        node = self._make_node(env)
        node.start_pull_loop()

        from serverless_sim.workload.invocation import Invocation
        request = Invocation(request_id="req-1", service_id="svc-a")
        env.process(self._put_request(env, node, request))
        env.run(until=1.0)

    @staticmethod
    def _put_request(env, node, request):
        yield node.queue.put(request)


# ------------------------------------------------------------------ #
# ClusterManager tests
# ------------------------------------------------------------------ #

class TestClusterManager:
    def test_creates_nodes_from_config(self):
        env = simpy.Environment()
        cm = ClusterManager(env=env, config=SAMPLE_CONFIG)

        assert len(cm.nodes) == 2
        assert "node-0" in cm.nodes
        assert "node-1" in cm.nodes

    def test_node_capacities(self):
        env = simpy.Environment()
        cm = ClusterManager(env=env, config=SAMPLE_CONFIG)

        n0 = cm.get_node("node-0")
        assert n0.capacity.cpu == 8.0
        assert n0.capacity.memory == 8192

        n1 = cm.get_node("node-1")
        assert n1.capacity.cpu == 4.0
        assert n1.capacity.memory == 4096

    def test_get_enabled_nodes(self):
        env = simpy.Environment()
        cm = ClusterManager(env=env, config=SAMPLE_CONFIG)

        enabled = cm.get_enabled_nodes()
        assert len(enabled) == 2

        # Disable one
        cm.get_node("node-1").enabled = False
        enabled = cm.get_enabled_nodes()
        assert len(enabled) == 1
        assert enabled[0].node_id == "node-0"

    def test_get_node_raises_on_missing(self):
        env = simpy.Environment()
        cm = ClusterManager(env=env, config=SAMPLE_CONFIG)
        try:
            cm.get_node("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_start_all_and_put_request(self):
        """Start all nodes and push a request into one."""
        env = simpy.Environment()
        cm = ClusterManager(env=env, config=SAMPLE_CONFIG)
        cm.start_all()

        from serverless_sim.workload.invocation import Invocation
        request = Invocation(request_id="req-1", service_id="svc-a")
        node = cm.get_node("node-0")
        env.process(self._put(env, node, request))
        env.run(until=1.0)

    @staticmethod
    def _put(env, node, request):
        yield node.queue.put(request)
