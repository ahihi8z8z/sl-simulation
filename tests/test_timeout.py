"""Unit tests for Step 6: Timeout handling."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager


def _make_ctx(config, seed=42):
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    logger = logging.getLogger("test_timeout")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_run")
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    return ctx


class TestTimeout:
    def test_timeout_when_service_time_exceeds(self):
        """With timeout=0.5 and job_size=2.0 (service_time=2.0), requests should timeout."""
        config = {
            "simulation": {"duration": 5.0, "seed": 42, "export_mode": 0},
            "services": [
                {
                    "service_id": "svc-slow",
                    "arrival_rate": 2.0,
                    "job_size": 2.0,
                    "timeout": 0.5,
                    "memory": 256,
                    "cpu": 1.0,
                    "max_concurrency": 1,
                }
            ],
            "cluster": {
                "nodes": [
                    {"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192},
                ]
            },
        }
        ctx = _make_ctx(config)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=5.0)

        timed_out = [inv for inv in ctx.request_table.values() if inv.timed_out]
        assert len(timed_out) > 0, "Expected some requests to timeout"
        for inv in timed_out:
            assert inv.drop_reason == "timeout"

    def test_no_timeout_when_fast(self):
        """With timeout=10 and job_size=0.01, no requests should timeout."""
        config = {
            "simulation": {"duration": 5.0, "seed": 42, "export_mode": 0},
            "services": [
                {
                    "service_id": "svc-fast",
                    "arrival_rate": 5.0,
                    "job_size": 0.01,
                    "timeout": 10.0,
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
        ctx = _make_ctx(config)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=5.0)

        timed_out = [inv for inv in ctx.request_table.values() if inv.timed_out]
        assert len(timed_out) == 0, f"Expected no timeouts, got {len(timed_out)}"
        completed = [inv for inv in ctx.request_table.values() if inv.status == "completed"]
        assert len(completed) > 0

    def test_resources_released_on_timeout(self):
        """After timeout, resources should be released."""
        config = {
            "simulation": {"duration": 3.0, "seed": 42, "export_mode": 0},
            "services": [
                {
                    "service_id": "svc-slow",
                    "arrival_rate": 1.0,
                    "job_size": 5.0,
                    "timeout": 0.5,
                    "memory": 256,
                    "cpu": 1.0,
                    "max_concurrency": 1,
                }
            ],
            "cluster": {
                "nodes": [
                    {"node_id": "node-0", "cpu_capacity": 4.0, "memory_capacity": 4096},
                ]
            },
        }
        ctx = _make_ctx(config)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()

        ctx.env.run(until=10.0)

        # After everything settles, check no active requests remain
        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        active = sum(i.active_requests for i in instances if i.state != "evicted")
        assert active == 0
