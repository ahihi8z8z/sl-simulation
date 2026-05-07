"""Unit tests for Step 7: Basic monitoring."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.monitoring.metric_store import MetricStore
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.monitoring.monitor_api import MonitorAPI
from serverless_sim.workload.service_time import FixedServiceTime


def _make_ctx(seed=42):
    config = {
        "simulation": {"duration": 10.0, "seed": seed, "export_mode": 0},
        "services": [
            {
                "service_id": "svc-a",
                "lifecycle": {
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
                },
                "workload": {"arrival_rate": 5.0},
            }
        ],
        "cluster": {
            "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}]
        },
        "monitoring": {"interval": 1.0, "max_history_length": 100},
    }
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    logger = logging.getLogger("test_monitor")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_run")
    _provider = FixedServiceTime(duration=0.1)

    for _svc in ctx.config.get("services", []):

        ctx.service_time_providers[_svc["service_id"]] = _provider
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)

    mm = MonitorManager(ctx, interval=1.0, max_history=100)
    ctx.monitor_manager = mm
    return ctx


class TestMetricStore:
    def test_put_and_get_latest(self):
        store = MetricStore(max_history_length=10)
        store.put("foo", 1.0, 42)
        store.put("foo", 2.0, 99)
        assert store.get_latest("foo") == (2.0, 99)

    def test_query_range(self):
        store = MetricStore(max_history_length=10)
        for i in range(5):
            store.put("bar", float(i), i * 10)
        result = store.query_range("bar", 1.0, 3.0)
        assert len(result) == 3
        assert result[0] == (1.0, 10)

    def test_ring_buffer_limit(self):
        store = MetricStore(max_history_length=5)
        for i in range(10):
            store.put("x", float(i), i)
        entries = store.get_all_entries("x")
        assert len(entries) == 5
        assert entries[0] == (5.0, 5)

    def test_missing_metric(self):
        store = MetricStore()
        assert store.get_latest("nope") is None
        assert store.query_range("nope", 0, 10) == []


class TestMonitorManager:
    def test_periodic_collection(self):
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.monitor_manager.start()

        ctx.env.run(until=10.0)

        store = ctx.monitor_manager.store
        names = store.get_all_metric_names()
        assert "request.total" in names
        assert "cluster.cpu_utilization" in names
        assert "lifecycle.instances_total" in names

    def test_get_latest_has_data(self):
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.monitor_manager.start()

        ctx.env.run(until=10.0)

        store = ctx.monitor_manager.store
        entry = store.get_latest("request.completed")
        assert entry is not None
        assert entry[1] > 0  # Should have some completed requests


class TestMonitorAPI:
    def test_snapshot(self):
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.monitor_manager.start()

        ctx.env.run(until=10.0)

        api = MonitorAPI(ctx.monitor_manager)
        snap = api.get_snapshot()
        assert "request.total" in snap
        assert snap["request.total"] > 0

    def test_get_latest_value(self):
        ctx = _make_ctx()
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.monitor_manager.start()

        ctx.env.run(until=10.0)

        api = MonitorAPI(ctx.monitor_manager)
        val = api.get_latest_value("request.completed", default=0)
        assert val > 0
        missing = api.get_latest_value("nonexistent", default=-1)
        assert missing == -1
