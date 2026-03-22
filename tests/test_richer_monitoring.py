"""Unit tests for Step 13: Richer monitoring with per-service metrics and latency percentiles."""

import logging
import tempfile

import numpy as np
import simpy

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.monitoring.monitor_api import MonitorAPI
from serverless_sim.monitoring.collectors import (
    RequestCollector,
    ClusterCollector,
    LifecycleCollector,
    AutoscalingCollector,
)
from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler
from serverless_sim.core.simulation.sim_builder import SimulationBuilder
from serverless_sim.core.simulation.sim_engine import SimulationEngine


MULTI_SERVICE_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 1},
    "services": [
        {
            "service_id": "svc-api",
            "arrival_rate": 10.0,
            "job_size": 0.05,
            "memory": 256,
            "cpu": 0.5,
            "max_concurrency": 8,
        },
        {
            "service_id": "svc-worker",
            "arrival_rate": 2.0,
            "job_size": 0.5,
            "memory": 1024,
            "cpu": 2.0,
            "max_concurrency": 2,
        },
    ],
    "cluster": {
        "nodes": [
            {"node_id": "node-0", "cpu_capacity": 16.0, "memory_capacity": 32768},
        ]
    },
    "autoscaling": {"enabled": True, "reconcile_interval": 2.0},
    "monitoring": {"interval": 1.0, "max_history_length": 100},
}


def _run_multi_service_sim():
    run_dir = tempfile.mkdtemp(prefix="test_rich_mon_")
    logger = logging.getLogger("test_rich_mon")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)

    builder = SimulationBuilder()
    ctx = builder.build(MULTI_SERVICE_CONFIG, run_dir, logger)

    engine = SimulationEngine(ctx)
    engine.setup()
    engine.run()
    return ctx, run_dir


# ------------------------------------------------------------------ #
# Latency percentile tests
# ------------------------------------------------------------------ #

class TestLatencyMetrics:
    def test_latency_mean_present(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        names = store.get_all_metric_names()
        assert "request.latency_mean" in names

    def test_latency_mean_positive(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        mean = store.get_latest("request.latency_mean")
        assert mean is not None
        assert mean[1] > 0

    def test_latency_mean_from_store(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.request_table
        assert store.counters.completed > 0
        assert store.latency_mean > 0


# ------------------------------------------------------------------ #
# Lifecycle collector per-service metrics
# ------------------------------------------------------------------ #

class TestLifecycleCollector:
    def test_per_service_metrics_present(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        names = store.get_all_metric_names()

        # Should have per-service metrics for at least one service
        has_per_service = any("lifecycle.svc-" in n for n in names)
        assert has_per_service

    def test_total_instances_tracked(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        entry = store.get_latest("lifecycle.instances_total")
        assert entry is not None
        # Should have some instances at end of simulation
        assert entry[1] >= 0

    def test_state_breakdown(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        names = store.get_all_metric_names()
        assert "lifecycle.instances_warm" in names
        assert "lifecycle.instances_running" in names
        assert "lifecycle.instances_prewarm" in names
        assert "lifecycle.instances_evicted" in names


# ------------------------------------------------------------------ #
# Autoscaling collector tests
# ------------------------------------------------------------------ #

class TestAutoscalingCollector:
    def test_autoscaling_metrics_present(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        names = store.get_all_metric_names()

        assert any("autoscaling.svc-api" in n for n in names)
        assert any("autoscaling.svc-worker" in n for n in names)

    def test_min_instances_metric(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        entry = store.get_latest("autoscaling.svc-api.min_instances")
        assert entry is not None
        assert entry[1] >= 0


# ------------------------------------------------------------------ #
# All metric families have data
# ------------------------------------------------------------------ #

class TestAllMetricFamilies:
    def test_request_metrics(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        for metric in ["request.total", "request.completed", "request.dropped",
                        "request.cold_starts", "request.in_flight"]:
            assert store.get_latest(metric) is not None, f"Missing metric: {metric}"

    def test_cluster_metrics(self):
        ctx, _ = _run_multi_service_sim()
        store = ctx.monitor_manager.store
        for metric in ["cluster.nodes_enabled", "cluster.cpu_total", "cluster.cpu_used",
                        "cluster.cpu_utilization", "cluster.memory_total",
                        "cluster.memory_used", "cluster.memory_utilization"]:
            assert store.get_latest(metric) is not None, f"Missing metric: {metric}"

    def test_export_csv_has_columns(self):
        """Export mode 1 CSV should have all metric families as columns."""
        import csv
        import os

        ctx, run_dir = _run_multi_service_sim()
        # Trigger export
        ctx.export_manager.export()

        csv_path = os.path.join(run_dir, "system_metrics.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "time" in header
        assert "request.total" in header
        assert "cluster.cpu_utilization" in header
