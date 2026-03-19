"""Unit tests for Step 8: Export (3 modes)."""

import csv
import os
import tempfile

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.export.export_manager import ExportManager


CONFIG = {
    "simulation": {"duration": 5.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "arrival_rate": 5.0,
            "job_size": 0.1,
            "timeout": 10.0,
            "memory": 256,
            "cpu": 1.0,
            "max_concurrency": 4,
        }
    ],
    "cluster": {
        "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}]
    },
    "monitoring": {"interval": 1.0, "max_history_length": 100},
}


def _run_sim(export_mode: int):
    """Run a full simulation and export with given mode. Returns (ctx, run_dir)."""
    run_dir = tempfile.mkdtemp(prefix="test_export_")
    env = simpy.Environment()
    rng = np.random.default_rng(42)
    logger = logging.getLogger("test_export")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)

    ctx = SimContext(env=env, config=CONFIG, rng=rng, logger=logger, run_dir=run_dir)
    ctx.cluster_manager = ClusterManager(env=env, config=CONFIG, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    ctx.monitor_manager = MonitorManager(ctx, interval=1.0, max_history=100)

    # ExportManager must be created BEFORE running the simulation so that
    # mode 2 streaming trace captures all requests as they finalize.
    em = ExportManager(ctx, mode=export_mode)
    ctx.export_manager = em

    ctx.cluster_manager.start_all()
    ctx.workload_manager.start()
    ctx.monitor_manager.start()

    ctx.env.run(until=5.0)

    paths = em.export()
    return ctx, run_dir, paths


class TestExportMode0:
    def test_only_summary(self):
        ctx, run_dir, paths = _run_sim(0)
        assert len(paths) == 1
        assert os.path.exists(os.path.join(run_dir, "summary.txt"))
        assert not os.path.exists(os.path.join(run_dir, "system_metrics.csv"))
        assert not os.path.exists(os.path.join(run_dir, "request_trace.csv"))

    def test_summary_content(self):
        ctx, run_dir, _ = _run_sim(0)
        with open(os.path.join(run_dir, "summary.txt")) as f:
            content = f.read()
        assert "Simulation Summary" in content
        assert "Total requests:" in content
        assert "Completed:" in content


class TestExportMode1:
    def test_summary_and_metrics(self):
        ctx, run_dir, paths = _run_sim(1)
        assert len(paths) == 2
        assert os.path.exists(os.path.join(run_dir, "summary.txt"))
        assert os.path.exists(os.path.join(run_dir, "system_metrics.csv"))
        assert not os.path.exists(os.path.join(run_dir, "request_trace.csv"))

    def test_metrics_csv_has_header_and_data(self):
        ctx, run_dir, _ = _run_sim(1)
        path = os.path.join(run_dir, "system_metrics.csv")
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert "time" in header
        assert len(rows) > 0


class TestExportMode2:
    def test_all_files(self):
        ctx, run_dir, paths = _run_sim(2)
        assert len(paths) == 3
        assert os.path.exists(os.path.join(run_dir, "summary.txt"))
        assert os.path.exists(os.path.join(run_dir, "system_metrics.csv"))
        assert os.path.exists(os.path.join(run_dir, "request_trace.csv"))

    def test_request_trace_rows(self):
        ctx, run_dir, _ = _run_sim(2)
        path = os.path.join(run_dir, "request_trace.csv")
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert "request_id" in header
        assert "arrival_time" in header
        # Number of rows should match request table
        assert len(rows) == len(ctx.request_table)
