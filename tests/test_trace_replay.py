"""Tests for TraceReplayGenerator and service_time provider integration."""

import os
import tempfile
import logging

import simpy
import numpy as np

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.trace_generator import TraceReplayGenerator
from serverless_sim.workload.service_time import FixedServiceTime
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer


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


def _write_trace(tmpdir, rows):
    """Write a trace CSV and return path."""
    path = os.path.join(tmpdir, "trace.csv")
    with open(path, "w") as f:
        f.write("timestamp,function_id\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    return path


def _make_config(trace_path=None):
    config = {
        "simulation": {"duration": 10.0, "seed": 42},
        "services": [{
            "service_id": "svc-a",
            "arrival_rate": 1.0,
            "max_concurrency": 4,
            "lifecycle": LIFECYCLE_CFG,
        }],
        "cluster": {
            "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}],
        },
        "service_time": {"mode": "fixed", "duration": 0.1},
    }
    if trace_path:
        config["workload"] = {
            "generator": "trace",
            "trace_path": trace_path,
        }
    return config


def _make_ctx(config, tmpdir):
    env = simpy.Environment()
    rng = np.random.default_rng(42)
    logger = logging.getLogger("test_trace")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir=tmpdir)
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)

    # Service time provider
    from serverless_sim.workload.service_time import create_service_time_provider
    ctx.service_time_provider = create_service_time_provider(config)

    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    return ctx


# ------------------------------------------------------------------
# TraceReplayGenerator unit tests
# ------------------------------------------------------------------

class TestTraceReplayGenerator:
    def test_load_records(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a"),
            (0.2, "svc-a"),
            (0.3, "svc-b"),
        ])
        gen = TraceReplayGenerator(path)
        assert gen.record_count == 3
        assert gen.function_ids == {"svc-a", "svc-b"}

    def test_records_sorted_by_timestamp(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.5, "svc-a"),
            (0.1, "svc-a"),
            (0.3, "svc-a"),
        ])
        gen = TraceReplayGenerator(path)
        assert gen._records[0].timestamp == 0.1
        assert gen._records[1].timestamp == 0.3
        assert gen._records[2].timestamp == 0.5

    def test_replay_creates_requests(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a"),
            (0.5, "svc-a"),
            (1.0, "svc-a"),
        ])
        config = _make_config(trace_path=path)
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=5.0)

        assert ctx.request_table.counters.total == 3
        assert ctx.request_table.counters.completed == 3

    def test_stop_time_respected(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a"),
            (0.5, "svc-a"),
            (2.0, "svc-a"),
            (3.0, "svc-a"),
        ])
        config = _make_config(trace_path=path)
        config["simulation"]["duration"] = 1.0
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=1.0)
        ctx.env.run(until=5.0)

        assert ctx.request_table.counters.total == 2

    def test_unmatched_function_ids_skipped(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a"),
            (0.2, "svc-unknown"),
            (0.3, "svc-a"),
        ])
        config = _make_config(trace_path=path)
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=5.0)

        assert ctx.request_table.counters.total == 2

    def test_service_time_assigned_by_provider(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [(0.1, "svc-a")])
        config = _make_config(trace_path=path)
        config["service_time"] = {"mode": "fixed", "duration": 0.5}
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=5.0)

        assert ctx.request_table.counters.completed == 1


# ------------------------------------------------------------------
# ServiceTimeProvider tests
# ------------------------------------------------------------------

class TestServiceTimeProvider:
    def test_fixed_service_time(self):
        from serverless_sim.workload.invocation import Invocation
        provider = FixedServiceTime(duration=0.42)
        inv = Invocation(request_id="r1", service_id="svc-a")
        rng = np.random.default_rng(42)
        provider.assign(inv, rng)
        assert inv.service_time == 0.42

    def test_sample_csv_service_time(self):
        from serverless_sim.workload.service_time import SampleCsvServiceTime
        from serverless_sim.workload.invocation import Invocation

        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "durations.csv")
        with open(csv_path, "w") as f:
            f.write("duration\n0.1\n0.2\n0.3\n")

        provider = SampleCsvServiceTime(csv_path)
        inv = Invocation(request_id="r1", service_id="svc-a")
        rng = np.random.default_rng(42)
        provider.assign(inv, rng)
        assert inv.service_time in (0.1, 0.2, 0.3)


# ------------------------------------------------------------------
# Config-driven selection tests
# ------------------------------------------------------------------

class TestConfigSelection:
    def test_poisson_default(self):
        tmpdir = tempfile.mkdtemp()
        config = _make_config()
        ctx = _make_ctx(config, tmpdir)
        from serverless_sim.workload.generators import PoissonFixedSizeGenerator
        assert isinstance(ctx.workload_manager.generator, PoissonFixedSizeGenerator)

    def test_trace_generator_from_config(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [(0.1, "svc-a")])
        config = _make_config(trace_path=path)
        ctx = _make_ctx(config, tmpdir)
        assert isinstance(ctx.workload_manager.generator, TraceReplayGenerator)


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------

class TestTraceReplayIntegration:
    def test_fixed_service_time_execution(self):
        """All requests should complete with fixed service time."""
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.0, "svc-a"),
            (0.0, "svc-a"),
        ])
        config = _make_config(trace_path=path)
        config["service_time"] = {"mode": "fixed", "duration": 0.5}
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=10.0)

        assert ctx.request_table.counters.completed == 2
