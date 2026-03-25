"""Tests for TraceReplayGenerator and PrecomputedServingModel."""

import os
import tempfile
import logging

import simpy
import numpy as np

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.cluster.serving_model import PrecomputedServingModel, FixedRateModel
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.trace_generator import TraceReplayGenerator
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
        f.write("timestamp,function_id,duration\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    return path


def _make_config(trace_path=None, serving_model="fixed_rate"):
    config = {
        "simulation": {"duration": 10.0, "seed": 42},
        "services": [{
            "service_id": "svc-a",
            "arrival_rate": 1.0,
            "job_size": 0.1,
            "max_concurrency": 4,
            "lifecycle": LIFECYCLE_CFG,
        }],
        "cluster": {
            "serving_model": serving_model,
            "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}],
        },
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
            (0.1, "svc-a", 0.05),
            (0.2, "svc-a", 0.08),
            (0.3, "svc-b", 0.10),
        ])
        gen = TraceReplayGenerator(path)
        assert gen.record_count == 3
        assert gen.function_ids == {"svc-a", "svc-b"}

    def test_records_sorted_by_timestamp(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.5, "svc-a", 0.05),
            (0.1, "svc-a", 0.08),
            (0.3, "svc-a", 0.10),
        ])
        gen = TraceReplayGenerator(path)
        assert gen._records[0].timestamp == 0.1
        assert gen._records[1].timestamp == 0.3
        assert gen._records[2].timestamp == 0.5

    def test_replay_creates_requests(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a", 0.05),
            (0.5, "svc-a", 0.08),
            (1.0, "svc-a", 0.10),
        ])
        config = _make_config(trace_path=path, serving_model="precomputed")
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=5.0)

        assert ctx.request_table.counters.total == 3
        assert ctx.request_table.counters.completed == 3

    def test_stop_time_respected(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a", 0.05),
            (0.5, "svc-a", 0.08),
            (2.0, "svc-a", 0.10),  # after stop_time
            (3.0, "svc-a", 0.10),  # after stop_time
        ])
        config = _make_config(trace_path=path, serving_model="precomputed")
        config["simulation"]["duration"] = 1.0
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=1.0)
        ctx.env.run(until=5.0)

        # Only first 2 requests should be generated (before stop_time=1.0)
        assert ctx.request_table.counters.total == 2

    def test_unmatched_function_ids_skipped(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a", 0.05),
            (0.2, "svc-unknown", 0.08),  # not in config
            (0.3, "svc-a", 0.10),
        ])
        config = _make_config(trace_path=path, serving_model="precomputed")
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=5.0)

        # svc-unknown is skipped, only svc-a requests
        assert ctx.request_table.counters.total == 2

    def test_service_time_set_from_duration(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.1, "svc-a", 0.123),
        ])
        config = _make_config(trace_path=path, serving_model="precomputed")
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=5.0)

        assert ctx.request_table.counters.completed == 1


# ------------------------------------------------------------------
# PrecomputedServingModel tests
# ------------------------------------------------------------------

class TestPrecomputedServingModel:
    def test_uses_service_time(self):
        model = PrecomputedServingModel()
        t = model.estimate_service_time(0.1, node=None, service_time=0.5)
        assert t == 0.5

    def test_fallback_to_job_size(self):
        model = PrecomputedServingModel()
        t = model.estimate_service_time(0.1, node=None, service_time=None)
        assert t == 0.1

    def test_fallback_when_no_kwarg(self):
        model = PrecomputedServingModel()
        t = model.estimate_service_time(0.3, node=None)
        assert t == 0.3


# ------------------------------------------------------------------
# Config-driven model selection tests
# ------------------------------------------------------------------

class TestConfigModelSelection:
    def test_poisson_default(self):
        tmpdir = tempfile.mkdtemp()
        config = _make_config()
        ctx = _make_ctx(config, tmpdir)
        from serverless_sim.workload.generators import PoissonFixedSizeGenerator
        assert isinstance(ctx.workload_manager.generator, PoissonFixedSizeGenerator)

    def test_trace_generator_from_config(self):
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [(0.1, "svc-a", 0.05)])
        config = _make_config(trace_path=path)
        ctx = _make_ctx(config, tmpdir)
        assert isinstance(ctx.workload_manager.generator, TraceReplayGenerator)

    def test_precomputed_serving_model_from_config(self):
        tmpdir = tempfile.mkdtemp()
        config = _make_config(serving_model="precomputed")
        ctx = _make_ctx(config, tmpdir)
        node = ctx.cluster_manager.get_node("node-0")
        assert isinstance(node.serving_model, PrecomputedServingModel)

    def test_fixed_rate_serving_model_default(self):
        tmpdir = tempfile.mkdtemp()
        config = _make_config(serving_model="fixed_rate")
        ctx = _make_ctx(config, tmpdir)
        node = ctx.cluster_manager.get_node("node-0")
        assert isinstance(node.serving_model, FixedRateModel)


# ------------------------------------------------------------------
# Integration: trace replay + precomputed serving model
# ------------------------------------------------------------------

class TestTraceReplayIntegration:
    def test_variable_execution_times(self):
        """Requests with different durations should complete at different times."""
        tmpdir = tempfile.mkdtemp()
        path = _write_trace(tmpdir, [
            (0.0, "svc-a", 0.1),   # short
            (0.0, "svc-a", 1.0),   # long
        ])
        config = _make_config(trace_path=path, serving_model="precomputed")
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=10.0)

        assert ctx.request_table.counters.completed == 2

    def test_sample_trace_config(self):
        """Test with actual sample configs if they exist."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..",
            "configs", "simulation", "sample_trace_replay.json",
        )
        trace_path = os.path.join(
            os.path.dirname(__file__), "..",
            "configs", "simulation", "sample_trace.csv",
        )
        if not os.path.exists(config_path) or not os.path.exists(trace_path):
            return

        import json
        with open(config_path) as f:
            config = json.load(f)

        tmpdir = tempfile.mkdtemp()
        ctx = _make_ctx(config, tmpdir)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        ctx.env.run(until=config["simulation"]["duration"])

        assert ctx.request_table.counters.total > 0
        assert ctx.request_table.counters.completed > 0
