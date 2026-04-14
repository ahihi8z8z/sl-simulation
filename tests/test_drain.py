"""Unit tests for drain period and truncated request handling."""

import json
import logging
import os
import tempfile

import numpy as np
import simpy

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.core.simulation.sim_builder import SimulationBuilder
from serverless_sim.core.simulation.sim_engine import SimulationEngine
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.monitoring.collectors import RequestCollector


# Config where requests take longer than the simulation duration
# so some will be in-flight at the end.
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

SLOW_SERVICE_CONFIG = {
    "simulation": {"duration": 3.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-slow",
            "arrival_rate": 5.0,
            # 2s service time configured via service_time provider
            "max_concurrency": 1,
            "lifecycle": LIFECYCLE_256_1,
        }
    ],
    "cluster": {
        "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}]
    },
    "monitoring": {"interval": 1.0, "max_history_length": 100},
}


def _make_ctx(config, seed=42):
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    logger = logging.getLogger("test_drain")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    run_dir = tempfile.mkdtemp(prefix="test_drain_")
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir=run_dir)
    from serverless_sim.workload.service_time import FixedServiceTime
    ctx.service_time_provider = FixedServiceTime(duration=2.0)
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    ctx.monitor_manager = MonitorManager(ctx, interval=1.0, max_history=100)
    return ctx


# ------------------------------------------------------------------ #
# Generator stop_time
# ------------------------------------------------------------------ #

class TestGeneratorStopTime:
    def test_generator_stops_at_duration(self):
        """No requests should arrive after stop_time."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        stop_time = 3.0
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=stop_time)

        # Run well past stop_time
        ctx.env.run(until=10.0)

        # Completed requests are flushed; check remaining in-flight ones
        for inv in ctx.request_table.values():
            assert inv.arrival_time <= stop_time, (
                f"Request {inv.request_id} arrived at {inv.arrival_time} after stop_time={stop_time}"
            )
        # With rate=5 and stop_time=3, expect ~15 total requests
        total = len(ctx.request_table)
        assert total > 0
        assert total <= 30, f"Too many requests ({total}) for stop_time={stop_time}"

    def test_generator_without_stop_time_continues(self):
        """Without stop_time, requests arrive throughout the run."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()  # no stop_time

        ctx.env.run(until=10.0)

        # Without stop_time, total requests should be much higher than
        # with stop_time=3 (rate=5 over 10s => ~50 vs ~15)
        total = len(ctx.request_table)
        assert total > 30, f"Expected many requests without stop_time, got {total}"
        # In-flight requests (with 2s service time) should include late arrivals
        if ctx.request_table.active_count > 0:
            max_arrival = max(inv.arrival_time for inv in ctx.request_table.values())
            assert max_arrival > 3.0, "Requests should arrive beyond t=3 without stop_time"


# ------------------------------------------------------------------ #
# Drain period
# ------------------------------------------------------------------ #

class TestDrainPeriod:
    def test_drain_completes_in_flight_requests(self):
        """With drain period, in-flight requests should complete instead of being truncated."""
        config = {**SLOW_SERVICE_CONFIG}
        config["simulation"] = {**config["simulation"], "drain_timeout": 30.0}

        run_dir = tempfile.mkdtemp(prefix="test_drain_")
        logger = logging.getLogger("test_drain_complete")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(config, run_dir, logger)
        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        completed = ctx.request_table.counters.completed
        truncated = ctx.request_table.counters.truncated

        # With generous drain, most requests should complete
        assert completed > 0
        # Some may still be truncated (requests arriving just before stop_time)
        # but far fewer than without drain
        total = len(ctx.request_table)
        assert completed / total > 0.5, f"Expected most to complete, got {completed}/{total}"

    def test_zero_drain_causes_truncation(self):
        """With drain_timeout=0, in-flight requests should be marked truncated."""
        config = {**SLOW_SERVICE_CONFIG}
        config["simulation"] = {**config["simulation"], "drain_timeout": 0}

        run_dir = tempfile.mkdtemp(prefix="test_drain_zero_")
        logger = logging.getLogger("test_drain_zero")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(config, run_dir, logger)
        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        truncated = ctx.request_table.counters.truncated
        # With slow service (2s) and high rate (5/s), many should be in-flight at t=3
        assert truncated > 0, "Expected truncated requests with drain_timeout=0"

    def test_default_drain_uses_max_timeout(self):
        """Default drain_timeout should equal max service timeout."""
        engine = SimulationEngine(_make_ctx(SLOW_SERVICE_CONFIG))
        drain = engine._get_drain_timeout()
        assert drain == 30.0  # max timeout from svc-slow

    def test_explicit_drain_timeout_from_config(self):
        """drain_timeout in config overrides the default."""
        config = {**SLOW_SERVICE_CONFIG}
        config["simulation"] = {**config["simulation"], "drain_timeout": 5.0}
        engine = SimulationEngine(_make_ctx(config))
        drain = engine._get_drain_timeout()
        assert drain == 5.0


# ------------------------------------------------------------------ #
# Truncated sweep
# ------------------------------------------------------------------ #

class TestTruncatedSweep:
    def test_mark_truncated_sets_status(self):
        """_mark_truncated should set status='truncated' on non-terminal requests."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=3.0)
        ctx.env.run(until=3.0)  # No drain — stop immediately

        engine = SimulationEngine(ctx)
        engine._mark_truncated()

        # After _mark_truncated, all requests should be finalized (no in-flight remaining)
        assert ctx.request_table.active_count == 0
        # All requests should be accounted for via counters
        c = ctx.request_table.counters
        assert c.completed + c.truncated == c.total - c.dropped

    def test_truncated_has_completion_time(self):
        """Truncated requests should be counted after _mark_truncated."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=3.0)
        ctx.env.run(until=3.0)

        engine = SimulationEngine(ctx)
        engine._mark_truncated()

        # Truncated requests are finalized/flushed, verify via counter
        assert ctx.request_table.counters.truncated > 0

    def test_truncated_does_not_affect_completed(self):
        """Completed requests should not be changed by truncation sweep."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=3.0)
        ctx.env.run(until=3.0)

        # Count terminal states before sweep (already finalized via counters)
        completed_before = ctx.request_table.counters.completed

        engine = SimulationEngine(ctx)
        engine._mark_truncated()

        # Completed count should not change after truncation sweep
        assert ctx.request_table.counters.completed == completed_before


# ------------------------------------------------------------------ #
# RequestCollector includes truncated
# ------------------------------------------------------------------ #

class TestRequestCollectorTruncated:
    def test_collector_reports_truncated(self):
        """RequestCollector should include request.truncated metric."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=3.0)
        ctx.env.run(until=3.0)

        engine = SimulationEngine(ctx)
        engine._mark_truncated()

        collector = RequestCollector()
        metrics = collector.collect(ctx.env.now, ctx)

        assert "request.truncated" in metrics
        assert metrics["request.truncated"] == ctx.request_table.counters.truncated


# ------------------------------------------------------------------ #
# Summary includes truncated
# ------------------------------------------------------------------ #

class TestSummaryTruncated:
    def test_summary_shows_truncated(self):
        """Summary.txt should include Truncated count."""
        config = {**SLOW_SERVICE_CONFIG}
        config["simulation"] = {**config["simulation"], "drain_timeout": 0, "export_mode": 0}

        run_dir = tempfile.mkdtemp(prefix="test_summary_trunc_")
        logger = logging.getLogger("test_summary_trunc")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(config, run_dir, logger)
        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        summary_path = os.path.join(run_dir, "summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["requests"]["truncated"] > 0
