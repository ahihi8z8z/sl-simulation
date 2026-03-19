"""Unit tests for drain period and truncated request handling."""

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
SLOW_SERVICE_CONFIG = {
    "simulation": {"duration": 3.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-slow",
            "arrival_rate": 5.0,
            "job_size": 2.0,       # 2s service time — many will be in-flight at t=3
            "timeout": 30.0,
            "memory": 256,
            "cpu": 1.0,
            "max_concurrency": 1,
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

        for inv in ctx.request_table.values():
            assert inv.arrival_time <= stop_time, (
                f"Request {inv.request_id} arrived at {inv.arrival_time} after stop_time={stop_time}"
            )

    def test_generator_without_stop_time_continues(self):
        """Without stop_time, requests arrive throughout the run."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()  # no stop_time

        ctx.env.run(until=10.0)

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

        completed = sum(1 for inv in ctx.request_table.values() if inv.status == "completed")
        truncated = sum(1 for inv in ctx.request_table.values() if inv.status == "truncated")

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

        truncated = sum(1 for inv in ctx.request_table.values() if inv.status == "truncated")
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

        for inv in ctx.request_table.values():
            assert inv.status in ("completed", "timed_out", "truncated"), (
                f"Request {inv.request_id} has unexpected status: {inv.status}"
            )

    def test_truncated_has_completion_time(self):
        """Truncated requests should have a completion_time set."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=3.0)
        ctx.env.run(until=3.0)

        engine = SimulationEngine(ctx)
        engine._mark_truncated()

        for inv in ctx.request_table.values():
            if inv.status == "truncated":
                assert inv.completion_time is not None
                assert inv.drop_reason == "simulation_end"

    def test_truncated_does_not_affect_completed(self):
        """Completed and timed_out requests should not be changed by truncation sweep."""
        ctx = _make_ctx(SLOW_SERVICE_CONFIG)
        ctx.cluster_manager.start_all()
        ctx.workload_manager.start(stop_time=3.0)
        ctx.env.run(until=3.0)

        # Count terminal states before sweep
        completed_before = sum(1 for inv in ctx.request_table.values() if inv.status == "completed")
        timed_out_before = sum(1 for inv in ctx.request_table.values() if inv.timed_out)

        engine = SimulationEngine(ctx)
        engine._mark_truncated()

        completed_after = sum(1 for inv in ctx.request_table.values() if inv.status == "completed")
        timed_out_after = sum(1 for inv in ctx.request_table.values() if inv.timed_out)

        assert completed_after == completed_before
        assert timed_out_after == timed_out_before


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
        truncated_count = sum(1 for inv in ctx.request_table.values() if inv.status == "truncated")
        assert metrics["request.truncated"] == truncated_count


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

        summary_path = os.path.join(run_dir, "summary.txt")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            content = f.read()
        assert "Truncated:" in content
