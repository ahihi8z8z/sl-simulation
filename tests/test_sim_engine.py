"""Unit tests for Step 9: SimulationEngine + SimulationBuilder + CLI end-to-end."""

import logging
import os
import tempfile

from serverless_sim.core.simulation.sim_builder import SimulationBuilder
from serverless_sim.core.simulation.sim_engine import SimulationEngine


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

CONFIG = {
    "simulation": {"duration": 5.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "job_size": 0.1,
            "max_concurrency": 4,
            "lifecycle": LIFECYCLE_256_1,
        }
    ],
    "cluster": {
        "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}]
    },
    "monitoring": {"interval": 1.0, "max_history_length": 100},
    "workload": {"arrival_rate": 5.0},
}

CONFIG_WITH_AUTOSCALING = {
    **CONFIG,
    "autoscaling": {"enabled": True, "reconcile_interval": 2.0},
}

CONFIG_WITH_CONTROLLER = {
    **CONFIG_WITH_AUTOSCALING,
    "controller": {"enabled": True, "interval": 2.0, "cpu_high": 0.8, "cpu_low": 0.3, "prewarm_max": 5},
}


def _make_logger():
    logger = logging.getLogger("test_engine")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    return logger


# ------------------------------------------------------------------ #
# SimulationBuilder tests
# ------------------------------------------------------------------ #

class TestSimulationBuilder:
    def test_build_minimal(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG, run_dir, _make_logger())

        assert ctx.cluster_manager is not None
        assert ctx.workload_manager is not None
        assert ctx.lifecycle_manager is not None
        assert ctx.dispatcher is not None
        assert ctx.monitor_manager is not None
        assert ctx.export_manager is not None

    def test_build_with_autoscaling(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG_WITH_AUTOSCALING, run_dir, _make_logger())
        assert ctx.autoscaling_manager is not None

    def test_build_with_controller(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG_WITH_CONTROLLER, run_dir, _make_logger())
        assert ctx.controller is not None

    def test_export_mode_override(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG, run_dir, _make_logger(), export_mode_override=2)
        assert ctx.export_manager.mode == 2


# ------------------------------------------------------------------ #
# SimulationEngine tests
# ------------------------------------------------------------------ #

class TestSimulationEngine:
    def test_full_run(self):
        """Run setup → run → shutdown without errors."""
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG, run_dir, _make_logger())

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        # Verify simulation ran past duration (includes drain period)
        assert ctx.env.now >= 5.0
        assert len(ctx.request_table) > 0

    def test_full_run_produces_summary(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG, run_dir, _make_logger())

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        assert os.path.exists(os.path.join(run_dir, "summary.json"))

    def test_full_run_with_export_mode_2(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG, run_dir, _make_logger(), export_mode_override=2)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        assert os.path.exists(os.path.join(run_dir, "summary.json"))
        assert os.path.exists(os.path.join(run_dir, "system_metrics.csv"))
        assert os.path.exists(os.path.join(run_dir, "request_trace.csv"))

    def test_get_snapshot(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG, run_dir, _make_logger())

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run(until=3.0)

        snap = engine.get_snapshot()
        assert "request.total" in snap
        assert snap["request.total"] > 0

    def test_run_with_autoscaling_and_controller(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG_WITH_CONTROLLER, run_dir, _make_logger())

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        assert ctx.env.now >= 5.0
        assert len(ctx.request_table) > 0

    def test_completed_requests_exist(self):
        run_dir = tempfile.mkdtemp(prefix="test_engine_")
        builder = SimulationBuilder()
        ctx = builder.build(CONFIG, run_dir, _make_logger())

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()

        completed = ctx.request_table.counters.completed
        assert completed > 0
