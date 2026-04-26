"""Unit tests for queue depth limit (max_queue_depth on nodes)."""

import logging
import tempfile

import numpy as np
import simpy

from serverless_sim.core.simulation.sim_context import SimContext
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

QUEUE_LIMIT_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-flood",
            "job_size": 1.0,
            "lifecycle": LIFECYCLE_256_1,
            "workload": {"arrival_rate": 100.0},
        }
    ],
    "cluster": {
        "nodes": [
            {
                "node_id": "node-0",
                "cpu_capacity": 8.0,
                "memory_capacity": 2048,
                "max_queue_depth": 5,
            },
        ]
    },
    "monitoring": {"interval": 1.0, "max_history_length": 100},
}


class TestQueueDepthLimit:
    def test_requests_dropped_when_queue_full(self):
        """With max_queue_depth=5 and high arrival rate, some requests
        should be dropped with reason 'no_capacity'."""
        run_dir = tempfile.mkdtemp(prefix="test_queue_depth_")
        logger = logging.getLogger("test_queue_depth")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(QUEUE_LIMIT_CONFIG, run_dir, logger)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        dropped = ctx.request_table.counters.dropped
        assert dropped > 0, "Expected some requests to be dropped due to queue depth limit"

    def test_drop_reason_is_no_capacity(self):
        """Dropped requests should have drop_reason containing 'no_capacity'."""
        run_dir = tempfile.mkdtemp(prefix="test_queue_depth_reason_")
        logger = logging.getLogger("test_queue_depth_reason")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        # Use export_mode=2 so we can inspect the trace
        config = {**QUEUE_LIMIT_CONFIG}
        config["simulation"] = {**config["simulation"], "export_mode": 2}
        ctx = builder.build(config, run_dir, logger)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        # Verify drops occurred
        dropped = ctx.request_table.counters.dropped
        assert dropped > 0, "Expected drops from queue depth limit"

        # Check the trace CSV for no_capacity reason
        import csv
        import os

        trace_path = os.path.join(run_dir, "request_trace.csv")
        assert os.path.exists(trace_path)
        with open(trace_path) as f:
            reader = csv.DictReader(f)
            drop_reasons = [
                row["drop_reason"] for row in reader if row.get("dropped") == "True"
            ]
        assert any("no_capacity" in r for r in drop_reasons), (
            f"Expected 'no_capacity' in drop reasons, got {drop_reasons}"
        )

    def test_completed_requests_still_work(self):
        """Even with queue depth limit, some requests should complete successfully."""
        run_dir = tempfile.mkdtemp(prefix="test_queue_depth_ok_")
        logger = logging.getLogger("test_queue_depth_ok")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(QUEUE_LIMIT_CONFIG, run_dir, logger)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        completed = ctx.request_table.counters.completed
        assert completed > 0, "Expected some requests to complete despite queue depth limit"

