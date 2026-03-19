"""Unit tests for Step 10: Extended lifecycle states + transition resources."""

import logging
import tempfile

import numpy as np
import simpy

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.lifecycle.state_definition import StateDefinition
from serverless_sim.lifecycle.transition_definition import TransitionDefinition
from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.monitoring.monitor_manager import MonitorManager
from serverless_sim.core.simulation.sim_builder import SimulationBuilder
from serverless_sim.core.simulation.sim_engine import SimulationEngine


EXTENDED_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 2},
    "services": [
        {
            "service_id": "svc-ext",
            "arrival_rate": 3.0,
            "job_size": 0.1,
            "memory": 512,
            "cpu": 1.0,
            "max_concurrency": 4,
        }
    ],
    "cluster": {
        "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 16384}]
    },
    "lifecycle": {
        "states": [
            {"name": "null", "category": "stable"},
            {"name": "prewarm", "category": "stable", "steady_memory": 0.0},
            {"name": "code_loaded", "category": "stable", "service_bound": True},
            {"name": "warm", "category": "stable", "service_bound": True, "reusable": True},
            {"name": "running", "category": "transient", "service_bound": True, "reusable": False},
            {"name": "evicted", "category": "stable", "reusable": False},
        ],
        "transitions": [
            {"from": "null", "to": "prewarm", "time": 0.3, "cpu": 0.1},
            {"from": "prewarm", "to": "code_loaded", "time": 0.4, "cpu": 0.2, "memory": 64},
            {"from": "code_loaded", "to": "warm", "time": 0.2, "cpu": 0.1},
            {"from": "warm", "to": "running", "time": 0.0},
            {"from": "running", "to": "warm", "time": 0.0},
            {"from": "warm", "to": "evicted", "time": 0.0},
            {"from": "prewarm", "to": "evicted", "time": 0.0},
            {"from": "code_loaded", "to": "evicted", "time": 0.0},
        ],
    },
    "monitoring": {"interval": 1.0, "max_history_length": 100},
}


# ------------------------------------------------------------------ #
# State machine unit tests
# ------------------------------------------------------------------ #

class TestExtendedStateMachine:
    def test_from_config_creates_all_states(self):
        sm = OpenWhiskExtendedStateMachine.from_config(EXTENDED_CONFIG)
        assert "null" in sm.states
        assert "prewarm" in sm.states
        assert "code_loaded" in sm.states
        assert "warm" in sm.states
        assert "running" in sm.states
        assert "evicted" in sm.states

    def test_from_config_creates_transitions(self):
        sm = OpenWhiskExtendedStateMachine.from_config(EXTENDED_CONFIG)
        assert sm.get_transition("null", "prewarm") is not None
        assert sm.get_transition("prewarm", "code_loaded") is not None
        assert sm.get_transition("code_loaded", "warm") is not None

    def test_path_through_extended_states(self):
        sm = OpenWhiskExtendedStateMachine.from_config(EXTENDED_CONFIG)
        path = sm.find_path("null", "warm")
        assert path == ["null", "prewarm", "code_loaded", "warm"]

    def test_transition_resources(self):
        sm = OpenWhiskExtendedStateMachine.from_config(EXTENDED_CONFIG)
        t = sm.get_transition("prewarm", "code_loaded")
        assert t.transition_time == 0.4
        assert t.transition_cpu == 0.2
        assert t.transition_memory == 64

    def test_state_properties(self):
        sm = OpenWhiskExtendedStateMachine.from_config(EXTENDED_CONFIG)
        code_loaded = sm.get_state("code_loaded")
        assert code_loaded.service_bound is True
        assert code_loaded.category == "stable"
        running = sm.get_state("running")
        assert running.category == "transient"
        assert running.reusable is False

    def test_missing_required_state_raises(self):
        bad_config = {
            "lifecycle": {
                "states": [
                    {"name": "null", "category": "stable"},
                    {"name": "warm", "category": "stable"},
                    # missing running and evicted
                ],
                "transitions": [],
            }
        }
        import pytest
        with pytest.raises(ValueError, match="missing required states"):
            OpenWhiskExtendedStateMachine.from_config(bad_config)

    def test_fallback_to_default(self):
        sm = OpenWhiskExtendedStateMachine.from_config({})
        assert "null" in sm.states
        assert "warm" in sm.states
        assert "code_loaded" not in sm.states  # default has no code_loaded

    def test_default_path(self):
        sm = OpenWhiskExtendedStateMachine.default()
        path = sm.find_path("null", "warm")
        assert path == ["null", "prewarm", "warm"]


# ------------------------------------------------------------------ #
# End-to-end with extended states
# ------------------------------------------------------------------ #

class TestExtendedLifecycleSimulation:
    def test_simulation_with_extended_states(self):
        """Run a full simulation with extended lifecycle and verify instances
        go through the extended path."""
        run_dir = tempfile.mkdtemp(prefix="test_ext_lifecycle_")
        logger = logging.getLogger("test_ext_lifecycle")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(EXTENDED_CONFIG, run_dir, logger)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()
        engine.shutdown()

        # Should have processed some requests
        completed = ctx.request_table.counters.completed
        assert completed > 0

    def test_cold_start_includes_transition_time(self):
        """Cold start should take at least the sum of transition times
        (null->prewarm: 0.3 + prewarm->code_loaded: 0.4 + code_loaded->warm: 0.2 = 0.9s)."""
        run_dir = tempfile.mkdtemp(prefix="test_ext_cold_")
        logger = logging.getLogger("test_ext_cold")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(EXTENDED_CONFIG, run_dir, logger)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()

        # Cold start requests are flushed, but latencies are recorded
        assert ctx.request_table.counters.cold_starts > 0
        # At least one latency should be >= 0.9s (transition time sum)
        assert any(lat >= 0.9 for lat in ctx.request_table.latencies), (
            f"Expected at least one latency >= 0.9s, got {ctx.request_table.latencies}"
        )

    def test_warm_hit_faster_than_cold_start(self):
        """Warm hits should be faster than cold starts.

        Since individual invocations are flushed, we verify indirectly:
        cold starts have transition overhead (>=0.9s), so if there are both
        cold and warm hits, the average latency should be lower than the
        cold start transition time, proving warm hits pull the average down.
        """
        run_dir = tempfile.mkdtemp(prefix="test_ext_warm_")
        logger = logging.getLogger("test_ext_warm")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(EXTENDED_CONFIG, run_dir, logger)

        engine = SimulationEngine(ctx)
        engine.setup()
        engine.run()

        cold_count = ctx.request_table.counters.cold_starts
        completed = ctx.request_table.counters.completed
        warm_count = completed - cold_count

        if cold_count > 0 and warm_count > 0:
            avg_latency = sum(ctx.request_table.latencies) / len(ctx.request_table.latencies)
            # Cold start transition overhead is 0.9s; average should be less
            # because warm hits (~0.1s job_size) pull it down
            assert avg_latency < 0.9, (
                f"Average latency {avg_latency:.3f}s should be < 0.9s cold start overhead"
            )
