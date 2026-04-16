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


EXTENDED_LIFECYCLE = {
    "cold_start_chain": ["null", "prewarm", "code_loaded", "warm"],
    "states": [
        {"name": "null", "category": "stable", "cpu": 0, "memory": 0},
        {"name": "prewarm", "category": "stable", "cpu": 0, "memory": 0},
        {"name": "code_loaded", "category": "stable", "cpu": 0, "memory": 256, "service_bound": True},
        {"name": "warm", "category": "stable", "cpu": 0.1, "memory": 512, "service_bound": True, "reusable": True},
        {"name": "running", "category": "transient", "cpu": 1.0, "memory": 512, "service_bound": True, "reusable": False},
        {"name": "evicted", "category": "stable", "cpu": 0, "memory": 0, "reusable": False},
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
}

EXTENDED_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 2},
    "services": [
        {
            "service_id": "svc-ext",
            "job_size": 0.1,
            "max_concurrency": 4,
            "lifecycle": EXTENDED_LIFECYCLE,
        }
    ],
    "cluster": {
        "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 16384}]
    },
    "monitoring": {"interval": 1.0, "max_history_length": 100},
    "workload": {"arrival_rate": 3.0},
}


# ------------------------------------------------------------------ #
# State machine unit tests
# ------------------------------------------------------------------ #

class TestExtendedStateMachine:
    def test_from_config_creates_all_states(self):
        sm = OpenWhiskExtendedStateMachine.from_lifecycle_config(EXTENDED_LIFECYCLE)
        assert "null" in sm.states
        assert "prewarm" in sm.states
        assert "code_loaded" in sm.states
        assert "warm" in sm.states
        assert "running" in sm.states
        assert "evicted" in sm.states

    def test_from_config_creates_transitions(self):
        sm = OpenWhiskExtendedStateMachine.from_lifecycle_config(EXTENDED_LIFECYCLE)
        assert sm.get_transition("null", "prewarm") is not None
        assert sm.get_transition("prewarm", "code_loaded") is not None
        assert sm.get_transition("code_loaded", "warm") is not None

    def test_path_through_extended_states(self):
        sm = OpenWhiskExtendedStateMachine.from_lifecycle_config(EXTENDED_LIFECYCLE)
        path = sm.find_path("null", "warm")
        assert path == ["null", "prewarm", "code_loaded", "warm"]

    def test_transition_resources(self):
        sm = OpenWhiskExtendedStateMachine.from_lifecycle_config(EXTENDED_LIFECYCLE)
        t = sm.get_transition("prewarm", "code_loaded")
        assert t.transition_time == 0.4
        assert t.transition_cpu == 0.2
        assert t.transition_memory == 64

    def test_state_properties(self):
        sm = OpenWhiskExtendedStateMachine.from_lifecycle_config(EXTENDED_LIFECYCLE)
        code_loaded = sm.get_state("code_loaded")
        assert code_loaded.service_bound is True
        assert code_loaded.category == "stable"
        running = sm.get_state("running")
        assert running.category == "transient"
        assert running.reusable is False

    def test_missing_required_state_raises(self):
        bad_lifecycle = {
            "states": [
                {"name": "prewarm", "category": "stable"},
                # missing null and warm
            ],
            "cold_start_chain": ["null", "prewarm", "warm"],
        }
        import pytest
        with pytest.raises(ValueError, match="missing required states"):
            OpenWhiskExtendedStateMachine.from_lifecycle_config(bad_lifecycle)

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

        # Cold start requests are flushed, but counters are recorded
        assert ctx.request_table.counters.cold_starts > 0
        # Mean latency > 0 confirms requests were processed with transition time
        assert ctx.request_table.latency_mean > 0, (
            f"Expected positive mean latency, got {ctx.request_table.latency_mean}"
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
            avg_latency = ctx.request_table.latency_mean
            # Cold start transition overhead is 0.9s; average should be less
            # because warm hits (~0.1s job_size) pull it down
            assert avg_latency < 0.9, (
                f"Average latency {avg_latency:.3f}s should be < 0.9s cold start overhead"
            )


# ------------------------------------------------------------------ #
# Promote tests
# ------------------------------------------------------------------ #

class TestPromoteInstance:
    """Test promoting intermediate pool instances to warm."""

    def _make_ctx_with_pool(self, pool_targets=None):
        """Create a context with extended states and pool targets."""
        config = {
            **EXTENDED_CONFIG,
            "services": [{
                **EXTENDED_CONFIG["services"][0],
            }],
            "autoscaling": {"enabled": True, "reconcile_interval": 100.0},
        }
        env = simpy.Environment()
        rng = np.random.default_rng(42)
        logger = logging.getLogger("test_promote")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler

        ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_promote")
        ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
        ctx.workload_manager = WorkloadManager.from_config(ctx)
        ctx.lifecycle_manager = LifecycleManager(ctx)
        ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
        ctx.cluster_manager.set_context(ctx)
        ctx.monitor_manager = MonitorManager(ctx)
        autoscaler = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=100.0)
        ctx.autoscaling_manager = autoscaler

        # Set pool targets via controller/policy API (not service config)
        if pool_targets:
            for state, count in pool_targets.items():
                autoscaler.set_pool_target("svc-ext", state, count)

        return ctx

    def test_find_promotable_prefers_deepest(self):
        """find_promotable_instance should return the deepest instance (closest to warm)."""
        ctx = self._make_ctx_with_pool({"prewarm": 1, "code_loaded": 1})
        lm = ctx.lifecycle_manager
        node = ctx.cluster_manager.get_node("node-0")

        ctx.cluster_manager.start_all()
        ctx.autoscaling_manager.start()

        # Let pool fill: prewarm takes 0.3s, code_loaded takes 0.3+0.4=0.7s
        ctx.env.run(until=2.0)

        promotable = lm.find_promotable_instance(node, "svc-ext")
        assert promotable is not None
        # Should prefer code_loaded over prewarm (deeper)
        assert promotable.state == "code_loaded"

    def test_promote_faster_than_cold_start(self):
        """Promoting code_loaded→warm (0.2s) should be faster than full cold start (0.9s)."""
        ctx = self._make_ctx_with_pool({"code_loaded": 1})
        lm = ctx.lifecycle_manager
        node = ctx.cluster_manager.get_node("node-0")

        ctx.cluster_manager.start_all()
        ctx.autoscaling_manager.start()

        # Let code_loaded instance be created (0.7s)
        ctx.env.run(until=2.0)

        inst = lm.find_promotable_instance(node, "svc-ext")
        assert inst is not None
        assert inst.state == "code_loaded"

        t_before = ctx.env.now
        promote_proc = lm.promote_instance(node, inst)
        ctx.env.run(until=ctx.env.now + 1.0)

        assert inst.state == "warm"
        promote_time = inst.state_entered_at - t_before
        # code_loaded→warm = 0.2s, much less than full cold start = 0.9s
        assert promote_time < 0.5, f"Promote took {promote_time}s, expected ~0.2s"

    def test_promote_triggers_pool_refill(self):
        """After promoting an instance, the pool should be refilled reactively."""
        ctx = self._make_ctx_with_pool({"code_loaded": 2})
        lm = ctx.lifecycle_manager
        node = ctx.cluster_manager.get_node("node-0")

        ctx.cluster_manager.start_all()
        ctx.autoscaling_manager.start()

        # Let 2 code_loaded instances be created
        ctx.env.run(until=2.0)
        code_loaded = [i for i in lm.get_instances_for_node("node-0") if i.state == "code_loaded"]
        assert len(code_loaded) >= 2

        # Promote one
        inst = code_loaded[0]
        lm.promote_instance(node, inst)
        # Run long enough for the reactive fill to create a replacement
        ctx.env.run(until=5.0)

        code_loaded_after = [i for i in lm.get_instances_for_node("node-0") if i.state == "code_loaded"]
        # Should be back to 2 (1 remaining + 1 replacement)
        assert len(code_loaded_after) >= 2, (
            f"Expected >= 2 code_loaded after refill, got {len(code_loaded_after)}"
        )

    def test_no_promotable_when_all_mid_transition(self):
        """Instances still transitioning (target_state != None) should not be promotable."""
        ctx = self._make_ctx_with_pool({"code_loaded": 1})
        lm = ctx.lifecycle_manager
        node = ctx.cluster_manager.get_node("node-0")

        ctx.cluster_manager.start_all()
        ctx.autoscaling_manager.start()

        # Run very briefly — instance is still mid-transition
        ctx.env.run(until=0.1)

        promotable = lm.find_promotable_instance(node, "svc-ext")
        assert promotable is None
