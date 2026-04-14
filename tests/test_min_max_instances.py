"""Tests for min_instances and max_instances enforcement."""

import logging

import numpy as np
import simpy

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.invocation import Invocation
from serverless_sim.workload.service_time import FixedServiceTime
from serverless_sim.scheduling.load_balancer import ShardingContainerPoolBalancer
from serverless_sim.lifecycle.lifecycle_manager import LifecycleManager
from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler
from serverless_sim.monitoring.monitor_manager import MonitorManager


def _make_lifecycle(memory=256, cpu=1.0):
    return {
        "cold_start_chain": ["null", "prewarm", "warm"],
        "states": [
            {"name": "null", "category": "stable", "cpu": 0, "memory": 0},
            {"name": "prewarm", "category": "stable", "cpu": 0, "memory": memory // 2},
            {"name": "warm", "category": "stable", "cpu": 0.1, "memory": memory, "service_bound": True, "reusable": True},
            {"name": "running", "category": "transient", "cpu": cpu, "memory": memory, "service_bound": True, "reusable": False},
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


def _make_config(
    min_instances=0,
    max_instances=0,
    arrival_rate=0.0,
    memory=256,
    cpu=1.0,
    max_concurrency=4,
    node_memory=8192,
    pool_targets=None,
):
    return {
        "simulation": {"duration": 100.0, "seed": 42, "export_mode": 0},
        "services": [
            {
                "service_id": "svc-a",
                "arrival_rate": arrival_rate,
                "max_concurrency": max_concurrency,
                "min_instances": min_instances,
                "max_instances": max_instances,
                "lifecycle": _make_lifecycle(memory=memory, cpu=cpu),
            }
        ],
        "cluster": {
            "nodes": [
                {"node_id": "node-0", "cpu_capacity": 16.0, "memory_capacity": node_memory},
            ]
        },
    }


def _make_ctx(config, seed=42):
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    logger = logging.getLogger("test_min_max")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir="/tmp/test_run")
    ctx.service_time_provider = FixedServiceTime(duration=0.1)
    ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
    ctx.workload_manager = WorkloadManager.from_config(ctx)
    ctx.lifecycle_manager = LifecycleManager(ctx)
    ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
    ctx.cluster_manager.set_context(ctx)
    ctx.monitor_manager = MonitorManager(ctx)
    return ctx


def _setup_autoscaler(ctx, reconcile_interval=100.0, pool_targets=None):
    autoscaler = OpenWhiskPoolAutoscaler(ctx, reconcile_interval=reconcile_interval)
    ctx.autoscaling_manager = autoscaler
    if pool_targets:
        # Set pool targets directly to avoid triggering fill before start()
        autoscaler._pool_targets.setdefault("svc-a", {}).update(pool_targets)
    return autoscaler


# ------------------------------------------------------------------ #
# TestMinInstances
# ------------------------------------------------------------------ #

class TestMinInstances:
    def test_min_instances_fill_on_startup(self):
        """min_instances=3 -> 3 warm containers created at startup."""
        config = _make_config(min_instances=3)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Let startup fill complete (cold start ~0.5s with default state machine)
        ctx.env.run(until=2.0)

        warm = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        assert len(warm) >= 3, f"Expected >= 3 warm instances, got {len(warm)}"

    def test_min_instances_no_refill_when_warm_to_running(self):
        """min_instances=2, request takes warm -> running, NO new warm created.

        min_instances counts alive (warm + running) like provisioned concurrency.
        warm -> running doesn't reduce alive count, so no refill needed.
        """
        config = _make_config(min_instances=2, arrival_rate=0.0)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Let 2 warm instances be created
        ctx.env.run(until=2.0)

        warm_before = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        assert len(warm_before) >= 2

        # Simulate a request consuming a warm instance
        inst = warm_before[0]
        inv = Invocation(request_id="r1", service_id="svc-a", arrival_time=ctx.env.now)
        ctx.lifecycle_manager.start_execution(inst, inv)

        # 1 warm + 1 running = 2 alive >= min_instances=2 → no refill
        alive = autoscaler._count_alive_instances("svc-a")
        assert alive >= 2, f"Expected alive >= 2, got {alive}"

        # Let some time pass
        ctx.env.run(until=5.0)

        # Total instances should still be 2 (no extra created)
        total = autoscaler._count_total_instances("svc-a")
        assert total == 2, f"Expected 2 total instances (no refill), got {total}"

    def test_min_instances_prevents_idle_eviction(self):
        """min_instances=2, 2 warm idle, reconcile runs -> NOT evicted."""
        config = _make_config(min_instances=2)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx, reconcile_interval=1.0)
        # Set very short idle timeout
        autoscaler.set_idle_timeout("svc-a", 0.1)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Let 2 warm instances be created
        ctx.env.run(until=2.0)

        warm = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        assert len(warm) >= 2

        # Run reconcile (idle timeout is 0.1s, so instances are eligible)
        ctx.env.run(until=5.0)

        # Warm count should NOT drop below min_instances
        warm_after = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        assert len(warm_after) >= 2, (
            f"Expected >= 2 warm (min_instances), got {len(warm_after)}"
        )


# ------------------------------------------------------------------ #
# TestMaxInstances
# ------------------------------------------------------------------ #

class TestMaxInstances:
    def test_max_instances_blocks_cold_start(self):
        """max_instances=2, 2 instances exist (each fully occupied), new request -> drop 'max_instances'."""
        config = _make_config(max_instances=2, min_instances=2, arrival_rate=0.0, max_concurrency=1)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Let 2 warm instances be created (min_instances=2)
        ctx.env.run(until=2.0)

        warm = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        assert len(warm) >= 2

        # Now send a request that would need a new instance
        # First use the 2 warm instances with long jobs to keep them occupied
        node = ctx.cluster_manager.get_node("node-0")
        for inst in warm:
            inv = Invocation(
                request_id=f"use-{inst.instance_id}",
                service_id="svc-a",
                arrival_time=ctx.env.now,
            )
            inv.service_time = 100.0  # long job: instance stays running
            ctx.lifecycle_manager.start_execution(inst, inv)

        # Now all instances are running (fully occupied, max_concurrency=1), send another request
        inv_drop = Invocation(
            request_id="r-drop",
            service_id="svc-a",
            arrival_time=ctx.env.now,
        )
        inv_drop.status = "arrived"
        ctx.request_table.register(inv_drop)
        node.queue.put(inv_drop)

        ctx.env.run(until=5.0)

        # The request should be dropped with reason "max_instances"
        assert inv_drop.dropped is True
        assert inv_drop.drop_reason == "max_instances"
        assert inv_drop.status == "dropped"

    def test_max_instances_zero_is_unlimited(self):
        """max_instances=0, create many instances -> no drop."""
        config = _make_config(max_instances=0, arrival_rate=20.0)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        autoscaler.start()

        ctx.env.run(until=5.0)

        # No drops due to max_instances
        # (drops could still happen for other reasons like capacity)
        dropped = [
            inv for inv in ctx.request_table.values()
            if inv.drop_reason == "max_instances"
        ]
        assert len(dropped) == 0

    def test_max_instances_counts_all_states(self):
        """max_instances=3 with 3 instances (all running, fully occupied) -> next request dropped."""
        config = _make_config(max_instances=3, min_instances=3, arrival_rate=0.0, max_concurrency=1)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Let 3 warm instances be created (min_instances=3)
        ctx.env.run(until=2.0)

        total = autoscaler._count_total_instances("svc-a")
        assert total >= 3, f"Expected >= 3 total instances, got {total}"

        # Make all instances running with long jobs (max_concurrency=1, so fully occupied)
        node = ctx.cluster_manager.get_node("node-0")
        warm = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        for inst in warm:
            inv = Invocation(
                request_id=f"use-{inst.instance_id}",
                service_id="svc-a",
                arrival_time=ctx.env.now,
            )
            inv.service_time = 100.0  # long job: instance stays running
            ctx.lifecycle_manager.start_execution(inst, inv)

        # Send a new request - all 3 running (fully occupied), would need cold start
        # but total=3 >= max=3 so should be dropped
        inv_drop = Invocation(
            request_id="r-over-max",
            service_id="svc-a",
            arrival_time=ctx.env.now,
        )
        inv_drop.status = "arrived"
        ctx.request_table.register(inv_drop)
        node.queue.put(inv_drop)

        ctx.env.run(until=5.0)

        assert inv_drop.dropped is True
        assert inv_drop.drop_reason == "max_instances"

    def test_max_instances_allows_when_below(self):
        """max_instances=5, 1 instance -> cold start ok."""
        config = _make_config(max_instances=5, min_instances=1, arrival_rate=0.0)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        # Let 1 warm instance be created
        ctx.env.run(until=2.0)

        total = autoscaler._count_total_instances("svc-a")
        assert total >= 1

        # Use the warm instance
        node = ctx.cluster_manager.get_node("node-0")
        warm = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        for inst in warm:
            inv = Invocation(
                request_id=f"use-{inst.instance_id}",
                service_id="svc-a",
                arrival_time=ctx.env.now,
            )
            inv.service_time = 0.1
            ctx.lifecycle_manager.start_execution(inst, inv)

        # Send a request - should not be dropped since total < max_instances
        inv_ok = Invocation(
            request_id="r-ok",
            service_id="svc-a",
            arrival_time=ctx.env.now,
        )
        inv_ok.service_time = 0.1
        inv_ok.status = "arrived"
        ctx.request_table.register(inv_ok)
        node.queue.put(inv_ok)

        ctx.env.run(until=5.0)

        assert inv_ok.dropped is False
        assert inv_ok.drop_reason != "max_instances"

    def test_max_instances_drop_reason(self):
        """Dropped request has drop_reason='max_instances' (instance fully occupied)."""
        config = _make_config(max_instances=1, min_instances=1, arrival_rate=0.0, max_concurrency=1)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)

        ctx.cluster_manager.start_all()
        autoscaler.start()

        ctx.env.run(until=2.0)

        # Use the warm instance
        node = ctx.cluster_manager.get_node("node-0")
        warm = [
            i for i in ctx.lifecycle_manager.get_instances_for_node("node-0")
            if i.state == "warm"
        ]
        assert len(warm) >= 1
        inst = warm[0]
        inv = Invocation(
            request_id="use-1",
            service_id="svc-a",
            arrival_time=ctx.env.now,
        )
        inv.service_time = 100.0  # long job: instance stays running (max_concurrency=1, fully occupied)
        ctx.lifecycle_manager.start_execution(inst, inv)

        # Send another request
        inv_drop = Invocation(
            request_id="r-drop-reason",
            service_id="svc-a",
            arrival_time=ctx.env.now,
        )
        inv_drop.status = "arrived"
        ctx.request_table.register(inv_drop)
        node.queue.put(inv_drop)

        ctx.env.run(until=5.0)

        assert inv_drop.drop_reason == "max_instances"


# ------------------------------------------------------------------ #
# TestBudgetPriority
# ------------------------------------------------------------------ #

class TestBudgetPriority:
    def test_warm_filled_before_pool_targets(self):
        """max_instances=4, min_instances=2, pool_targets={'prewarm':3}
        -> 2 warm + 2 prewarm (not 3 prewarm, budget limits it)."""
        config = _make_config(max_instances=4, min_instances=2)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx, pool_targets={"prewarm": 3})

        ctx.cluster_manager.start_all()
        autoscaler.start()

        ctx.env.run(until=3.0)

        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        warm = [i for i in instances if i.state == "warm"]
        prewarm = [i for i in instances if i.state == "prewarm"]
        total = len(instances)

        # Total should not exceed max_instances=4
        assert total <= 4, f"Total {total} exceeds max_instances=4"
        # Warm should be filled first (min_instances=2)
        assert len(warm) >= 2, f"Expected >= 2 warm (min_instances), got {len(warm)}"

    def test_pool_targets_respect_max(self):
        """max_instances=3, pool_targets={'prewarm':5}
        -> only 3 created total (2 warm if min=2, 1 prewarm)."""
        config = _make_config(max_instances=3, min_instances=2)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx, pool_targets={"prewarm": 5})

        ctx.cluster_manager.start_all()
        autoscaler.start()

        ctx.env.run(until=3.0)

        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        total = len(instances)

        assert total <= 3, f"Total {total} exceeds max_instances=3"

    def test_no_pool_targets_only_min(self):
        """No pool_targets set, min_instances=2 -> 2 warm created, no intermediate."""
        config = _make_config(min_instances=2)
        ctx = _make_ctx(config)
        autoscaler = _setup_autoscaler(ctx)  # no pool_targets

        ctx.cluster_manager.start_all()
        autoscaler.start()

        ctx.env.run(until=3.0)

        instances = ctx.lifecycle_manager.get_instances_for_node("node-0")
        warm = [i for i in instances if i.state == "warm"]
        prewarm = [i for i in instances if i.state == "prewarm"]

        assert len(warm) >= 2, f"Expected >= 2 warm (min_instances), got {len(warm)}"
        assert len(prewarm) == 0, f"Expected 0 prewarm (no pool_targets), got {len(prewarm)}"


# ------------------------------------------------------------------ #
# TestValidation
# ------------------------------------------------------------------ #

class TestValidation:
    def test_min_greater_than_max_raises(self):
        """min_instances=5, max_instances=3 -> ValueError from config loader."""
        import pytest
        import json
        import tempfile
        import os
        from serverless_sim.core.config.loader import load_config

        config = _make_config(min_instances=5, max_instances=3)
        # Write to temp file for load_config
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(config, f)
            with pytest.raises(ValueError, match="min_instances.*must be <= max_instances"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_defaults(self):
        """No min/max in config -> min=0, max=0 (unlimited)."""
        from serverless_sim.workload.service_class import ServiceClass

        cfg = {
            "service_id": "svc-test",
            "arrival_rate": 1.0,
            "max_concurrency": 1,
            "lifecycle": _make_lifecycle(memory=128, cpu=0.5),
        }
        svc = ServiceClass.from_config(cfg)
        assert svc.min_instances == 0
        assert svc.max_instances == 0
