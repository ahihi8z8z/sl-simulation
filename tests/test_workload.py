"""Unit tests for Step 3: Workload generation."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.workload.service_class import ServiceClass
from serverless_sim.workload.invocation import Invocation
from serverless_sim.workload.generators import PoissonFixedSizeGenerator
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.service_time import FixedServiceTime


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

SAMPLE_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "max_concurrency": 2,
            "lifecycle": LIFECYCLE_256_1,
            "workload": {"arrival_rate": 10.0},
        }
    ],
    "cluster": {
        "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}]
    },
}


def _make_ctx(duration: float = 10.0, seed: int = 42) -> SimContext:
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    logger = logging.getLogger("test_workload")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ctx = SimContext(
        env=env,
        config=SAMPLE_CONFIG,
        rng=rng,
        logger=logger,
        run_dir="/tmp/test_run",
    )
    ctx.service_time_provider = FixedServiceTime(duration=0.1)
    return ctx


# ------------------------------------------------------------------ #
# ServiceClass tests
# ------------------------------------------------------------------ #

class TestServiceClass:
    def test_from_config(self):
        cfg = SAMPLE_CONFIG["services"][0]
        svc = ServiceClass.from_config(cfg)
        assert svc.service_id == "svc-a"
        assert svc.peak_memory == 256
        assert svc.peak_cpu == 1.0
        assert svc.max_concurrency == 2

    def test_defaults(self):
        cfg = {
            "service_id": "svc-b",
            "max_concurrency": 1,
            "lifecycle": {
                "cold_start_chain": ["null", "prewarm", "warm"],
                "states": [
                    {"name": "null", "category": "stable", "cpu": 0, "memory": 0},
                    {"name": "prewarm", "category": "stable", "cpu": 0, "memory": 64},
                    {"name": "warm", "category": "stable", "cpu": 0.1, "memory": 128, "service_bound": True, "reusable": True},
                    {"name": "running", "category": "transient", "cpu": 0.5, "memory": 128, "service_bound": True, "reusable": False},
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
            },
        }
        svc = ServiceClass.from_config(cfg)
        assert svc.min_instances == 0
        assert svc.max_instances == 0


# ------------------------------------------------------------------ #
# Invocation tests
# ------------------------------------------------------------------ #

class TestInvocation:
    def test_defaults(self):
        inv = Invocation(request_id="r1", service_id="svc-a", arrival_time=1.0)
        assert inv.status == "created"
        assert inv.dropped is False
        assert inv.cold_start is False
        assert inv.dispatch_time is None


# ------------------------------------------------------------------ #
# PoissonFixedSizeGenerator tests
# ------------------------------------------------------------------ #

class TestPoissonGenerator:
    def test_generates_requests(self):
        """Run for 10s at rate=10 → expect ~100 requests (Poisson)."""
        ctx = _make_ctx(duration=10.0, seed=42)
        svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])

        gen = PoissonFixedSizeGenerator(arrival_rate=10.0)
        gen.attach(ctx)
        gen.start_for_service(svc)

        ctx.env.run(until=10.0)

        n = len(ctx.request_table)
        # With rate=10 and duration=10, expect ~100; allow wide margin
        assert 50 < n < 200, f"Expected ~100 requests, got {n}"

    def test_arrival_times_valid(self):
        ctx = _make_ctx(duration=5.0, seed=123)
        svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])

        gen = PoissonFixedSizeGenerator(arrival_rate=10.0)
        gen.attach(ctx)
        gen.start_for_service(svc)

        ctx.env.run(until=5.0)

        for inv in ctx.request_table.values():
            assert 0.0 < inv.arrival_time <= 5.0
            assert inv.service_id == "svc-a"
            assert inv.status == "arrived"

    def test_deterministic_with_seed(self):
        """Same seed → same number of requests."""
        counts = []
        for _ in range(2):
            ctx = _make_ctx(duration=10.0, seed=99)
            svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])
            gen = PoissonFixedSizeGenerator(arrival_rate=10.0)
            gen.attach(ctx)
            gen.start_for_service(svc)
            ctx.env.run(until=10.0)
            counts.append(len(ctx.request_table))
        assert counts[0] == counts[1]


# ------------------------------------------------------------------ #
# WorkloadManager tests
# ------------------------------------------------------------------ #

class TestWorkloadManager:
    def test_from_config(self):
        ctx = _make_ctx()
        wm = WorkloadManager.from_config(ctx)
        assert "svc-a" in wm.services

    def test_start_generates_requests(self):
        ctx = _make_ctx(duration=10.0, seed=42)
        wm = WorkloadManager.from_config(ctx)
        wm.start()

        ctx.env.run(until=10.0)

        n = len(ctx.request_table)
        assert n > 0, "No requests generated"
        assert 50 < n < 200, f"Expected ~100 requests, got {n}"

    def test_register_multiple_services(self):
        ctx = _make_ctx()
        wm = WorkloadManager(ctx)
        svc_a = ServiceClass(service_id="svc-a",
                             max_concurrency=1)
        svc_b = ServiceClass(service_id="svc-b",
                             max_concurrency=2)
        wm.register_service(svc_a)
        wm.register_service(svc_b)
        wm.start()

        ctx.env.run(until=10.0)

        service_ids = {inv.service_id for inv in ctx.request_table.values()}
        assert "svc-a" in service_ids
        assert "svc-b" in service_ids


# ------------------------------------------------------------------ #
# AggregateTraceGenerator tests
# ------------------------------------------------------------------ #

import os
import tempfile

from serverless_sim.workload.trace_generator import AggregateTraceGenerator


class TestAggregateTraceGenerator:
    def _write_csv(self, rows: list[str]) -> str:
        """Write CSV content to a temp file and return path."""
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="test_agg_")
        with os.fdopen(fd, "w") as f:
            f.write("minute,count,duration\n")
            for row in rows:
                f.write(row + "\n")
        return path

    def test_total_request_count(self):
        """5 requests in minute 0 + 10 in minute 1 = 15 total."""
        path = self._write_csv([
            "0,5,0.1",
            "1,10,0.1",
        ])
        ctx = _make_ctx(duration=120.0)
        svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])

        gen = AggregateTraceGenerator(path)
        gen.attach(ctx)
        gen.start_for_service(svc)

        ctx.env.run(until=120.0)

        assert len(ctx.request_table) == 15

    def test_even_spacing_within_minute(self):
        """10 requests in minute 0 → interval = 6s, times at 0, 6, 12, ..., 54."""
        path = self._write_csv([
            "0,10,0.1",
        ])
        ctx = _make_ctx(duration=60.0)
        svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])

        gen = AggregateTraceGenerator(path)
        gen.attach(ctx)
        gen.start_for_service(svc)

        ctx.env.run(until=60.0)

        times = sorted(inv.arrival_time for inv in ctx.request_table.values())
        assert len(times) == 10
        # Interval should be 6.0s
        for i in range(1, len(times)):
            assert abs(times[i] - times[i - 1] - 6.0) < 0.001

    def test_stop_time_respected(self):
        """Generator stops when stop_time is reached."""
        path = self._write_csv([
            "0,5,0.1",   # interval=12s, times: 0,12,24,36,48
            "1,10,0.1",  # minute 1 = 60s, beyond stop_time=30
        ])
        ctx = _make_ctx(duration=30.0)
        svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])

        gen = AggregateTraceGenerator(path)
        gen.attach(ctx)
        gen.start_for_service(svc, stop_time=30.0)

        ctx.env.run(until=120.0)

        # stop_time=30 → only requests at t=0,12,24 pass (3 requests)
        # t=36 exceeds stop_time, minute 1 never starts
        assert len(ctx.request_table) == 3

    def test_zero_count_rows_skipped(self):
        """Rows with count=0 are ignored."""
        path = self._write_csv([
            "0,0,0.1",
            "1,3,0.1",
        ])
        ctx = _make_ctx(duration=120.0)
        svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])

        gen = AggregateTraceGenerator(path)
        gen.attach(ctx)
        gen.start_for_service(svc)

        ctx.env.run(until=120.0)

        assert len(ctx.request_table) == 3

    def test_multiple_services_separate_files(self):
        """Per-service architecture: each service has its own generator and file."""
        path_a = self._write_csv(["0,3,0.1"])
        path_b = self._write_csv(["0,2,0.5"])
        ctx = _make_ctx(duration=60.0)
        svc_a = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])
        svc_b = ServiceClass(service_id="svc-b", max_concurrency=1)

        gen_a = AggregateTraceGenerator(path_a)
        gen_a.attach(ctx)
        gen_a.start_for_service(svc_a)

        gen_b = AggregateTraceGenerator(path_b)
        gen_b.attach(ctx)
        gen_b.start_for_service(svc_b)

        ctx.env.run(until=60.0)

        service_ids = {inv.service_id for inv in ctx.request_table.values()}
        assert "svc-a" in service_ids
        assert "svc-b" in service_ids
        assert len(ctx.request_table) == 5
