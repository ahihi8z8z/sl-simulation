"""Unit tests for Step 3: Workload generation."""

import simpy
import numpy as np
import logging

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.workload.service_class import ServiceClass
from serverless_sim.workload.invocation import Invocation
from serverless_sim.workload.generators import PoissonFixedSizeGenerator
from serverless_sim.workload.workload_manager import WorkloadManager


SAMPLE_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42, "export_mode": 0},
    "services": [
        {
            "service_id": "svc-a",
            "arrival_rate": 10.0,
            "job_size": 0.5,
            "timeout": 5.0,
            "memory": 256,
            "cpu": 1.0,
            "max_concurrency": 2,
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
    return SimContext(
        env=env,
        config=SAMPLE_CONFIG,
        rng=rng,
        logger=logger,
        run_dir="/tmp/test_run",
    )


# ------------------------------------------------------------------ #
# ServiceClass tests
# ------------------------------------------------------------------ #

class TestServiceClass:
    def test_from_config(self):
        cfg = SAMPLE_CONFIG["services"][0]
        svc = ServiceClass.from_config(cfg)
        assert svc.service_id == "svc-a"
        assert svc.arrival_rate == 10.0
        assert svc.job_size == 0.5
        assert svc.timeout == 5.0
        assert svc.memory == 256
        assert svc.cpu == 1.0
        assert svc.max_concurrency == 2

    def test_defaults(self):
        cfg = {
            "service_id": "svc-b",
            "arrival_rate": 1.0,
            "job_size": 0.1,
            "timeout": 3.0,
            "memory": 128,
            "cpu": 0.5,
            "max_concurrency": 1,
        }
        svc = ServiceClass.from_config(cfg)
        assert svc.prewarm_count == 0
        assert svc.idle_timeout == 60.0
        assert svc.arrival_mode == "poisson"


# ------------------------------------------------------------------ #
# Invocation tests
# ------------------------------------------------------------------ #

class TestInvocation:
    def test_defaults(self):
        inv = Invocation(request_id="r1", service_id="svc-a", arrival_time=1.0)
        assert inv.status == "created"
        assert inv.dropped is False
        assert inv.timed_out is False
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

        gen = PoissonFixedSizeGenerator()
        gen.attach(ctx)
        gen.start_for_service(svc)

        ctx.env.run(until=10.0)

        n = len(ctx.request_table)
        # With rate=10 and duration=10, expect ~100; allow wide margin
        assert 50 < n < 200, f"Expected ~100 requests, got {n}"

    def test_arrival_times_valid(self):
        ctx = _make_ctx(duration=5.0, seed=123)
        svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])

        gen = PoissonFixedSizeGenerator()
        gen.attach(ctx)
        gen.start_for_service(svc)

        ctx.env.run(until=5.0)

        for inv in ctx.request_table.values():
            assert 0.0 < inv.arrival_time <= 5.0
            assert inv.service_id == "svc-a"
            assert inv.job_size == 0.5
            assert inv.status == "arrived"

    def test_deterministic_with_seed(self):
        """Same seed → same number of requests."""
        counts = []
        for _ in range(2):
            ctx = _make_ctx(duration=10.0, seed=99)
            svc = ServiceClass.from_config(SAMPLE_CONFIG["services"][0])
            gen = PoissonFixedSizeGenerator()
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
        assert wm.services["svc-a"].arrival_rate == 10.0

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
        svc_a = ServiceClass(service_id="svc-a", arrival_rate=5.0, job_size=0.1,
                             timeout=3.0, memory=128, cpu=0.5, max_concurrency=1)
        svc_b = ServiceClass(service_id="svc-b", arrival_rate=2.0, job_size=0.2,
                             timeout=5.0, memory=256, cpu=1.0, max_concurrency=2)
        wm.register_service(svc_a)
        wm.register_service(svc_b)
        wm.start()

        ctx.env.run(until=10.0)

        service_ids = {inv.service_id for inv in ctx.request_table.values()}
        assert "svc-a" in service_ids
        assert "svc-b" in service_ids
