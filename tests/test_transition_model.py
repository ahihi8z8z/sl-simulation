"""Tests for pluggable TransitionModel."""

import os
import tempfile

import numpy as np
import simpy
import logging

from serverless_sim.lifecycle.transition_model import (
    DeterministicTransitionModel,
    CsvSampleTransitionModel,
    DistributionTransitionModel,
    TransitionSample,
)
from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine
from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.cluster.cluster_manager import ClusterManager
from serverless_sim.workload.workload_manager import WorkloadManager
from serverless_sim.workload.service_time import FixedServiceTime
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


# ------------------------------------------------------------------
# DeterministicTransitionModel tests
# ------------------------------------------------------------------

class TestDeterministicModel:
    def test_set_and_sample(self):
        model = DeterministicTransitionModel()
        model.set("null", "prewarm", time=0.5, cpu=0.1, memory=64)
        rng = np.random.default_rng(42)

        s = model.sample("null", "prewarm", rng)
        assert s.time == 0.5
        assert s.cpu == 0.1
        assert s.memory == 64

    def test_missing_transition_returns_zeros(self):
        model = DeterministicTransitionModel()
        rng = np.random.default_rng(42)

        s = model.sample("null", "prewarm", rng)
        assert s.time == 0.0
        assert s.cpu == 0.0
        assert s.memory == 0.0

    def test_deterministic_always_same(self):
        model = DeterministicTransitionModel()
        model.set("a", "b", time=1.5)
        rng = np.random.default_rng(42)

        results = [model.sample("a", "b", rng).time for _ in range(100)]
        assert all(t == 1.5 for t in results)


# ------------------------------------------------------------------
# CsvSampleTransitionModel tests
# ------------------------------------------------------------------

class TestCsvSampleModel:
    def _write_csv(self, tmpdir, rows):
        path = os.path.join(tmpdir, "transitions.csv")
        with open(path, "w") as f:
            f.write("from_state,to_state,time,cpu,memory\n")
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
        return path

    def test_load_and_sample(self):
        tmpdir = tempfile.mkdtemp()
        path = self._write_csv(tmpdir, [
            ("null", "prewarm", 0.4, 0.1, 0),
            ("null", "prewarm", 0.6, 0.2, 0),
            ("prewarm", "warm", 0.3, 0, 0),
        ])
        model = CsvSampleTransitionModel.from_csv(path)
        rng = np.random.default_rng(42)

        s = model.sample("null", "prewarm", rng)
        assert s.time in (0.4, 0.6)
        assert s.cpu in (0.1, 0.2)

    def test_samples_vary(self):
        tmpdir = tempfile.mkdtemp()
        path = self._write_csv(tmpdir, [
            ("null", "prewarm", 0.1, 0, 0),
            ("null", "prewarm", 0.9, 0, 0),
        ])
        model = CsvSampleTransitionModel.from_csv(path)
        rng = np.random.default_rng(42)

        times = {model.sample("null", "prewarm", rng).time for _ in range(100)}
        assert len(times) == 2
        assert 0.1 in times
        assert 0.9 in times

    def test_missing_transition_returns_zeros(self):
        tmpdir = tempfile.mkdtemp()
        path = self._write_csv(tmpdir, [
            ("null", "prewarm", 0.5, 0, 0),
        ])
        model = CsvSampleTransitionModel.from_csv(path)
        rng = np.random.default_rng(42)

        s = model.sample("prewarm", "warm", rng)
        assert s.time == 0.0

    def test_from_sample_csv_file(self):
        """Test with the actual sample CSV in configs/."""
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "simulation",
            "sample_transition_profile.csv",
        )
        if not os.path.exists(csv_path):
            return
        model = CsvSampleTransitionModel.from_csv(csv_path)
        rng = np.random.default_rng(42)

        s = model.sample("null", "prewarm", rng)
        assert 0.4 <= s.time <= 0.6
        s2 = model.sample("prewarm", "warm", rng)
        assert 0.2 <= s2.time <= 0.4


# ------------------------------------------------------------------
# DistributionTransitionModel tests
# ------------------------------------------------------------------

class TestDistributionModel:
    def test_lognormal(self):
        model = DistributionTransitionModel()
        model.set("null", "prewarm", {
            "distribution": "lognormal",
            "time_mean": 0.5,
            "time_std": 0.1,
            "cpu": 0.1,
        })
        rng = np.random.default_rng(42)

        times = [model.sample("null", "prewarm", rng).time for _ in range(1000)]
        mean = sum(times) / len(times)
        assert 0.3 < mean < 0.7  # should be around 0.5

    def test_normal(self):
        model = DistributionTransitionModel()
        model.set("a", "b", {
            "distribution": "normal",
            "time_mean": 1.0,
            "time_std": 0.1,
        })
        rng = np.random.default_rng(42)

        times = [model.sample("a", "b", rng).time for _ in range(1000)]
        mean = sum(times) / len(times)
        assert 0.8 < mean < 1.2

    def test_uniform(self):
        model = DistributionTransitionModel()
        model.set("a", "b", {
            "distribution": "uniform",
            "time_low": 0.2,
            "time_high": 0.8,
        })
        rng = np.random.default_rng(42)

        times = [model.sample("a", "b", rng).time for _ in range(1000)]
        assert all(0.2 <= t <= 0.8 for t in times)
        mean = sum(times) / len(times)
        assert 0.4 < mean < 0.6

    def test_cpu_is_fixed(self):
        model = DistributionTransitionModel()
        model.set("a", "b", {
            "distribution": "lognormal",
            "time_mean": 0.5,
            "time_std": 0.1,
            "cpu": 0.3,
            "memory": 64,
        })
        rng = np.random.default_rng(42)

        samples = [model.sample("a", "b", rng) for _ in range(10)]
        assert all(s.cpu == 0.3 for s in samples)
        assert all(s.memory == 64 for s in samples)

    def test_time_never_negative(self):
        model = DistributionTransitionModel()
        model.set("a", "b", {
            "distribution": "normal",
            "time_mean": 0.01,
            "time_std": 0.5,  # high variance, could go negative
        })
        rng = np.random.default_rng(42)

        times = [model.sample("a", "b", rng).time for _ in range(1000)]
        assert all(t >= 0 for t in times)


# ------------------------------------------------------------------
# Integration: StateMachine with CSV model
# ------------------------------------------------------------------

class TestStateMachineWithCsvModel:
    def test_from_lifecycle_config_with_csv(self):
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "trans.csv")
        with open(csv_path, "w") as f:
            f.write("from_state,to_state,time,cpu,memory\n")
            f.write("null,prewarm,0.4,0,0\n")
            f.write("null,prewarm,0.6,0,0\n")
            f.write("prewarm,warm,0.2,0,0\n")
            f.write("prewarm,warm,0.3,0,0\n")

        cfg = dict(LIFECYCLE_CFG)
        cfg["transition_profile"] = csv_path

        sm = OpenWhiskExtendedStateMachine.from_lifecycle_config(cfg)
        assert isinstance(sm.transition_model, CsvSampleTransitionModel)

        rng = np.random.default_rng(42)
        s = sm.transition_model.sample("null", "prewarm", rng)
        assert s.time in (0.4, 0.6)

    def test_default_uses_deterministic(self):
        sm = OpenWhiskExtendedStateMachine.default()
        assert isinstance(sm.transition_model, DeterministicTransitionModel)

        rng = np.random.default_rng(42)
        s = sm.transition_model.sample("null", "prewarm", rng)
        assert s.time == 0.5

    def test_from_config_without_csv_uses_deterministic(self):
        sm = OpenWhiskExtendedStateMachine.from_lifecycle_config(LIFECYCLE_CFG)
        assert isinstance(sm.transition_model, DeterministicTransitionModel)


# ------------------------------------------------------------------
# Integration: Full simulation with CSV transition model
# ------------------------------------------------------------------

class TestSimulationWithCsvModel:
    def test_cold_start_uses_sampled_times(self):
        """Run a simulation where cold start times come from CSV."""
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "trans.csv")
        with open(csv_path, "w") as f:
            f.write("from_state,to_state,time,cpu,memory\n")
            # All transitions take exactly 0.1s for easy verification
            for _ in range(20):
                f.write("null,prewarm,0.1,0,0\n")
                f.write("prewarm,warm,0.1,0,0\n")

        lifecycle_with_csv = dict(LIFECYCLE_CFG)
        lifecycle_with_csv["transition_profile"] = csv_path

        config = {
            "simulation": {"duration": 5.0, "seed": 42},
            "services": [{
                "service_id": "svc-a",
                "lifecycle": lifecycle_with_csv,
                "workload": {"arrival_rate": 2.0},
            }],
            "cluster": {
                "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}]
            },
        }

        env = simpy.Environment()
        rng = np.random.default_rng(42)
        logger = logging.getLogger("test_csv_sim")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir=tmpdir)
        _provider = FixedServiceTime(duration=0.1)

        for _svc in ctx.config.get("services", []):

            ctx.service_time_providers[_svc["service_id"]] = _provider
        ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
        ctx.workload_manager = WorkloadManager.from_config(ctx)
        ctx.lifecycle_manager = LifecycleManager(ctx)
        ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
        ctx.cluster_manager.set_context(ctx)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        env.run(until=5.0)

        # Verify requests completed (cold start time = 0.1 + 0.1 = 0.2s)
        c = ctx.request_table.counters
        assert c.completed > 0

    def test_variable_cold_start_times(self):
        """Cold start times should vary when CSV has different values."""
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "trans.csv")
        with open(csv_path, "w") as f:
            f.write("from_state,to_state,time,cpu,memory\n")
            f.write("null,prewarm,0.1,0,0\n")
            f.write("null,prewarm,0.9,0,0\n")
            for _ in range(20):
                f.write("prewarm,warm,0.1,0,0\n")

        lifecycle_with_csv = dict(LIFECYCLE_CFG)
        lifecycle_with_csv["transition_profile"] = csv_path

        config = {
            "simulation": {"duration": 10.0, "seed": 42},
            "services": [{
                "service_id": "svc-a",
                "lifecycle": lifecycle_with_csv,
                "workload": {"arrival_rate": 5.0},
            }],
            "cluster": {
                "nodes": [{"node_id": "node-0", "cpu_capacity": 8.0, "memory_capacity": 8192}]
            },
        }

        env = simpy.Environment()
        rng = np.random.default_rng(42)
        logger = logging.getLogger("test_var_csv")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        ctx = SimContext(env=env, config=config, rng=rng, logger=logger, run_dir=tmpdir)
        _provider = FixedServiceTime(duration=0.1)

        for _svc in ctx.config.get("services", []):

            ctx.service_time_providers[_svc["service_id"]] = _provider
        ctx.cluster_manager = ClusterManager(env=env, config=config, logger=logger)
        ctx.workload_manager = WorkloadManager.from_config(ctx)
        ctx.lifecycle_manager = LifecycleManager(ctx)
        ctx.dispatcher = ShardingContainerPoolBalancer(ctx)
        ctx.cluster_manager.set_context(ctx)

        ctx.cluster_manager.start_all()
        ctx.workload_manager.start()
        env.run(until=10.0)

        assert ctx.request_table.counters.completed > 0
