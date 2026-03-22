"""Scale benchmark: measure simulation wall-clock time across scenarios."""

import json
import logging
import os
import sys
import tempfile
import time

import numpy as np
import simpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serverless_sim.core.simulation.sim_context import SimContext
from serverless_sim.core.simulation.sim_builder import SimulationBuilder
from serverless_sim.core.simulation.sim_engine import SimulationEngine


def make_config(n_services, n_nodes, arrival_rate_per_svc, duration, max_queue_depth=0):
    services = []
    for i in range(n_services):
        services.append({
            "service_id": f"svc-{i}",
            "arrival_rate": arrival_rate_per_svc,
            "job_size": 0.1,
            "memory": 256,
            "cpu": 0.5,
            "max_concurrency": 4,
            "min_instances": 1,
            "max_instances": 0,
        })

    nodes = []
    for i in range(n_nodes):
        nodes.append({
            "node_id": f"node-{i}",
            "cpu_capacity": 32.0,
            "memory_capacity": 65536,
            "max_queue_depth": max_queue_depth,
        })

    return {
        "simulation": {"duration": duration, "seed": 42, "export_mode": 0, "drain_timeout": 5.0},
        "services": services,
        "cluster": {"nodes": nodes},
        "autoscaling": {"enabled": True, "reconcile_interval": 1.0},
        "monitoring": {"interval": 1.0, "max_history_length": 100},
    }


def run_scenario(name, config, progress=False):
    run_dir = tempfile.mkdtemp(prefix="bench_")
    logger = logging.getLogger(f"bench_{name}")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)

    builder = SimulationBuilder()
    ctx = builder.build(config, run_dir, logger)
    engine = SimulationEngine(ctx)
    engine.setup()

    t0 = time.monotonic()
    engine.run(progress=progress)
    engine.shutdown()
    elapsed = time.monotonic() - t0

    total = len(ctx.request_table)
    completed = ctx.request_table.counters.completed
    dropped = ctx.request_table.counters.dropped

    print(f"  {name}:")
    print(f"    Duration: {config['simulation']['duration']}s sim")
    print(f"    Services: {len(config['services'])}, Nodes: {len(config['cluster']['nodes'])}")
    total_rate = sum(s['arrival_rate'] for s in config['services'])
    print(f"    Total arrival rate: {total_rate:.0f} req/s")
    print(f"    Expected requests: ~{total_rate * config['simulation']['duration']:.0f}")
    print(f"    Actual requests: {total}")
    print(f"    Completed: {completed}, Dropped: {dropped}")
    print(f"    Wall-clock: {elapsed:.3f}s")
    if total > 0:
        print(f"    Throughput: {total / elapsed:.0f} sim-requests/real-second")
    print()
    return elapsed


def main():
    print("=" * 60)
    print("Scale Benchmark")
    print("=" * 60)
    print()

    scenarios = [
        ("small", make_config(
            n_services=1, n_nodes=1, arrival_rate_per_svc=10, duration=10)),
        ("medium", make_config(
            n_services=3, n_nodes=2, arrival_rate_per_svc=50, duration=30)),
        ("large_requests", make_config(
            n_services=5, n_nodes=3, arrival_rate_per_svc=100, duration=60)),
        ("many_services", make_config(
            n_services=20, n_nodes=5, arrival_rate_per_svc=20, duration=30)),
        ("many_nodes", make_config(
            n_services=5, n_nodes=20, arrival_rate_per_svc=50, duration=30)),
        ("stress", make_config(
            n_services=10, n_nodes=10, arrival_rate_per_svc=200, duration=60)),
        ("extreme", make_config(
            n_services=10, n_nodes=10, arrival_rate_per_svc=20, duration=6000)),
    ]

    use_progress = "--progress" in sys.argv

    results = []
    for name, config in scenarios:
        if not use_progress:
            print(f"  Running {name}...", end=" ", flush=True)
        elapsed = run_scenario(name, config, progress=use_progress)
        results.append((name, elapsed))

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, elapsed in results:
        print(f"  {name:20s} {elapsed:8.3f}s")


if __name__ == "__main__":
    main()
