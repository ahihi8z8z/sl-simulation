from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class SummaryWriter:
    """Writes summary.txt at end of simulation."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    def write(self, wall_clock_seconds: float | None = None) -> str:
        """Write summary.txt and return the file path.

        Parameters
        ----------
        wall_clock_seconds : float | None
            Wall-clock time the simulation took (seconds).
        """
        run_dir = self.ctx.run_dir
        path = os.path.join(run_dir, "summary.txt")

        store = self.ctx.request_table
        c = store.counters
        total = c.total
        completed = c.completed
        dropped = c.dropped
        truncated = c.truncated
        cold_starts = c.cold_starts

        config = self.ctx.config
        duration = config["simulation"]["duration"]

        sim_end_time = self.ctx.env.now

        with open(path, "w") as f:
            f.write("=== Simulation Summary ===\n\n")
            f.write(f"Duration (config): {duration:.1f}s\n")
            f.write(f"Simulation end time: {sim_end_time:.1f}s\n")
            if wall_clock_seconds is not None:
                f.write(f"Wall-clock time: {wall_clock_seconds:.2f}s\n")
            f.write(f"Seed: {config['simulation']['seed']}\n")
            f.write(f"Services: {len(config['services'])}\n")
            f.write(f"Nodes: {len(config['cluster']['nodes'])}\n\n")

            f.write("--- Request Statistics ---\n")
            f.write(f"Total requests: {total}\n")
            f.write(f"Completed: {completed}\n")
            f.write(f"Dropped: {dropped}\n")
            f.write(f"Truncated: {truncated}\n")
            f.write(f"Cold starts: {cold_starts}\n")

            if completed > 0:
                f.write(f"\n--- Latency (seconds) ---\n")
                f.write(f"Mean: {store.latency_mean:.4f}\n")

            if total > 0:
                f.write(f"\n--- Rates ---\n")
                f.write(f"Throughput: {completed / duration:.2f} req/s\n")
                f.write(f"Drop rate: {dropped / total * 100:.1f}%\n")
                f.write(f"Cold start rate: {cold_starts / max(completed, 1) * 100:.1f}%\n")

        return path
