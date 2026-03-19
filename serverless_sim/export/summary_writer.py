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

        table = self.ctx.request_table
        total = len(table)
        completed = sum(1 for inv in table.values() if inv.status == "completed")
        dropped = sum(1 for inv in table.values() if inv.dropped)
        timed_out = sum(1 for inv in table.values() if inv.timed_out)
        truncated = sum(1 for inv in table.values() if inv.status == "truncated")
        cold_starts = sum(1 for inv in table.values() if inv.cold_start and inv.status == "completed")

        # Latency stats for completed requests
        latencies = []
        for inv in table.values():
            if inv.status == "completed" and inv.completion_time and inv.arrival_time:
                latencies.append(inv.completion_time - inv.arrival_time)

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
            f.write(f"Timed out: {timed_out}\n")
            f.write(f"Truncated: {truncated}\n")
            f.write(f"Cold starts: {cold_starts}\n")

            if latencies:
                latencies.sort()
                avg = sum(latencies) / len(latencies)
                p50 = latencies[len(latencies) // 2]
                p95 = latencies[int(len(latencies) * 0.95)]
                p99 = latencies[int(len(latencies) * 0.99)]
                f.write(f"\n--- Latency (seconds) ---\n")
                f.write(f"Mean: {avg:.4f}\n")
                f.write(f"P50:  {p50:.4f}\n")
                f.write(f"P95:  {p95:.4f}\n")
                f.write(f"P99:  {p99:.4f}\n")

            if total > 0:
                f.write(f"\n--- Rates ---\n")
                f.write(f"Throughput: {completed / duration:.2f} req/s\n")
                f.write(f"Drop rate: {dropped / total * 100:.1f}%\n")
                f.write(f"Timeout rate: {timed_out / total * 100:.1f}%\n")
                f.write(f"Cold start rate: {cold_starts / max(completed, 1) * 100:.1f}%\n")

        return path
