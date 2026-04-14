from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class SimulationEngine:
    """Top-level runtime object for standalone simulation."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self._wall_start: float | None = None
        self._wall_end: float | None = None

    def setup(self) -> None:
        """Start all SimPy processes."""
        duration = self.ctx.config["simulation"]["duration"]
        start_delay = self.ctx.config.get("workload", {}).get("start_delay", 0)
        self.ctx.cluster_manager.start_all()
        self.ctx.workload_manager.start(stop_time=duration + start_delay)
        self.ctx.monitor_manager.start()
        if self.ctx.autoscaling_manager:
            self.ctx.autoscaling_manager.start()
        if self.ctx.controller:
            self.ctx.controller.start()
        self.ctx.logger.info("SimulationEngine: setup complete")

    def run(self, until: float | None = None, progress: bool = False) -> None:
        """Run the simulation with an optional drain period.

        The drain period lets in-flight requests finish after workload
        generation stops at ``duration``.  Configure via
        ``config["simulation"]["drain_timeout"]`` (default: 30s).

        Parameters
        ----------
        progress : bool
            If True, show a tqdm progress bar (requires ``tqdm``).
        """
        config_sim = self.ctx.config["simulation"]
        duration = config_sim["duration"]
        start_delay = self.ctx.config.get("workload", {}).get("start_delay", 0)
        if until is None:
            until = duration + start_delay

        drain_timeout = self._get_drain_timeout()
        total = until + drain_timeout

        self.ctx.logger.info(
            "SimulationEngine: running until t=%.1f (drain=%.1f, total=%.1f)",
            until, drain_timeout, total,
        )
        self._wall_start = time.monotonic()

        if progress:
            self._run_with_progress(total)
        else:
            self.ctx.env.run(until=total)

        self._wall_end = time.monotonic()
        self.ctx.logger.info("SimulationEngine: simulation finished at t=%.3f", self.ctx.env.now)

    def _run_with_progress(self, total: float, chunk: float = 1.0) -> None:
        """Run simulation in chunks with a tqdm progress bar."""
        try:
            from tqdm import tqdm
        except ImportError:
            self.ctx.logger.warning("tqdm not installed, running without progress bar")
            self.ctx.env.run(until=total)
            return

        env = self.ctx.env
        store = self.ctx.request_table
        with tqdm(total=total, unit="s", desc="Sim") as pbar:
            while env.now < total:
                next_stop = min(env.now + chunk, total)
                env.run(until=next_stop)
                pbar.update(next_stop - pbar.n)
                c = store.counters
                pbar.set_postfix_str(
                    f"req={c.total:,} done={c.completed:,} drop={c.dropped:,}",
                    refresh=False,
                )

    def _get_drain_timeout(self) -> float:
        """Return the drain timeout from config (default 30s)."""
        config_sim = self.ctx.config["simulation"]
        drain = config_sim.get("drain_timeout")
        if drain is not None:
            return float(drain)
        return 30.0

    def shutdown(self) -> None:
        """Mark in-flight requests as truncated, then export results."""
        self._mark_truncated()
        wall_clock = None
        if self._wall_start is not None and self._wall_end is not None:
            wall_clock = self._wall_end - self._wall_start
        if self.ctx.export_manager:
            self.ctx.export_manager.export(wall_clock_seconds=wall_clock)
        self.ctx.logger.info("SimulationEngine: shutdown complete")

    def _mark_truncated(self) -> None:
        """Sweep in-flight requests and mark them as truncated."""
        # Collect list first since finalize() mutates _active
        in_flight = list(self.ctx.request_table.values())
        truncated_count = 0
        for inv in in_flight:
            inv.status = "truncated"
            inv.drop_reason = "simulation_end"
            if inv.completion_time is None:
                inv.completion_time = self.ctx.env.now
            self.ctx.request_table.finalize(inv)
            truncated_count += 1
        if truncated_count > 0:
            self.ctx.logger.info(
                "SimulationEngine: marked %d in-flight requests as truncated",
                truncated_count,
            )

    def get_snapshot(self) -> dict:
        """Collect a one-time metric snapshot."""
        self.ctx.monitor_manager.collect_once()
        from serverless_sim.monitoring.monitor_api import MonitorAPI
        api = MonitorAPI(self.ctx.monitor_manager)
        return api.get_snapshot()
