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
        self.ctx.cluster_manager.start_all()
        self.ctx.workload_manager.start(stop_time=duration)
        self.ctx.monitor_manager.start()
        if self.ctx.autoscaling_manager:
            self.ctx.autoscaling_manager.start()
        if self.ctx.controller:
            self.ctx.controller.start()
        self.ctx.logger.info("SimulationEngine: setup complete")

    def run(self, until: float | None = None) -> None:
        """Run the simulation with an optional drain period.

        The drain period lets in-flight requests finish after workload
        generation stops at ``duration``.  Configure via
        ``config["simulation"]["drain_timeout"]`` (default: max service
        timeout).
        """
        config_sim = self.ctx.config["simulation"]
        duration = config_sim["duration"]
        if until is None:
            until = duration

        drain_timeout = self._get_drain_timeout()
        total = until + drain_timeout

        self.ctx.logger.info(
            "SimulationEngine: running until t=%.1f (drain=%.1f, total=%.1f)",
            until, drain_timeout, total,
        )
        self._wall_start = time.monotonic()
        self.ctx.env.run(until=total)
        self._wall_end = time.monotonic()
        self.ctx.logger.info("SimulationEngine: simulation finished at t=%.3f", self.ctx.env.now)

    def _get_drain_timeout(self) -> float:
        """Return the drain timeout from config, defaulting to max service timeout."""
        config_sim = self.ctx.config["simulation"]
        drain = config_sim.get("drain_timeout")
        if drain is not None:
            return float(drain)
        # Default: max timeout across all services
        services = self.ctx.config.get("services", [])
        if services:
            return max(svc.get("timeout", 30.0) for svc in services)
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
        """Sweep request table and mark non-terminal requests as truncated."""
        terminal_statuses = {"completed", "timed_out"}
        truncated_count = 0
        for inv in self.ctx.request_table.values():
            if inv.status not in terminal_statuses and not inv.dropped:
                inv.status = "truncated"
                inv.drop_reason = "simulation_end"
                if inv.completion_time is None:
                    inv.completion_time = self.ctx.env.now
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
