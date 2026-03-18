from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class SimulationEngine:
    """Top-level runtime object for standalone simulation."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    def setup(self) -> None:
        """Start all SimPy processes."""
        self.ctx.cluster_manager.start_all()
        self.ctx.workload_manager.start()
        self.ctx.monitor_manager.start()
        self.ctx.logger.info("SimulationEngine: setup complete")

    def run(self, until: float | None = None) -> None:
        """Run the simulation."""
        if until is None:
            until = self.ctx.config["simulation"]["duration"]
        self.ctx.logger.info("SimulationEngine: running until t=%.1f", until)
        self.ctx.env.run(until=until)
        self.ctx.logger.info("SimulationEngine: simulation finished at t=%.3f", self.ctx.env.now)

    def shutdown(self) -> None:
        """Export results and clean up."""
        if self.ctx.export_manager:
            self.ctx.export_manager.export()
        self.ctx.logger.info("SimulationEngine: shutdown complete")

    def get_snapshot(self) -> dict:
        """Collect a one-time metric snapshot."""
        self.ctx.monitor_manager.collect_once()
        from serverless_sim.monitoring.monitor_api import MonitorAPI
        api = MonitorAPI(self.ctx.monitor_manager)
        return api.get_snapshot()
