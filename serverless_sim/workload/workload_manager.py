from __future__ import annotations

from typing import TYPE_CHECKING

from serverless_sim.workload.generators import PoissonFixedSizeGenerator
from serverless_sim.workload.service_class import ServiceClass

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class WorkloadManager:
    """Manages services and workload generation."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.services: dict[str, ServiceClass] = {}
        self.generator = PoissonFixedSizeGenerator()
        self.generator.attach(ctx)

    def register_service(self, service: ServiceClass) -> None:
        """Register a service class."""
        self.services[service.service_id] = service
        self.ctx.logger.info("Registered service: %s (rate=%.1f)", service.service_id, service.arrival_rate)

    def start(self, stop_time: float | None = None) -> None:
        """Start arrival generators for all registered services.

        Parameters
        ----------
        stop_time : float | None
            If set, generators stop producing requests after this time.
        """
        for service in self.services.values():
            self.generator.start_for_service(service, stop_time=stop_time)
            self.ctx.logger.info("Started generator for service: %s", service.service_id)

    @classmethod
    def from_config(cls, ctx: SimContext) -> "WorkloadManager":
        """Build a WorkloadManager and register services from config."""
        wm = cls(ctx)
        for svc_cfg in ctx.config["services"]:
            service = ServiceClass.from_config(svc_cfg)
            wm.register_service(service)
        return wm
