from __future__ import annotations

from typing import TYPE_CHECKING

from serverless_sim.workload.generators import BaseGenerator, PoissonFixedSizeGenerator
from serverless_sim.workload.service_class import ServiceClass

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class WorkloadManager:
    """Manages services and workload generation."""

    def __init__(self, ctx: SimContext, generator: BaseGenerator | None = None):
        self.ctx = ctx
        self.services: dict[str, ServiceClass] = {}
        self.generator = generator or PoissonFixedSizeGenerator()
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
        """Build a WorkloadManager and register services from config.

        If ``config["workload"]["generator"]`` is ``"trace"`` and
        ``config["workload"]["trace_path"]`` is set, uses
        TraceReplayGenerator instead of PoissonFixedSizeGenerator.
        """
        workload_cfg = ctx.config.get("workload", {})
        gen_type = workload_cfg.get("generator", "poisson")

        start_minute = workload_cfg.get("start_minute", None)
        end_minute = workload_cfg.get("end_minute", None)
        column_map = workload_cfg.get("column_map", None)

        generator: BaseGenerator
        if gen_type == "trace":
            from serverless_sim.workload.trace_generator import TraceReplayGenerator
            trace_path = workload_cfg["trace_path"]
            generator = TraceReplayGenerator(trace_path, start_minute=start_minute,
                                             end_minute=end_minute,
                                             column_map=column_map)
        elif gen_type == "aggregate_trace":
            from serverless_sim.workload.trace_generator import AggregateTraceGenerator
            trace_path = workload_cfg["trace_path"]
            scale = workload_cfg.get("scale", 1.0)
            generator = AggregateTraceGenerator(trace_path, scale=scale,
                                                start_minute=start_minute,
                                                end_minute=end_minute,
                                                column_map=column_map)
        elif gen_type == "gamma":
            from serverless_sim.workload.generators import GammaArrivalGenerator
            alpha = workload_cfg.get("gamma_alpha", 1.0)
            beta = workload_cfg.get("gamma_beta", 1.0)
            generator = GammaArrivalGenerator(alpha=alpha, beta=beta)
        elif gen_type == "gamma_window":
            from serverless_sim.workload.generators import GammaWindowGenerator
            trace_path = workload_cfg["trace_path"]
            scale_alpha = workload_cfg.get("scale_alpha", 1.0)
            scale_beta = workload_cfg.get("scale_beta", 1.0)
            generator = GammaWindowGenerator(csv_path=trace_path,
                                              scale_alpha=scale_alpha,
                                              scale_beta=scale_beta)
        else:
            generator = PoissonFixedSizeGenerator()

        wm = cls(ctx, generator=generator)
        for svc_cfg in ctx.config["services"]:
            service = ServiceClass.from_config(svc_cfg)
            wm.register_service(service)
        return wm
