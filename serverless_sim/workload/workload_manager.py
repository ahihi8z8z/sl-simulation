from __future__ import annotations

from typing import TYPE_CHECKING

from serverless_sim.workload.generators import BaseGenerator, PoissonFixedSizeGenerator
from serverless_sim.workload.service_class import ServiceClass

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


def _build_generator(workload_cfg: dict) -> BaseGenerator:
    """Build a single generator instance from a per-service workload config."""
    gen_type = workload_cfg.get("generator", "poisson")

    start_minute = workload_cfg.get("start_minute", None)
    end_minute = workload_cfg.get("end_minute", None)
    column_map = workload_cfg.get("column_map", None)

    if gen_type == "trace":
        from serverless_sim.workload.trace_generator import TraceReplayGenerator
        trace_path = workload_cfg["trace_path"]
        scale = int(workload_cfg.get("scale", 1))
        return TraceReplayGenerator(
            trace_path,
            start_minute=start_minute,
            end_minute=end_minute,
            column_map=column_map,
            scale=scale,
        )

    if gen_type == "aggregate_trace":
        from serverless_sim.workload.trace_generator import AggregateTraceGenerator
        trace_path = workload_cfg["trace_path"]
        scale = workload_cfg.get("scale", 1.0)
        return AggregateTraceGenerator(
            trace_path,
            scale=scale,
            start_minute=start_minute,
            end_minute=end_minute,
            column_map=column_map,
        )

    if gen_type == "gamma":
        from serverless_sim.workload.generators import GammaArrivalGenerator
        alpha = workload_cfg.get("gamma_alpha", 1.0)
        beta = workload_cfg.get("gamma_beta", 1.0)
        return GammaArrivalGenerator(alpha=alpha, beta=beta)

    if gen_type == "gamma_window":
        from serverless_sim.workload.generators import GammaWindowGenerator
        trace_path = workload_cfg["trace_path"]
        scale_alpha = workload_cfg.get("scale_alpha", 1.0)
        scale_beta = workload_cfg.get("scale_beta", 1.0)
        return GammaWindowGenerator(
            csv_path=trace_path,
            scale_alpha=scale_alpha,
            scale_beta=scale_beta,
        )
        
    if gen_type == "weibull":
        from serverless_sim.workload.generators import WeibullGenerator
        shape = workload_cfg.get("weibull_shape", 1.0)
        scale = workload_cfg.get("weibull_scale", 1.0)
        limit = workload_cfg.get("weibull_limit", 1000)
        return WeibullGenerator(
            shape=shape, 
            scale=scale, 
            limit=limit)

    arrival_rate = workload_cfg.get("arrival_rate", 1.0)
    return PoissonFixedSizeGenerator(arrival_rate=arrival_rate)


class WorkloadManager:
    """Manages services and per-service workload generation.

    Each service owns its own generator instance built from
    ``services[i].workload``.  Services without a workload block fall
    back to a default Poisson generator.
    """

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.services: dict[str, ServiceClass] = {}
        self.generators: dict[str, BaseGenerator] = {}

    def register_service(
        self,
        service: ServiceClass,
        generator: BaseGenerator | None = None,
    ) -> None:
        """Register a service class and its generator."""
        if generator is None:
            generator = PoissonFixedSizeGenerator()
        generator.attach(self.ctx)
        self.services[service.service_id] = service
        self.generators[service.service_id] = generator
        self.ctx.logger.info(
            "Registered service: %s (generator=%s)",
            service.service_id, type(generator).__name__,
        )

    def start(self, stop_time: float | None = None) -> None:
        """Start arrival generators for all registered services."""
        for service_id, service in self.services.items():
            generator = self.generators[service_id]
            generator.start_for_service(service, stop_time=stop_time)
            self.ctx.logger.info("Started generator for service: %s", service_id)

    @classmethod
    def from_config(cls, ctx: SimContext) -> "WorkloadManager":
        """Build a WorkloadManager and register services from config.

        Each service entry may include a ``workload`` sub-block selecting
        its generator (``trace``, ``aggregate_trace``, ``gamma``,
        ``gamma_window``, or ``poisson``).  If absent, a default Poisson
        generator is used.
        """
        wm = cls(ctx)
        for svc_cfg in ctx.config["services"]:
            service = ServiceClass.from_config(svc_cfg)
            workload_cfg = svc_cfg.get("workload", {})
            generator = _build_generator(workload_cfg)
            wm.register_service(service, generator=generator)
        return wm
