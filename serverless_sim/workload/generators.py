from __future__ import annotations

from typing import TYPE_CHECKING

from serverless_sim.workload.invocation import Invocation

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.workload.service_class import ServiceClass


class BaseGenerator:
    """Interface for workload generators."""

    def attach(self, ctx: SimContext) -> None:
        raise NotImplementedError

    def start_for_service(self, service: ServiceClass) -> None:
        raise NotImplementedError


class PoissonFixedSizeGenerator(BaseGenerator):
    """Exponential inter-arrival with fixed request size.

    Each generated Invocation is placed directly into the
    ``ctx.request_table`` and yielded via a callback so that
    downstream components (load balancer, etc.) can route it.
    """

    def __init__(self):
        self.ctx: SimContext | None = None
        self._request_counter = 0

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx

    def start_for_service(self, service: ServiceClass) -> None:
        """Kick off the SimPy arrival process for *service*."""
        self.ctx.env.process(self._arrival_loop(service))

    def _arrival_loop(self, service: ServiceClass):
        """SimPy process: generate requests with exponential inter-arrival."""
        ctx = self.ctx
        rng = ctx.rng
        env = ctx.env

        mean_interval = 1.0 / service.arrival_rate

        while True:
            # Exponential inter-arrival time
            interval = rng.exponential(mean_interval)
            yield env.timeout(interval)

            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

            inv = Invocation(
                request_id=request_id,
                service_id=service.service_id,
                arrival_time=env.now,
                job_size=service.job_size,
                timeout=service.timeout,
                status="arrived",
            )

            # Register in central request table
            ctx.request_table[request_id] = inv

            ctx.logger.debug(
                "t=%.3f | ARRIVE | %s service=%s job_size=%.3f",
                env.now,
                request_id,
                service.service_id,
                service.job_size,
            )

            # Dispatch through load balancer if available
            if ctx.dispatcher is not None:
                ctx.dispatcher.dispatch(inv)
