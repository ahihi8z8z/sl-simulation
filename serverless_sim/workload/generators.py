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

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        raise NotImplementedError


class GammaArrivalGenerator(BaseGenerator):
    """Gamma-distributed inter-arrival times.

    Inter-arrival ~ Gamma(alpha, 1/beta) where:
      alpha (shape): <1 = bursty, =1 = Poisson, >1 = regular
      beta (rate): controls speed. Mean inter-arrival = alpha/beta.
      Mean arrival rate = beta/alpha.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.ctx: SimContext | None = None
        self._request_counter = 0
        self._alpha = alpha
        self._beta = beta

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        self.ctx.env.process(self._arrival_loop(service, stop_time))

    def _arrival_loop(self, service: ServiceClass, stop_time: float | None = None):
        ctx = self.ctx
        rng = ctx.rng
        env = ctx.env

        # numpy gamma: shape=alpha, scale=1/beta
        scale = 1.0 / self._beta

        while True:
            if stop_time is not None and env.now >= stop_time:
                return

            interval = rng.gamma(self._alpha, scale)
            yield env.timeout(interval)

            if stop_time is not None and env.now >= stop_time:
                return

            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

            inv = Invocation(
                request_id=request_id,
                service_id=service.service_id,
                arrival_time=env.now,
                job_size=service.job_size,
                status="arrived",
            )

            ctx.request_table[request_id] = inv

            ctx.logger.debug(
                "t=%.3f | ARRIVE | %s service=%s",
                env.now, request_id, service.service_id,
            )

            if ctx.dispatcher is not None:
                ctx.dispatcher.dispatch(inv)


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

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        self.ctx.env.process(self._arrival_loop(service, stop_time))

    def _arrival_loop(self, service: ServiceClass, stop_time: float | None = None):
        """SimPy process: generate requests with exponential inter-arrival."""
        ctx = self.ctx
        rng = ctx.rng
        env = ctx.env

        mean_interval = 1.0 / service.arrival_rate

        while True:
            if stop_time is not None and env.now >= stop_time:
                ctx.logger.debug(
                    "t=%.3f | GENERATOR_STOP | %s (stop_time=%.1f)",
                    env.now, service.service_id, stop_time,
                )
                return
            # Exponential inter-arrival time
            interval = rng.exponential(mean_interval)
            yield env.timeout(interval)

            # Check again after waiting — time may have crossed stop_time
            if stop_time is not None and env.now >= stop_time:
                ctx.logger.debug(
                    "t=%.3f | GENERATOR_STOP | %s (stop_time=%.1f)",
                    env.now, service.service_id, stop_time,
                )
                return

            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

            inv = Invocation(
                request_id=request_id,
                service_id=service.service_id,
                arrival_time=env.now,
                job_size=service.job_size,
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
