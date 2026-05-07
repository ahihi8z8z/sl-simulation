from __future__ import annotations

import csv as csv_mod
import math
from typing import TYPE_CHECKING

import numpy as np

from serverless_sim.workload.invocation import Invocation

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.workload.service_class import ServiceClass


class BaseGenerator:
    """Interface for workload generators.

    Each generator gets its own rng (spawned from ctx.rng at attach time)
    so that arrival timing is deterministic regardless of what other
    modules (lifecycle, autoscaler) do with the shared rng.
    """

    def attach(self, ctx: SimContext) -> None:
        raise NotImplementedError

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        raise NotImplementedError

    def _make_invocation(self, ctx: SimContext, service: ServiceClass) -> Invocation:
        """Create an Invocation and assign service_time via provider."""
        request_id = ctx.next_request_id()
        inv = Invocation(
            request_id=request_id,
            service_id=service.service_id,
            arrival_time=ctx.env.now,
            status="arrived",
        )
        provider = ctx.service_time_providers.get(service.service_id)
        if provider is not None:
            provider.assign(inv, self._rng)
        ctx.request_table[request_id] = inv
        return inv

    def _dispatch(self, ctx: SimContext, inv: Invocation) -> None:
        """Dispatch through load balancer if available."""
        if ctx.dispatcher is not None:
            ctx.dispatcher.dispatch(inv)


class GammaArrivalGenerator(BaseGenerator):
    """Gamma-distributed inter-arrival times.

    Inter-arrival ~ Gamma(alpha, 1/beta) where:
      alpha (shape): <1 = bursty, =1 = Poisson, >1 = regular
      beta (rate): controls speed. Mean inter-arrival = alpha/beta.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.ctx: SimContext | None = None
        self._rng = None
        self._alpha = alpha
        self._beta = beta

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self._rng = np.random.default_rng(ctx.rng.spawn(1)[0])

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        self.ctx.env.process(self._arrival_loop(service, stop_time))

    def _arrival_loop(self, service: ServiceClass, stop_time: float | None = None):
        ctx = self.ctx
        rng = self._rng
        env = ctx.env
        scale = 1.0 / self._beta

        while True:
            if stop_time is not None and env.now >= stop_time:
                return

            interval = rng.gamma(self._alpha, scale)
            yield env.timeout(interval)

            if stop_time is not None and env.now >= stop_time:
                return

            inv = self._make_invocation(ctx, service)
            ctx.logger.debug("t=%.3f | ARRIVE | %s service=%s", env.now, inv.request_id, service.service_id)
            self._dispatch(ctx, inv)


class GammaWindowGenerator(BaseGenerator):
    """Gamma-distributed inter-arrival times with time-varying parameters.

    Reads a gamma windows CSV where each row defines alpha/beta for a
    time window. Windows with invalid alpha/beta are skipped.
    """

    def __init__(self, csv_path: str, scale_alpha: float = 1.0, scale_beta: float = 1.0):
        self.ctx: SimContext | None = None
        self._rng = None
        self._csv_path = csv_path
        self._scale_alpha = scale_alpha
        self._scale_beta = scale_beta
        self._windows: list[tuple[float, float, float, float]] = []

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self._rng = np.random.default_rng(ctx.rng.spawn(1)[0])
        self._load_windows()

    def _load_windows(self) -> None:
        with open(self._csv_path, newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                try:
                    alpha = float(row["alpha"])
                    beta = float(row["beta"])
                    start = float(row["window_start_timestamp"])
                    end = float(row["window_end_timestamp"])
                except (ValueError, TypeError):
                    continue
                alpha *= self._scale_alpha
                beta *= self._scale_beta
                if math.isnan(alpha) or math.isnan(beta) or alpha <= 0 or beta <= 0:
                    continue
                self._windows.append((start, end, alpha, beta))

        self._windows.sort(key=lambda w: w[0])
        if self._windows:
            base = self._windows[0][0]
            self._windows = [(s - base, e - base, a, b) for s, e, a, b in self._windows]

        self.ctx.logger.info(
            "GammaWindowGenerator: loaded %d windows from %s",
            len(self._windows), self._csv_path,
        )

    def _find_window(self, now: float) -> tuple[float, float] | None:
        for start, end, alpha, beta in self._windows:
            if start <= now < end:
                return alpha, beta
        return None

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        self.ctx.env.process(self._arrival_loop(service, stop_time))

    def _arrival_loop(self, service: ServiceClass, stop_time: float | None = None):
        ctx = self.ctx
        rng = self._rng
        env = ctx.env

        while True:
            if stop_time is not None and env.now >= stop_time:
                return

            params = self._find_window(env.now)
            if params is None:
                next_start = None
                for start, end, alpha, beta in self._windows:
                    if start > env.now:
                        next_start = start
                        break
                if next_start is None:
                    return
                yield env.timeout(next_start - env.now)
                continue

            alpha, beta = params
            interval = rng.gamma(alpha, beta)
            if interval <= 0:
                continue

            yield env.timeout(interval)

            if stop_time is not None and env.now >= stop_time:
                return

            inv = self._make_invocation(ctx, service)
            ctx.logger.debug("t=%.3f | ARRIVE | %s service=%s", env.now, inv.request_id, service.service_id)
            self._dispatch(ctx, inv)


class PoissonFixedSizeGenerator(BaseGenerator):
    """Exponential inter-arrival times."""

    def __init__(self, arrival_rate: float = 1.0):
        self.ctx: SimContext | None = None
        self._rng = None
        self._arrival_rate = arrival_rate

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self._rng = np.random.default_rng(ctx.rng.spawn(1)[0])

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        self.ctx.env.process(self._arrival_loop(service, stop_time))

    def _arrival_loop(self, service: ServiceClass, stop_time: float | None = None):
        ctx = self.ctx
        rng = self._rng
        env = ctx.env
        mean_interval = 1.0 / self._arrival_rate

        while True:
            if stop_time is not None and env.now >= stop_time:
                return

            interval = rng.exponential(mean_interval)
            yield env.timeout(interval)

            if stop_time is not None and env.now >= stop_time:
                return

            inv = self._make_invocation(ctx, service)
            ctx.logger.debug("t=%.3f | ARRIVE | %s service=%s", env.now, inv.request_id, service.service_id)
            self._dispatch(ctx, inv)

class WeibullGenerator(BaseGenerator):
    """Weibull inter-arrival times."""

    def __init__(self, shape: float = 1.0, scale: float = 1.0, limit: int = 1000):
        self.ctx: SimContext | None = None
        self._rng = None
        self._shape = shape
        self._scale = scale
        self._limit = limit

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self._rng = np.random.default_rng(ctx.rng.spawn(1)[0])

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        self.ctx.env.process(self._arrival_loop(service, stop_time))

    def _arrival_loop(self, service: ServiceClass, stop_time: float | None = None):
        ctx = self.ctx
        rng = self._rng
        env = ctx.env
        intervals = rng.weibull(self._shape, self._limit) * self._scale
        count = 0

        while True:
            if stop_time is not None and env.now >= stop_time:
                return
            if count >= self._limit:
                return

            interval = intervals[count]
            count += 1

            yield env.timeout(interval)

            if stop_time is not None and env.now >= stop_time:
                return

            inv = self._make_invocation(ctx, service)
            ctx.logger.debug("t=%.3f | ARRIVE | %s service=%s", env.now, inv.request_id, service.service_id)
            self._dispatch(ctx, inv)