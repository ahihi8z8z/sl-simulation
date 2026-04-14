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


class GammaWindowGenerator(BaseGenerator):
    """Gamma-distributed inter-arrival times with time-varying parameters.

    Reads a gamma windows CSV (from fit_trace_gamma_windows.py) where each
    row defines alpha/beta for a time window. Parameters change as simulation
    time crosses window boundaries. Windows with NaN alpha/beta are skipped
    (no requests generated in that window).

    CSV format::

        window_index,window_seconds,window_start_timestamp,window_end_timestamp,alpha,beta
        0,1200.0,0.0,1200.0,1.85,28.84
        1,1200.0,1200.0,2400.0,3.54,41.0
    """

    def __init__(self, csv_path: str, scale_alpha: float = 1.0, scale_beta: float = 1.0):
        self.ctx: SimContext | None = None
        self._request_counter = 0
        self._csv_path = csv_path
        self._scale_alpha = scale_alpha
        self._scale_beta = scale_beta
        self._windows: list[tuple[float, float, float, float]] = []  # (start, end, alpha, beta)

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self._load_windows()

    def _load_windows(self) -> None:
        import csv as csv_mod
        import math

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

        # Sort by start time and convert to relative time (offset from first window)
        self._windows.sort(key=lambda w: w[0])
        if self._windows:
            base = self._windows[0][0]
            self._windows = [(s - base, e - base, a, b) for s, e, a, b in self._windows]

        self.ctx.logger.info(
            "GammaWindowGenerator: loaded %d windows from %s",
            len(self._windows), self._csv_path,
        )

    def _find_window(self, now: float) -> tuple[float, float] | None:
        """Return (alpha, beta) for the window containing `now`, or None if no window."""
        for start, end, alpha, beta in self._windows:
            if start <= now < end:
                return alpha, beta
        return None

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        self.ctx.env.process(self._arrival_loop(service, stop_time))

    def _arrival_loop(self, service: ServiceClass, stop_time: float | None = None):
        ctx = self.ctx
        rng = ctx.rng
        env = ctx.env

        while True:
            if stop_time is not None and env.now >= stop_time:
                return

            params = self._find_window(env.now)
            if params is None:
                # No window covers current time — skip to next window or stop
                next_start = None
                for start, end, alpha, beta in self._windows:
                    if start > env.now:
                        next_start = start
                        break
                if next_start is None:
                    return  # past all windows
                yield env.timeout(next_start - env.now)
                continue

            alpha, beta = params
            interval = rng.gamma(alpha, beta)
            if interval <= 0:
                continue

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
