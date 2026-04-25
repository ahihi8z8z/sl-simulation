"""Trace-replay workload generators.

Each generator instance replays a per-service trace file.  The CSV
contains only timing/count columns — service routing comes from the
service the generator is bound to via
``services[i].workload.trace_path``.

**TraceReplayGenerator** — per-request timestamps::

    timestamp
    0.001
    0.003

**AggregateTraceGenerator** — request counts per minute::

    minute,count
    0,5
    1,10

The aggregate generator treats each minute as a homogeneous Poisson
process with rate ``lambda = count / 60`` (requests per second), so
inter-arrival times are sampled from ``Exponential(1/lambda)``. The
realised count per minute therefore varies around ``count`` (Poisson
distributed).

Service time (execution duration) is NOT set by these generators —
it is handled by the service_time provider configured separately.
"""

from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from serverless_sim.workload.generators import BaseGenerator

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.workload.service_class import ServiceClass


MAX_TRACE_SIZE_MB = 500

DEFAULT_TRACE_COLUMNS = {"timestamp": "timestamp"}
DEFAULT_AGGREGATE_COLUMNS = {"minute": "minute", "count": "count"}


@dataclass
class TraceRecord:
    """One row from the trace CSV."""
    timestamp: float


class TraceReplayGenerator(BaseGenerator):
    """Replay requests from a pre-loaded trace file.

    The trace is loaded entirely into memory on construction.
    A warning is issued if the file exceeds ``MAX_TRACE_SIZE_MB``.

    Parameters
    ----------
    start_minute, end_minute : float | None
        If set, only replay records whose timestamp (in minutes) falls
        within [start_minute, end_minute). Timestamps are shifted so that
        start_minute becomes time 0 in the simulation.
    """

    def __init__(self, trace_path: str, start_minute: float | None = None,
                 end_minute: float | None = None,
                 column_map: dict[str, str] | None = None,
                 scale: int = 1):
        if not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale must be positive integer, got {scale!r}")
        self.ctx: SimContext | None = None
        self._start_minute = start_minute
        self._end_minute = end_minute
        self._scale = scale
        self._col = {**DEFAULT_TRACE_COLUMNS, **(column_map or {})}
        self._records: list[TraceRecord] = []
        self._load_trace(trace_path)

    def _load_trace(self, path: str) -> None:
        """Load and sort trace records by timestamp."""
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > MAX_TRACE_SIZE_MB:
            warnings.warn(
                f"Trace file {path} is {file_size_mb:.0f} MB "
                f"(>{MAX_TRACE_SIZE_MB} MB). This may use significant memory.",
                stacklevel=2,
            )

        c = self._col
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._records.append(TraceRecord(timestamp=float(row[c["timestamp"]])))

        self._records.sort(key=lambda r: r.timestamp)
        self._apply_time_range()

    def _apply_time_range(self) -> None:
        """Filter and shift records based on start_minute/end_minute."""
        start_sec = self._start_minute * 60.0 if self._start_minute is not None else None
        end_sec = self._end_minute * 60.0 if self._end_minute is not None else None

        if start_sec is not None or end_sec is not None:
            filtered = []
            for r in self._records:
                if start_sec is not None and r.timestamp < start_sec:
                    continue
                if end_sec is not None and r.timestamp >= end_sec:
                    continue
                filtered.append(r)
            offset = start_sec if start_sec is not None else 0.0
            for r in filtered:
                r.timestamp -= offset
            self._records = filtered

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self._rng = np.random.default_rng(ctx.rng.spawn(1)[0])

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        """Start replay for the bound service."""
        if self._records:
            self.ctx.env.process(self._replay_loop(service, stop_time))

    def _replay_loop(
        self,
        service: ServiceClass,
        stop_time: float | None,
    ):
        """SimPy process: emit requests at trace timestamps."""
        ctx = self.ctx
        env = ctx.env

        for record in self._records:
            if record.timestamp > env.now:
                yield env.timeout(record.timestamp - env.now)

            if stop_time is not None and env.now >= stop_time:
                ctx.logger.debug(
                    "t=%.3f | TRACE_STOP | %s (stop_time=%.1f)",
                    env.now, service.service_id, stop_time,
                )
                return

            for _ in range(self._scale):
                inv = self._make_invocation(ctx, service)
                ctx.logger.debug(
                    "t=%.3f | TRACE_ARRIVE | %s service=%s",
                    env.now, inv.request_id, service.service_id,
                )
                self._dispatch(ctx, inv)

    @property
    def record_count(self) -> int:
        """Total number of records loaded."""
        return len(self._records)


# ------------------------------------------------------------------
# Aggregate trace (counts per minute)
# ------------------------------------------------------------------

@dataclass
class AggregateRecord:
    """One row from the aggregate trace CSV."""
    minute: int
    count: float


class AggregateTraceGenerator(BaseGenerator):
    """Replay requests from an aggregate trace (counts per minute).

    CSV format::

        minute,count
        0,5
        1,10

    For each row, requests are sampled as a homogeneous Poisson process
    over that minute with rate ``lambda = count * scale / 60`` (req/s).
    Inter-arrival times come from ``Exponential(1/lambda)`` using a
    per-generator RNG spawned from ``ctx.rng`` (reproducible across
    runs with the same simulation seed).

    Parameters
    ----------
    start_minute, end_minute : int | None
        If set, only replay records whose minute falls within
        [start_minute, end_minute). Minutes are shifted so that
        start_minute becomes minute 0 in the simulation.
    """

    MINUTE_SECONDS = 60.0

    def __init__(self, trace_path: str, scale: float = 1.0,
                 start_minute: int | None = None, end_minute: int | None = None,
                 column_map: dict[str, str] | None = None):
        self.ctx: SimContext | None = None
        self._scale = scale
        self._start_minute = start_minute
        self._end_minute = end_minute
        self._col = {**DEFAULT_AGGREGATE_COLUMNS, **(column_map or {})}
        self._records: list[AggregateRecord] = []
        self._load_trace(trace_path)

    def _load_trace(self, path: str) -> None:
        """Load aggregate trace and sort by minute."""
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > MAX_TRACE_SIZE_MB:
            warnings.warn(
                f"Trace file {path} is {file_size_mb:.0f} MB "
                f"(>{MAX_TRACE_SIZE_MB} MB). This may use significant memory.",
                stacklevel=2,
            )

        c = self._col
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                count_raw = row[c["count"]]
                if not count_raw or count_raw == "":
                    continue
                count_val = float(count_raw)
                if count_val <= 0:
                    continue
                self._records.append(AggregateRecord(
                    minute=int(float(row[c["minute"]])),
                    count=count_val,
                ))

        self._records.sort(key=lambda r: r.minute)
        self._apply_time_range()

    def _apply_time_range(self) -> None:
        """Filter and shift records based on start_minute/end_minute."""
        if self._start_minute is not None or self._end_minute is not None:
            filtered = []
            for r in self._records:
                if self._start_minute is not None and r.minute < self._start_minute:
                    continue
                if self._end_minute is not None and r.minute >= self._end_minute:
                    continue
                filtered.append(r)
            offset = self._start_minute if self._start_minute is not None else 0
            for r in filtered:
                r.minute -= offset
            self._records = filtered

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx
        self._rng = np.random.default_rng(ctx.rng.spawn(1)[0])

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        """Start replay for the bound service."""
        if self._records:
            self.ctx.env.process(self._replay_loop(service, stop_time))

    def _replay_loop(
        self,
        service: ServiceClass,
        stop_time: float | None,
    ):
        """SimPy process: sample arrivals as a Poisson process per minute.

        For each record, the arrival rate is ``lambda = count * scale / 60``
        req/s. Inter-arrival times are drawn from ``Exponential(1/lambda)``
        using ``self._rng`` (spawned from ``ctx.rng`` at attach time, so
        independent of other modules' RNG draws).

        At each minute boundary the inter-arrival sampling restarts. By
        the memoryless property of the exponential distribution this is
        statistically equivalent to a piecewise-constant-rate Poisson
        process — the residual time within a minute can be discarded.
        """
        ctx = self.ctx
        env = ctx.env
        rng = self._rng

        for record in self._records:
            rate = record.count * self._scale / self.MINUTE_SECONDS
            if rate <= 0.0:
                continue

            minute_start = record.minute * self.MINUTE_SECONDS
            mean_interval = 1.0 / rate
            t_local = 0.0
            idx = 0

            while True:
                t_local += float(rng.exponential(mean_interval))
                if t_local >= self.MINUTE_SECONDS:
                    break

                target_time = minute_start + t_local
                if target_time > env.now:
                    yield env.timeout(target_time - env.now)

                if stop_time is not None and env.now >= stop_time:
                    ctx.logger.debug(
                        "t=%.3f | AGG_TRACE_STOP | %s (stop_time=%.1f)",
                        env.now, service.service_id, stop_time,
                    )
                    return

                inv = self._make_invocation(ctx, service)

                ctx.logger.debug(
                    "t=%.3f | AGG_ARRIVE | %s service=%s (minute=%d, idx=%d, lambda=%.4f)",
                    env.now, inv.request_id, service.service_id,
                    record.minute, idx, rate,
                )

                self._dispatch(ctx, inv)
                idx += 1

    @property
    def record_count(self) -> int:
        """Total number of aggregate records loaded."""
        return len(self._records)
