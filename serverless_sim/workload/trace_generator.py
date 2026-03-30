"""Trace-replay workload generators.

Two generators are available:

**TraceReplayGenerator** — per-request timestamps::

    timestamp,function_id,duration,memory
    0.001,func-a,0.150,256
    0.003,func-b,2.000,512

**AggregateTraceGenerator** — request counts per minute::

    minute,function_id,count,duration
    0,func-a,5,0.15
    1,func-a,10,0.15
    1,func-b,3,2.0

The aggregate generator distributes ``count`` requests evenly within
each minute (interval = 60 / count).
"""

from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from serverless_sim.workload.generators import BaseGenerator
from serverless_sim.workload.invocation import Invocation

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.workload.service_class import ServiceClass


MAX_TRACE_SIZE_MB = 500

# Default column names expected in CSVs
DEFAULT_TRACE_COLUMNS = {
    "timestamp": "timestamp",
    "function_id": "function_id",
    "duration": "duration",
    "memory": "memory",
}
DEFAULT_AGGREGATE_COLUMNS = {
    "minute": "minute",
    "function_id": "function_id",
    "count": "count",
    "duration": "duration",
}


@dataclass
class TraceRecord:
    """One row from the trace CSV."""
    timestamp: float
    function_id: str
    duration: float | None = None
    memory: float | None = None


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
                 column_map: dict[str, str] | None = None):
        self.ctx: SimContext | None = None
        self._request_counter = 0
        self._start_minute = start_minute
        self._end_minute = end_minute
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
                dur_col = c["duration"]
                mem_col = c["memory"]
                record = TraceRecord(
                    timestamp=float(row[c["timestamp"]]),
                    function_id=row[c["function_id"]],
                    duration=float(row[dur_col]) if dur_col in row and row[dur_col] else None,
                    memory=float(row[mem_col]) if mem_col in row and row[mem_col] else None,
                )
                self._records.append(record)

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
            # Shift timestamps so start_minute becomes time 0
            offset = start_sec if start_sec is not None else 0.0
            for r in filtered:
                r.timestamp -= offset
            self._records = filtered

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        """Start replay for a specific service (filters records by function_id)."""
        records = [r for r in self._records if r.function_id == service.service_id]
        if records:
            self.ctx.env.process(self._replay_loop(service, records, stop_time))

    def start_all(self, stop_time: float | None = None) -> None:
        """Start replay for ALL function_ids in the trace (no service filter).

        Use this when trace contains services not declared in config.
        Each unique function_id gets its own replay process.
        """
        by_function: dict[str, list[TraceRecord]] = {}
        for r in self._records:
            by_function.setdefault(r.function_id, []).append(r)

        for function_id, records in by_function.items():
            service = self.ctx.workload_manager.services.get(function_id)
            if service is None:
                self.ctx.logger.debug(
                    "Trace function_id '%s' not in config, skipping %d records",
                    function_id, len(records),
                )
                continue
            self.ctx.env.process(self._replay_loop(service, records, stop_time))

    def _replay_loop(
        self,
        service: ServiceClass,
        records: list[TraceRecord],
        stop_time: float | None,
    ):
        """SimPy process: emit requests at trace timestamps."""
        ctx = self.ctx
        env = ctx.env

        for record in records:
            # Wait until the record's timestamp
            if record.timestamp > env.now:
                yield env.timeout(record.timestamp - env.now)

            if stop_time is not None and env.now >= stop_time:
                ctx.logger.debug(
                    "t=%.3f | TRACE_STOP | %s (stop_time=%.1f)",
                    env.now, service.service_id, stop_time,
                )
                return

            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

            inv = Invocation(
                request_id=request_id,
                service_id=service.service_id,
                arrival_time=env.now,
                job_size=record.duration if record.duration is not None else service.job_size,
                service_time=record.duration,
                status="arrived",
            )

            ctx.request_table[request_id] = inv

            ctx.logger.debug(
                "t=%.3f | TRACE_ARRIVE | %s service=%s duration=%s",
                env.now, request_id, service.service_id,
                f"{record.duration:.3f}" if record.duration else "N/A",
            )

            if ctx.dispatcher is not None:
                ctx.dispatcher.dispatch(inv)

    @property
    def record_count(self) -> int:
        """Total number of records loaded."""
        return len(self._records)

    @property
    def function_ids(self) -> set[str]:
        """Unique function_ids in the trace."""
        return {r.function_id for r in self._records}


# ------------------------------------------------------------------
# Aggregate trace (counts per minute)
# ------------------------------------------------------------------

@dataclass
class AggregateRecord:
    """One row from the aggregate trace CSV."""
    minute: int
    function_id: str
    count: float
    duration: float | None = None


class AggregateTraceGenerator(BaseGenerator):
    """Replay requests from an aggregate trace (counts per minute).

    CSV format::

        minute,function_id,count,duration
        0,func-a,5,0.15
        1,func-a,10,0.15
        1,func-b,3,2.0

    For each row, ``count`` requests are distributed evenly within that
    minute (interval = 60 / count).  If ``duration`` is present, it is
    set as ``invocation.service_time`` for ``PrecomputedServingModel``.

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
        self._request_counter = 0
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
                dur_col = c["duration"]
                count_raw = row[c["count"]]
                if not count_raw or count_raw == "":
                    continue
                count_val = float(count_raw)
                if count_val <= 0:
                    continue
                record = AggregateRecord(
                    minute=int(float(row[c["minute"]])),
                    function_id=row[c["function_id"]],
                    count=count_val,
                    duration=float(row[dur_col]) if dur_col in row and row[dur_col] else None,
                )
                if record.count > 0:
                    self._records.append(record)

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
            # Shift minutes so start_minute becomes minute 0
            offset = self._start_minute if self._start_minute is not None else 0
            for r in filtered:
                r.minute -= offset
            self._records = filtered

    def attach(self, ctx: SimContext) -> None:
        self.ctx = ctx

    def start_for_service(self, service: ServiceClass, stop_time: float | None = None) -> None:
        """Start replay for a specific service."""
        records = [r for r in self._records if r.function_id == service.service_id]
        if records:
            self.ctx.env.process(self._replay_loop(service, records, stop_time))

    def start_all(self, stop_time: float | None = None) -> None:
        """Start replay for ALL function_ids in the trace."""
        by_function: dict[str, list[AggregateRecord]] = {}
        for r in self._records:
            by_function.setdefault(r.function_id, []).append(r)

        for function_id, records in by_function.items():
            service = self.ctx.workload_manager.services.get(function_id)
            if service is None:
                self.ctx.logger.debug(
                    "Aggregate trace function_id '%s' not in config, skipping",
                    function_id,
                )
                continue
            self.ctx.env.process(self._replay_loop(service, records, stop_time))

    def _replay_loop(
        self,
        service: ServiceClass,
        records: list[AggregateRecord],
        stop_time: float | None,
    ):
        """SimPy process: distribute requests evenly within each minute.

        Uses carry-over accumulation for fractional counts so that
        long-run totals are preserved.  E.g. four consecutive minutes
        with count=0.3 produce 0, 0, 0, 1 requests respectively.
        """
        ctx = self.ctx
        env = ctx.env
        remainder = 0.0

        for record in records:
            minute_start = record.minute * self.MINUTE_SECONDS

            # Accumulate fractional count with carry-over
            # Clamp to 0 to prevent negative values from eating remainder
            remainder += max(0.0, record.count * self._scale)
            emit_count = int(remainder + 1e-9)
            remainder -= emit_count

            if emit_count <= 0:
                continue

            interval = self.MINUTE_SECONDS / emit_count

            for i in range(emit_count):
                target_time = minute_start + i * interval

                if target_time > env.now:
                    yield env.timeout(target_time - env.now)

                if stop_time is not None and env.now >= stop_time:
                    ctx.logger.debug(
                        "t=%.3f | AGG_TRACE_STOP | %s (stop_time=%.1f)",
                        env.now, service.service_id, stop_time,
                    )
                    return

                self._request_counter += 1
                request_id = f"req-{self._request_counter}"

                inv = Invocation(
                    request_id=request_id,
                    service_id=service.service_id,
                    arrival_time=env.now,
                    job_size=record.duration if record.duration is not None else service.job_size,
                    service_time=record.duration,
                    status="arrived",
                )

                ctx.request_table[request_id] = inv

                ctx.logger.debug(
                    "t=%.3f | AGG_ARRIVE | %s service=%s (minute=%d, %d/%d)",
                    env.now, request_id, service.service_id,
                    record.minute, i + 1, emit_count,
                )

                if ctx.dispatcher is not None:
                    ctx.dispatcher.dispatch(inv)

    @property
    def record_count(self) -> int:
        """Total number of aggregate records loaded."""
        return len(self._records)

    @property
    def total_requests(self) -> int:
        """Total requests that will be generated (after scaling, with carry-over)."""
        remainder = 0.0
        total = 0
        for r in self._records:
            remainder += max(0.0, r.count * self._scale)
            emit = int(remainder + 1e-9)
            remainder -= emit
            total += emit
        return total

    @property
    def function_ids(self) -> set[str]:
        """Unique function_ids in the trace."""
        return {r.function_id for r in self._records}
