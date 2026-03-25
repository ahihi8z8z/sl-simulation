"""Trace-replay workload generator.

Reads a CSV trace file and replays requests according to timestamps.

CSV format::

    timestamp,function_id,duration,memory
    0.001,func-a,0.150,256
    0.003,func-b,2.000,512
    0.005,func-a,0.145,256

Columns:
- ``timestamp``: arrival time in simulation seconds (float, required)
- ``function_id``: maps to service_id (required)
- ``duration``: execution duration in seconds (optional, sets ``service_time``)
- ``memory``: memory in MB (optional, informational)

The generator creates ``Invocation`` objects at the specified timestamps.
If ``duration`` is present, it is set as ``invocation.service_time`` for
use with ``PrecomputedServingModel``.
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
    """

    def __init__(self, trace_path: str):
        self.ctx: SimContext | None = None
        self._request_counter = 0
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

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = TraceRecord(
                    timestamp=float(row["timestamp"]),
                    function_id=row["function_id"],
                    duration=float(row["duration"]) if "duration" in row and row["duration"] else None,
                    memory=float(row["memory"]) if "memory" in row and row["memory"] else None,
                )
                self._records.append(record)

        self._records.sort(key=lambda r: r.timestamp)

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
