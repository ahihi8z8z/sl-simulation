"""Memory-efficient request storage with streaming trace export.

Completed requests are flushed from memory after their counters and
latency are recorded.  An optional CSV trace writer streams rows as
requests finalise, so the full table never needs to be held in memory.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from serverless_sim.export.batch_csv_writer import BatchCSVWriter
from serverless_sim.export.request_trace_exporter import TRACE_HEADER

if TYPE_CHECKING:
    from serverless_sim.workload.invocation import Invocation

TERMINAL_STATUSES = {"completed", "timed_out", "dropped", "truncated"}


@dataclass
class RequestCounters:
    """Running counters — updated on finalize, never need to scan."""

    total: int = 0
    completed: int = 0
    dropped: int = 0
    timed_out: int = 0
    truncated: int = 0
    cold_starts: int = 0


class RequestStore:
    """Drop-in replacement for ``ctx.request_table`` (plain dict).

    Keeps only in-flight requests in memory.  Terminal requests are
    counted, their latencies recorded, and optionally streamed to a
    CSV trace file before being evicted from the dict.

    Backward-compatible API:
        store[request_id] = inv   # register
        len(store)                # total (including flushed)
        store.values()            # only in-flight invocations
    """

    def __init__(self):
        self._active: dict[str, Invocation] = {}
        self.counters = RequestCounters()
        self.latencies: list[float] = []  # kept sorted via bisect.insort
        self._latency_sum: float = 0.0
        self._trace_writer: BatchCSVWriter | None = None

    # ------------------------------------------------------------------
    # Trace writer lifecycle
    # ------------------------------------------------------------------

    def enable_trace(self, run_dir: str) -> None:
        """Open a streaming request_trace.csv writer."""
        path = os.path.join(run_dir, "request_trace.csv")
        self._trace_writer = BatchCSVWriter(path, TRACE_HEADER)
        self._trace_writer.open()

    def close_trace(self) -> None:
        """Flush and close the trace writer."""
        if self._trace_writer is not None:
            self._trace_writer.close()
            self._trace_writer = None

    # ------------------------------------------------------------------
    # Register / finalize
    # ------------------------------------------------------------------

    def register(self, inv: Invocation) -> None:
        """Add a new request (in-flight)."""
        self._active[inv.request_id] = inv
        self.counters.total += 1

    def finalize(self, inv: Invocation) -> None:
        """Record terminal state, stream to CSV, remove from memory."""
        status = inv.status
        if status == "completed":
            self.counters.completed += 1
            if inv.cold_start:
                self.counters.cold_starts += 1
            if inv.completion_time is not None and inv.arrival_time is not None:
                lat = inv.completion_time - inv.arrival_time
                bisect.insort(self.latencies, lat)
                self._latency_sum += lat
        elif status == "timed_out":
            self.counters.timed_out += 1
        elif status == "dropped":
            self.counters.dropped += 1
        elif status == "truncated":
            self.counters.truncated += 1

        if self._trace_writer is not None:
            self._write_trace_row(inv)

        self._active.pop(inv.request_id, None)

    # ------------------------------------------------------------------
    # Backward-compatible dict-like API
    # ------------------------------------------------------------------

    def __setitem__(self, key: str, inv: Invocation) -> None:
        """``store[request_id] = inv`` — same as register()."""
        self.register(inv)

    def __getitem__(self, key: str) -> Invocation:
        return self._active[key]

    def __contains__(self, key: str) -> bool:
        return key in self._active

    def __len__(self) -> int:
        """Total requests ever registered (including flushed)."""
        return self.counters.total

    def values(self):
        """Iterate over **in-flight** invocations only."""
        return self._active.values()

    def items(self):
        return self._active.items()

    @property
    def active_count(self) -> int:
        return len(self._active)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_trace_row(self, inv: Invocation) -> None:
        row = [
            inv.request_id,
            inv.service_id,
            self._fmt(inv.arrival_time),
            self._fmt(inv.dispatch_time),
            self._fmt(inv.queue_enter_time),
            self._fmt(inv.execution_start_time),
            self._fmt(inv.execution_end_time),
            self._fmt(inv.completion_time),
            inv.assigned_node_id or "",
            inv.assigned_instance_id or "",
            inv.cold_start,
            inv.dropped,
            inv.timed_out,
            inv.drop_reason or "",
            inv.status,
        ]
        self._trace_writer.write_row(row)

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:.6f}"
        return str(val)
