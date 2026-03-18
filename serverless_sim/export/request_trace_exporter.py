from __future__ import annotations

import os
from typing import TYPE_CHECKING

from serverless_sim.export.batch_csv_writer import BatchCSVWriter

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


TRACE_HEADER = [
    "request_id",
    "service_id",
    "arrival_time",
    "dispatch_time",
    "queue_enter_time",
    "execution_start_time",
    "execution_end_time",
    "completion_time",
    "assigned_node_id",
    "assigned_instance_id",
    "cold_start",
    "dropped",
    "timed_out",
    "drop_reason",
    "status",
]


class RequestTraceExporter:
    """Exports request_trace.csv from the request table."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    def export(self) -> str:
        """Write request_trace.csv and return the file path."""
        path = os.path.join(self.ctx.run_dir, "request_trace.csv")
        writer = BatchCSVWriter(path, TRACE_HEADER)
        writer.open()

        for inv in self.ctx.request_table.values():
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
            writer.write_row(row)

        writer.close()
        return path

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:.6f}"
        return str(val)
