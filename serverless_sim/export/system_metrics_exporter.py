from __future__ import annotations

import os
from typing import TYPE_CHECKING

from serverless_sim.export.batch_csv_writer import BatchCSVWriter

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class SystemMetricsExporter:
    """Exports system_metrics.csv from MonitorManager's metric store."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    def export(self) -> str:
        """Write system_metrics.csv and return the file path."""
        store = self.ctx.monitor_manager.store
        metric_names = sorted(store.get_all_metric_names())

        if not metric_names:
            return ""

        # Collect all unique timestamps
        all_timestamps = set()
        for name in metric_names:
            for t, _ in store.get_all_entries(name):
                all_timestamps.add(t)
        timestamps = sorted(all_timestamps)

        if not timestamps:
            return ""

        # Build lookup: (metric, time) → value
        lookup = {}
        for name in metric_names:
            for t, v in store.get_all_entries(name):
                lookup[(name, t)] = v

        header = ["time"] + metric_names
        path = os.path.join(self.ctx.run_dir, "system_metrics.csv")
        writer = BatchCSVWriter(path, header)
        writer.open()

        for t in timestamps:
            row = [f"{t:.3f}"]
            for name in metric_names:
                val = lookup.get((name, t), "")
                row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
            writer.write_row(row)

        writer.close()
        return path
