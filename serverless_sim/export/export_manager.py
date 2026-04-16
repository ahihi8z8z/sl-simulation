from __future__ import annotations

import os
from typing import TYPE_CHECKING

from serverless_sim.export.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class ExportManager:
    """Manages export mode (0/1/2) and coordinates exporters.

    Mode 0: summary.json only
    Mode 1: summary.json + system_metrics.csv
    Mode 2: summary.json + system_metrics.csv + request_trace.csv (streamed)
    """

    def __init__(self, ctx: SimContext, mode: int = 0):
        self.ctx = ctx
        self.mode = mode
        self.logger = ctx.logger

        # Mode 1+: enable streaming system metrics
        if self.mode >= 1:
            ctx.monitor_manager.enable_streaming(ctx.run_dir)

        # Mode 2: enable streaming trace on the request store
        if self.mode >= 2:
            ctx.request_table.enable_trace(ctx.run_dir)

    def export(self, wall_clock_seconds: float | None = None) -> list[str]:
        """Run all exporters based on mode. Returns list of written file paths."""
        paths = []

        # Close streaming trace before writing summary (flush remaining rows)
        if self.mode >= 2:
            self.ctx.request_table.close_trace()
            trace_path = os.path.join(self.ctx.run_dir, "request_trace.csv")
            if os.path.exists(trace_path):
                paths.append(trace_path)
                self.logger.info("Wrote %s", trace_path)

        # Mode 0+: always write summary
        sw = SummaryWriter(self.ctx)
        p = sw.write(wall_clock_seconds=wall_clock_seconds)
        paths.append(p)
        self.logger.info("Wrote %s", p)

        if self.mode >= 1:
            # Close streaming writer (flushes remaining buffer)
            self.ctx.monitor_manager.close_streaming()
            metrics_path = os.path.join(self.ctx.run_dir, "system_metrics.csv")
            if os.path.exists(metrics_path):
                paths.append(metrics_path)
                self.logger.info("Wrote %s", metrics_path)

        return paths
