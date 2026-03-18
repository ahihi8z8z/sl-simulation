from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from serverless_sim.export.summary_writer import SummaryWriter
from serverless_sim.export.system_metrics_exporter import SystemMetricsExporter
from serverless_sim.export.request_trace_exporter import RequestTraceExporter

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class ExportManager:
    """Manages export mode (0/1/2) and coordinates exporters.

    Mode 0: summary.txt only
    Mode 1: summary.txt + system_metrics.csv
    Mode 2: summary.txt + system_metrics.csv + request_trace.csv
    """

    def __init__(self, ctx: SimContext, mode: int = 0):
        self.ctx = ctx
        self.mode = mode
        self.logger = ctx.logger

    def export(self) -> list[str]:
        """Run all exporters based on mode. Returns list of written file paths."""
        paths = []

        # Mode 0+: always write summary
        sw = SummaryWriter(self.ctx)
        p = sw.write()
        paths.append(p)
        self.logger.info("Wrote %s", p)

        if self.mode >= 1:
            sme = SystemMetricsExporter(self.ctx)
            p = sme.export()
            if p:
                paths.append(p)
                self.logger.info("Wrote %s", p)

        if self.mode >= 2:
            rte = RequestTraceExporter(self.ctx)
            p = rte.export()
            if p:
                paths.append(p)
                self.logger.info("Wrote %s", p)

        return paths
