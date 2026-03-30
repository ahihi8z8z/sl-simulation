from __future__ import annotations

from typing import TYPE_CHECKING

from serverless_sim.export.batch_csv_writer import BatchCSVWriter
from serverless_sim.monitoring.metric_store import MetricStore
from serverless_sim.monitoring.collectors import (
    RequestCollector,
    ClusterCollector,
    LifecycleCollector,
    InterArrivalCollector,
    AutoscalingCollector,
)

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class MonitorManager:
    """Owns collectors and metric store, runs periodic collection."""

    def __init__(self, ctx: SimContext, interval: float = 1.0, max_history: int = 1000):
        self.ctx = ctx
        self.interval = interval
        self.store = MetricStore(max_history_length=max_history)
        self.collectors = [
            RequestCollector(),
            ClusterCollector(),
            LifecycleCollector(),
            InterArrivalCollector(),
            AutoscalingCollector(),
        ]
        self._stream_writer: BatchCSVWriter | None = None
        self._stream_metric_names: list[str] | None = None

    def enable_streaming(self, run_dir: str) -> None:
        """Enable streaming system_metrics.csv — writes rows as they are collected."""
        import os
        path = os.path.join(run_dir, "system_metrics.csv")
        # Header written lazily on first collect (metric names not known yet)
        self._stream_writer = BatchCSVWriter(path, [], buffer_size=100)
        self._stream_metric_names = None

    def close_streaming(self) -> None:
        """Flush and close the streaming writer."""
        if self._stream_writer is not None:
            self._stream_writer.close()
            self._stream_writer = None

    def start(self) -> None:
        """Start the periodic collection SimPy process."""
        self.ctx.env.process(self._periodic_loop())

    def _periodic_loop(self):
        while True:
            yield self.ctx.env.timeout(self.interval)
            self.collect_once()

    def collect_once(self) -> dict[str, float]:
        """Run all collectors and store results. Returns merged metrics."""
        t = self.ctx.env.now
        merged = {}
        for collector in self.collectors:
            metrics = collector.collect(t, self.ctx)
            for name, value in metrics.items():
                self.store.put(name, t, value)
                merged[name] = value

        # Stream to CSV
        if self._stream_writer is not None and merged:
            if self._stream_metric_names is None:
                # First collect — write header and open file
                self._stream_metric_names = sorted(merged.keys())
                self._stream_writer.header = ["time"] + self._stream_metric_names
                self._stream_writer.open()

            row = [f"{t:.3f}"]
            for name in self._stream_metric_names:
                val = merged.get(name, "")
                row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
            self._stream_writer.write_row(row)

        return merged
