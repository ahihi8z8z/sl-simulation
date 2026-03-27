from __future__ import annotations

from typing import TYPE_CHECKING

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
        return merged
