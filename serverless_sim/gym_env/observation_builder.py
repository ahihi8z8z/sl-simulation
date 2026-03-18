from __future__ import annotations

import numpy as np

# Fixed set of metrics for the observation vector (order matters!)
DEFAULT_OBS_METRICS = [
    "cluster.cpu_utilization",
    "cluster.memory_utilization",
    "request.completed",
    "request.dropped",
    "request.timed_out",
    "request.in_flight",
    "request.cold_starts",
    "lifecycle.instances_total",
    "lifecycle.instances_warm",
    "lifecycle.instances_running",
    "lifecycle.instances_prewarm",
]


class ObservationBuilder:
    """Converts monitor snapshot to fixed-size numpy vector."""

    def __init__(self, metric_names: list[str] | None = None):
        self.metric_names = metric_names or DEFAULT_OBS_METRICS
        self.obs_size = len(self.metric_names)

    def build(self, snapshot: dict) -> np.ndarray:
        """Build observation vector from a metric snapshot."""
        obs = np.zeros(self.obs_size, dtype=np.float32)
        for i, name in enumerate(self.metric_names):
            obs[i] = float(snapshot.get(name, 0.0))
        return obs
