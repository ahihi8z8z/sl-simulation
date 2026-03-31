from __future__ import annotations

import numpy as np

# Fixed set of metrics for the observation vector (order matters!)
DEFAULT_OBS_METRICS = [
    "cluster.cpu_utilization",
    "cluster.memory_utilization",
    "request.completed",
    "request.dropped",
    "request.in_flight",
    "request.cold_starts",
    "lifecycle.instances_total",
    "lifecycle.instances_warm",
    "lifecycle.instances_running",
    "lifecycle.instances_prewarm",
]

# Metrics that are cumulative counters — auto-converted to delta
CUMULATIVE_METRICS = {
    "request.completed",
    "request.dropped",
    "request.cold_starts",
    "request.total",
    "request.truncated",
    "lifecycle.instances_evicted",
    "lifecycle.total_cpu_seconds",
    "lifecycle.total_memory_seconds",
    "lifecycle.running_cpu_seconds",
    "lifecycle.running_memory_seconds",
}

# Computed virtual metrics — derived from deltas of other metrics
# Format: "computed.<name>": (numerator_metric, denominator_metric, default)
COMPUTED_METRICS = {
    # cold_start_ratio = d(cold_starts) / d(total)
    "computed.cold_start_ratio": ("request.cold_starts", "request.total", 0.0),
    # drop_ratio = d(dropped) / d(total)
    "computed.drop_ratio": ("request.dropped", "request.total", 0.0),
    # memory_efficiency = d(running_memory_seconds) / d(total_memory_seconds)
    "computed.memory_efficiency": (
        "lifecycle.running_memory_seconds",
        "lifecycle.total_memory_seconds",
        0.0,
    ),
    # cpu_efficiency = d(running_cpu_seconds) / d(total_cpu_seconds)
    "computed.cpu_efficiency": (
        "lifecycle.running_cpu_seconds",
        "lifecycle.total_cpu_seconds",
        0.0,
    ),
}


class ObservationBuilder:
    """Converts monitor snapshot to fixed-size numpy vector.

    Features:
    - Cumulative metrics auto-converted to per-step deltas
    - Computed virtual metrics (ratios derived from deltas):
        computed.cold_start_ratio  = d(cold_starts) / d(total)
        computed.drop_ratio        = d(dropped) / d(total)
        computed.memory_efficiency = d(running_mem_sec) / d(total_mem_sec)
        computed.cpu_efficiency    = d(running_cpu_sec) / d(total_cpu_sec)
    """

    def __init__(self, metric_names: list[str] | None = None,
                 step_duration: float = 5.0):
        self.metric_names = metric_names or DEFAULT_OBS_METRICS
        self.step_duration = step_duration
        self.obs_size = len(self.metric_names)
        self._prev: dict[str, float] = {}

        # Collect all cumulative metrics we need to track
        # (both directly requested and those needed for computed metrics)
        self._tracked_cumulative: set[str] = set()
        for name in self.metric_names:
            if name in CUMULATIVE_METRICS:
                self._tracked_cumulative.add(name)
            elif name.startswith("computed."):
                spec = COMPUTED_METRICS.get(name)
                if spec:
                    self._tracked_cumulative.add(spec[0])
                    self._tracked_cumulative.add(spec[1])
            elif name in ("computed.avg_inter_arrival_time", "computed.request_rate"):
                self._tracked_cumulative.add("request.total")

    def reset(self) -> None:
        """Reset delta tracking (call on env reset)."""
        self._prev.clear()

    def build(self, snapshot: dict) -> np.ndarray:
        """Build observation vector from a metric snapshot."""
        # Compute all deltas first
        deltas: dict[str, float] = {}
        for name in self._tracked_cumulative:
            raw = float(snapshot.get(name, 0.0))
            prev = self._prev.get(name, 0.0)
            deltas[name] = raw - prev
            self._prev[name] = raw

        obs = np.zeros(self.obs_size, dtype=np.float32)
        for i, name in enumerate(self.metric_names):
            if name in CUMULATIVE_METRICS:
                obs[i] = deltas.get(name, 0.0)
            elif name == "computed.avg_inter_arrival_time":
                d_total = deltas.get("request.total", 0.0)
                obs[i] = self.step_duration / max(d_total, 1.0)
            elif name == "computed.request_rate":
                d_total = deltas.get("request.total", 0.0)
                obs[i] = d_total / self.step_duration
            elif name.startswith("computed."):
                spec = COMPUTED_METRICS.get(name)
                if spec:
                    num = deltas.get(spec[0], 0.0)
                    den = deltas.get(spec[1], 0.0)
                    obs[i] = (num / den) if den > 0 else spec[2]
                else:
                    obs[i] = 0.0
            else:
                obs[i] = float(snapshot.get(name, 0.0))
        return obs
