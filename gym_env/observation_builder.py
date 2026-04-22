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
}

# Computed virtual metrics — derived from deltas of other metrics
# Format: "computed.<name>": (numerator_metric, denominator_metric, default)
COMPUTED_METRICS = {
    # cold_start_ratio = d(cold_starts) / d(total)
    "computed.cold_start_ratio": ("request.cold_starts", "request.total", 0.0),
    # drop_ratio = d(dropped) / d(total)
    "computed.drop_ratio": ("request.dropped", "request.total", 0.0),
}

# Time-averaged utilization over the step:
# d(total_{cpu,memory}_seconds) / (cluster.{cpu,memory}_capacity * step_duration).
# Handled separately because the denominator mixes an instantaneous
# capacity reading with a constant, not a cumulative delta.
UTILIZATION_STEP_METRICS = {
    "computed.memory_utilization_step": (
        "lifecycle.total_memory_seconds",
        "cluster.memory_capacity",
    ),
    "computed.cpu_utilization_step": (
        "lifecycle.total_cpu_seconds",
        "cluster.cpu_capacity",
    ),
}


class ObservationBuilder:
    """Converts monitor snapshot to fixed-size numpy vector.

    Features:
    - Cumulative metrics auto-converted to per-step deltas
    - Computed virtual metrics (ratios derived from deltas):
        computed.cold_start_ratio        = d(cold_starts) / d(total)
        computed.drop_ratio              = d(dropped) / d(total)
        computed.memory_utilization_step = d(total_mem_sec) / (cluster_mem * step_duration)
        computed.cpu_utilization_step    = d(total_cpu_sec) / (cluster_cpu * step_duration)
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
            elif name in COMPUTED_METRICS:
                spec = COMPUTED_METRICS[name]
                self._tracked_cumulative.add(spec[0])
                self._tracked_cumulative.add(spec[1])
            elif name in UTILIZATION_STEP_METRICS:
                num, _ = UTILIZATION_STEP_METRICS[name]
                self._tracked_cumulative.add(num)
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
            elif name in UTILIZATION_STEP_METRICS:
                num_key, cap_key = UTILIZATION_STEP_METRICS[name]
                d_num = deltas.get(num_key, 0.0)
                capacity = float(snapshot.get(cap_key, 0.0))
                max_possible = capacity * self.step_duration
                obs[i] = (d_num / max_possible) if max_possible > 0 else 0.0
            elif name in COMPUTED_METRICS:
                spec = COMPUTED_METRICS[name]
                num = deltas.get(spec[0], 0.0)
                den = deltas.get(spec[1], 0.0)
                obs[i] = (num / den) if den > 0 else spec[2]
            else:
                obs[i] = float(snapshot.get(name, 0.0))
        return obs
