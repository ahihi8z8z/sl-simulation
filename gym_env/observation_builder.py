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

# Metrics that are cumulative counters — auto-converted to delta.
# Names containing `*` are templates: `*` is replaced by service id when
# such a metric (or one referencing it via a computed spec) is requested.
CUMULATIVE_METRICS = {
    "request.completed",
    "request.dropped",
    "request.cold_starts",
    "request.total",
    "request.truncated",
    "request.*.completed",
    "request.*.dropped",
    "request.*.cold_starts",
    "request.*.total",
    "request.*.truncated",
    "lifecycle.instances_evicted",
    "lifecycle.total_cpu_seconds",
    "lifecycle.total_memory_seconds",
    "lifecycle.*.total_cpu_seconds",
    "lifecycle.*.total_memory_seconds",
    "lifecycle.*.running_cpu_seconds",
    "lifecycle.*.running_memory_seconds",
}

# Computed virtual metrics — derived from deltas of other metrics.
# `*` in key/spec is a per-service template, expanded at __init__.
# Format: "computed.<name>": (numerator_metric, denominator_metric, default)
COMPUTED_METRICS = {
    "computed.cold_start_ratio": ("request.cold_starts", "request.total", 0.0),
    "computed.drop_ratio": ("request.dropped", "request.total", 0.0),
    "computed.*.cold_start_ratio": ("request.*.cold_starts", "request.*.total", 0.0),
    "computed.*.drop_ratio": ("request.*.dropped", "request.*.total", 0.0),
    # Step latency: avg latency over requests completed in the last step
    # = d(latency_sum) / d(completed)
    "computed.latency_mean_step": ("request.latency_sum", "request.completed", 0.0),
    "computed.*.latency_mean_step": ("request.*.latency_sum", "request.*.completed", 0.0),
    # Per-pod utilization (DRe-SCale style): running / allocated, per service.
    "computed.*.cpu_util_per_pod_step": (
        "lifecycle.*.running_cpu_seconds", "lifecycle.*.total_cpu_seconds", 0.0,
    ),
    "computed.*.mem_util_per_pod_step": (
        "lifecycle.*.running_memory_seconds", "lifecycle.*.total_memory_seconds", 0.0,
    ),
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
    # Per-service footprint vs cluster capacity (sum across services = global)
    "computed.*.memory_utilization_step": (
        "lifecycle.*.total_memory_seconds",
        "cluster.memory_capacity",
    ),
    "computed.*.cpu_utilization_step": (
        "lifecycle.*.total_cpu_seconds",
        "cluster.cpu_capacity",
    ),
}

# Special computed metrics whose handling is hardcoded in build() — declared
# here so wildcard expansion + cumulative tracking work uniformly.
# Each entry: template_name -> tracked_cumulative_template
SPECIAL_COMPUTED = {
    "computed.avg_inter_arrival_time": "request.total",
    "computed.request_rate": "request.total",
    "computed.*.avg_inter_arrival_time": "request.*.total",
    "computed.*.request_rate": "request.*.total",
}


def _expand_wildcards(metric_names: list[str], service_ids: list[str] | None) -> list[str]:
    """Expand `*` in metric names to concrete service ids (sorted, deterministic).

    Strict — every error case raises ValueError.
    """
    expanded: list[str] = []
    seen: set[str] = set()
    sorted_svcs = sorted(service_ids) if service_ids else []

    for name in metric_names:
        n_stars = name.count("*")
        if n_stars == 0:
            if name in seen:
                raise ValueError(f"duplicate observation metric: '{name}'")
            seen.add(name)
            expanded.append(name)
            continue
        if n_stars > 1:
            raise ValueError(f"metric '{name}' must contain at most one '*'")
        # n_stars == 1
        if service_ids is None:
            raise ValueError(
                f"ObservationBuilder needs service_ids when wildcards are used "
                f"(metric '{name}')"
            )
        if not sorted_svcs:
            raise ValueError(
                f"wildcard metric '{name}' requires at least one service"
            )
        for svc in sorted_svcs:
            concrete = name.replace("*", svc)
            if concrete in seen:
                raise ValueError(f"duplicate observation metric: '{concrete}'")
            seen.add(concrete)
            expanded.append(concrete)
    return expanded


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
                 step_duration: float = 5.0,
                 service_ids: list[str] | None = None):
        raw_names = metric_names or DEFAULT_OBS_METRICS
        self.metric_names = _expand_wildcards(raw_names, service_ids)
        self.step_duration = step_duration
        self.obs_size = len(self.metric_names)
        self._prev: dict[str, float] = {}

        # Resolve each requested name against the templates in CUMULATIVE_METRICS,
        # COMPUTED_METRICS, UTILIZATION_STEP_METRICS, SPECIAL_COMPUTED — taking
        # the per-service template variant when the name has a service segment.
        self._cumulative: set[str] = set()                    # leaf cumulative names
        self._computed: dict[str, tuple[str, str, float]] = {}
        self._utilization: dict[str, tuple[str, str]] = {}
        self._special: dict[str, tuple[str, str]] = {}        # name -> (kind, dep)
        self._tracked_cumulative: set[str] = set()

        for name in self.metric_names:
            if self._match_set(name, CUMULATIVE_METRICS):
                self._cumulative.add(name)
                self._tracked_cumulative.add(name)
                continue
            spec = self._match_dict(name, COMPUTED_METRICS)
            if spec is not None:
                num, den, default = spec
                self._computed[name] = (num, den, default)
                self._tracked_cumulative.add(num)
                self._tracked_cumulative.add(den)
                continue
            util = self._match_dict(name, UTILIZATION_STEP_METRICS)
            if util is not None:
                num, cap = util
                self._utilization[name] = (num, cap)
                self._tracked_cumulative.add(num)
                continue
            special_dep = self._match_dict(name, SPECIAL_COMPUTED)
            if special_dep is not None:
                # Resolve the kind by stripping the (optional) service segment
                kind = name.rsplit(".", 1)[-1]   # "request_rate" or "avg_inter_arrival_time"
                self._special[name] = (kind, special_dep)
                self._tracked_cumulative.add(special_dep)
                continue
            # else: treat as raw snapshot read in build()

    @staticmethod
    def _template_match(name: str, template: str) -> dict[str, str] | None:
        """Return {} if name == template, {svc: <svc>} if template has `*` and
        name fits the pattern (single segment in place of `*`), else None."""
        if "*" not in template:
            return {} if name == template else None
        # Split on `*` once — at most one star per template
        before, after = template.split("*", 1)
        if not name.startswith(before) or not name.endswith(after):
            return None
        svc = name[len(before):len(name) - len(after)] if after else name[len(before):]
        if not svc or "." in svc:
            return None  # require a single, non-empty segment
        return {"svc": svc}

    @classmethod
    def _match_set(cls, name: str, templates: set[str]) -> bool:
        return any(cls._template_match(name, t) is not None for t in templates)

    @classmethod
    def _match_dict(cls, name: str, templates: dict):
        """Return the spec value with `*` substituted, or None."""
        for tmpl, spec in templates.items():
            m = cls._template_match(name, tmpl)
            if m is None:
                continue
            svc = m.get("svc")
            if svc is None:
                return spec
            # Substitute `*` in spec strings
            if isinstance(spec, tuple):
                return tuple(s.replace("*", svc) if isinstance(s, str) else s for s in spec)
            if isinstance(spec, str):
                return spec.replace("*", svc)
            return spec
        return None

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
            if name in self._cumulative:
                obs[i] = deltas.get(name, 0.0)
            elif name in self._special:
                kind, dep = self._special[name]
                d = deltas.get(dep, 0.0)
                if kind == "request_rate":
                    obs[i] = d / self.step_duration
                elif kind == "avg_inter_arrival_time":
                    obs[i] = self.step_duration / max(d, 1.0)
            elif name in self._utilization:
                num_key, cap_key = self._utilization[name]
                d_num = deltas.get(num_key, 0.0)
                capacity = float(snapshot.get(cap_key, 0.0))
                max_possible = capacity * self.step_duration
                obs[i] = (d_num / max_possible) if max_possible > 0 else 0.0
            elif name in self._computed:
                num_key, den_key, default = self._computed[name]
                num = deltas.get(num_key, 0.0)
                den = deltas.get(den_key, 0.0)
                obs[i] = (num / den) if den > 0 else default
            else:
                obs[i] = float(snapshot.get(name, 0.0))
        return obs
