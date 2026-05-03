"""Helper for randomizing services[*].workload.start_minute on each gym reset.

Auto-detects upper bound from the trace file (CSV) so the agent can land
anywhere within the trace's effective range. The same `start_minute` is
shared across all services with a trace-based workload (one draw per reset).
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache

# Generators we know how to handle (others are skipped silently).
_TRACE_GENERATORS = {"trace", "aggregate_trace"}


@lru_cache(maxsize=64)
def _trace_max_minute(trace_path: str, generator: str) -> int:
    """Return the largest minute index that still produces traffic."""
    if not os.path.exists(trace_path):
        return 0
    if generator == "aggregate_trace":
        max_minute = 0
        with open(trace_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    max_minute = max(max_minute, int(float(row["minute"])))
                except (KeyError, ValueError, TypeError):
                    continue
        return max_minute
    # trace replay → minutes derived from timestamp column (seconds)
    max_ts = 0.0
    min_ts: float | None = None
    with open(trace_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestamp"])
            except (KeyError, ValueError, TypeError):
                continue
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if ts > max_ts:
                max_ts = ts
    if min_ts is None:
        return 0
    return int((max_ts - min_ts) // 60)


def apply_random_start_minute(sim_config: dict, gym_config: dict, np_random) -> int | None:
    """Mutate sim_config services in place; return the chosen start_minute.

    gym_config layout::

        "random_start_minute": {
            "enabled": true,
            "min": 0,        // optional, default 0
            "max": null      // optional; null = auto-detect from trace
        }

    Returns ``None`` when disabled or when no trace-based service is present.
    """
    cfg = gym_config.get("random_start_minute") or {}
    if not cfg.get("enabled"):
        return None

    services = sim_config.get("services", [])
    trace_services = [
        s for s in services
        if (s.get("workload", {}).get("generator") in _TRACE_GENERATORS)
        and s["workload"].get("trace_path")
    ]
    if not trace_services:
        return None

    lo = int(cfg.get("min", 0))
    hi = cfg.get("max", None)
    if hi is None:
        # Auto-detect: use the SHORTEST trace so all services have valid traffic
        # at every offset in [lo, hi].
        detected = []
        for svc in trace_services:
            wl = svc["workload"]
            detected.append(_trace_max_minute(wl["trace_path"], wl["generator"]))
            # Subtract the existing start_minute so we don't overshoot.
            existing = int(wl.get("start_minute") or 0)
            detected[-1] = max(detected[-1] - existing, 0)
        hi = min(detected) if detected else 0
    else:
        hi = int(hi)

    if hi <= lo:
        # Nothing to randomize over — leave config untouched.
        return None

    start_minute = int(np_random.integers(lo, hi))
    for svc in trace_services:
        svc["workload"]["start_minute"] = start_minute
    return start_minute
