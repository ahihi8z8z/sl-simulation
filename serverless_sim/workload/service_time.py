"""Service time providers — determine execution duration per request.

Configured per-service via ``services[i].service_time``:

    {"mode": "fixed", "duration": 0.45}
    {"mode": "sample_csv", "csv_path": "data/durations.csv"}

Omitted → defaults to FixedServiceTime(0.1).

The provider's ``assign()`` is called by generators after creating
each Invocation, setting ``inv.service_time`` before dispatch.
"""

from __future__ import annotations

import csv as csv_mod
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from serverless_sim.workload.invocation import Invocation


class BaseServiceTimeProvider:
    """Interface for service time providers."""

    def assign(self, inv: Invocation, rng: np.random.Generator) -> None:
        raise NotImplementedError


class FixedServiceTime(BaseServiceTimeProvider):
    """Every request gets the same fixed duration."""

    def __init__(self, duration: float = 0.1):
        self.duration = duration

    def assign(self, inv: Invocation, rng: np.random.Generator) -> None:
        inv.service_time = self.duration


class SampleCsvServiceTime(BaseServiceTimeProvider):
    """Sample duration from a CSV file.

    CSV must have a ``duration`` column. Each request gets a randomly
    sampled value from the loaded durations.
    """

    def __init__(self, csv_path: str):
        self._durations: list[float] = []
        self._load(csv_path)

    def _load(self, path: str) -> None:
        with open(path, newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                try:
                    val = float(row["duration"])
                except (ValueError, TypeError, KeyError):
                    continue
                if math.isnan(val) or val <= 0:
                    continue
                self._durations.append(val)
        if not self._durations:
            raise ValueError(f"No valid durations found in {path}")

    def assign(self, inv: Invocation, rng: np.random.Generator) -> None:
        inv.service_time = self._durations[rng.integers(len(self._durations))]


def create_service_time_provider(svc_cfg: dict) -> BaseServiceTimeProvider:
    """Create a provider from a single ``services[i].service_time`` dict."""
    st_cfg = svc_cfg.get("service_time", {})
    mode = st_cfg.get("mode", "fixed")

    if mode == "fixed":
        duration = st_cfg.get("duration", 0.1)
        return FixedServiceTime(duration=duration)
    elif mode == "sample_csv":
        csv_path = st_cfg["csv_path"]
        return SampleCsvServiceTime(csv_path=csv_path)
    else:
        raise ValueError(f"Unknown service_time mode: {mode}")
