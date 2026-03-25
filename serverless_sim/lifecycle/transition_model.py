"""Pluggable models for sampling transition latency and resource usage.

Three built-in models:

- **DeterministicTransitionModel** — fixed values (current default)
- **CsvSampleTransitionModel** — sample from a CSV trace file
- **DistributionTransitionModel** — sample from a statistical distribution
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class TransitionSample:
    """One sampled set of transition parameters."""
    time: float = 0.0
    cpu: float = 0.0
    memory: float = 0.0


class BaseTransitionModel:
    """Interface for transition parameter sampling."""

    def sample(self, from_state: str, to_state: str, rng: np.random.Generator) -> TransitionSample:
        """Sample transition time, cpu, memory for a given transition."""
        raise NotImplementedError


class DeterministicTransitionModel(BaseTransitionModel):
    """Fixed values per transition (no randomness)."""

    def __init__(self):
        self._values: dict[tuple[str, str], TransitionSample] = {}

    def set(self, from_state: str, to_state: str, time: float = 0.0,
            cpu: float = 0.0, memory: float = 0.0) -> None:
        self._values[(from_state, to_state)] = TransitionSample(time, cpu, memory)

    def sample(self, from_state: str, to_state: str, rng: np.random.Generator) -> TransitionSample:
        return self._values.get(
            (from_state, to_state),
            TransitionSample(),
        )


class CsvSampleTransitionModel(BaseTransitionModel):
    """Sample transition parameters from a pre-loaded CSV trace.

    CSV format::

        from_state,to_state,time,cpu,memory
        null,prewarm,0.45,0.1,0
        null,prewarm,0.52,0.12,0
        prewarm,warm,0.28,0.0,0

    Each row is one observed transition.  ``sample()`` picks a random
    row matching ``(from_state, to_state)``.  Falls back to zeros if
    no data for a transition.
    """

    def __init__(self):
        # (from, to) → list of TransitionSample
        self._data: dict[tuple[str, str], list[TransitionSample]] = {}

    @classmethod
    def from_csv(cls, path: str) -> CsvSampleTransitionModel:
        """Load transition samples from a CSV file."""
        model = cls()
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["from_state"], row["to_state"])
                sample = TransitionSample(
                    time=float(row.get("time", 0)),
                    cpu=float(row.get("cpu", 0)),
                    memory=float(row.get("memory", 0)),
                )
                model._data.setdefault(key, []).append(sample)
        return model

    def sample(self, from_state: str, to_state: str, rng: np.random.Generator) -> TransitionSample:
        key = (from_state, to_state)
        samples = self._data.get(key)
        if not samples:
            return TransitionSample()
        idx = rng.integers(0, len(samples))
        return samples[idx]


class DistributionTransitionModel(BaseTransitionModel):
    """Sample from statistical distributions per transition.

    Config format per transition::

        {"distribution": "lognormal", "time_mean": 0.5, "time_std": 0.2,
         "cpu": 0.1, "memory": 0}

    Supported distributions for time: ``"lognormal"``, ``"normal"``, ``"uniform"``.
    CPU and memory are fixed (deterministic).
    """

    def __init__(self):
        self._configs: dict[tuple[str, str], dict] = {}

    def set(self, from_state: str, to_state: str, config: dict) -> None:
        self._configs[(from_state, to_state)] = config

    def sample(self, from_state: str, to_state: str, rng: np.random.Generator) -> TransitionSample:
        cfg = self._configs.get((from_state, to_state))
        if cfg is None:
            return TransitionSample()

        dist = cfg.get("distribution", "deterministic")
        time_val = self._sample_time(dist, cfg, rng)
        return TransitionSample(
            time=max(0.0, time_val),
            cpu=cfg.get("cpu", 0.0),
            memory=cfg.get("memory", 0.0),
        )

    @staticmethod
    def _sample_time(dist: str, cfg: dict, rng: np.random.Generator) -> float:
        if dist == "deterministic":
            return cfg.get("time", 0.0)
        elif dist == "lognormal":
            mean = cfg.get("time_mean", 0.5)
            std = cfg.get("time_std", 0.1)
            # Convert to lognormal params
            sigma2 = np.log(1 + (std / mean) ** 2)
            mu = np.log(mean) - sigma2 / 2
            return float(rng.lognormal(mu, np.sqrt(sigma2)))
        elif dist == "normal":
            mean = cfg.get("time_mean", 0.5)
            std = cfg.get("time_std", 0.1)
            return float(rng.normal(mean, std))
        elif dist == "uniform":
            low = cfg.get("time_low", 0.3)
            high = cfg.get("time_high", 0.7)
            return float(rng.uniform(low, high))
        else:
            return cfg.get("time", 0.0)
