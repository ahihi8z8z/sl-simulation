from __future__ import annotations

from collections import deque
from typing import Any


class MetricStore:
    """Ring buffer metric store.

    Stores time-series data per metric name in a bounded deque.
    Each entry is a (timestamp, value) tuple.
    """

    def __init__(self, max_history_length: int = 1000):
        self.max_history_length = max_history_length
        self._data: dict[str, deque[tuple[float, Any]]] = {}

    def put(self, metric_name: str, timestamp: float, value: Any) -> None:
        """Append a data point for *metric_name*."""
        buf = self._data.setdefault(
            metric_name, deque(maxlen=self.max_history_length)
        )
        buf.append((timestamp, value))

    def get_latest(self, metric_name: str) -> tuple[float, Any] | None:
        """Return the most recent (timestamp, value) or None."""
        buf = self._data.get(metric_name)
        if buf:
            return buf[-1]
        return None

    def query_range(
        self, metric_name: str, start: float, end: float
    ) -> list[tuple[float, Any]]:
        """Return all entries with start <= timestamp <= end."""
        buf = self._data.get(metric_name)
        if not buf:
            return []
        return [(t, v) for t, v in buf if start <= t <= end]

    def get_all_metric_names(self) -> list[str]:
        return list(self._data.keys())

    def get_all_entries(self, metric_name: str) -> list[tuple[float, Any]]:
        buf = self._data.get(metric_name)
        if not buf:
            return []
        return list(buf)

    def __len__(self) -> int:
        return sum(len(buf) for buf in self._data.values())
