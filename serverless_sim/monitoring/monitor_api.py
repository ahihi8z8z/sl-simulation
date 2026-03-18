from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.monitoring.monitor_manager import MonitorManager


class MonitorAPI:
    """Read-only API for querying metrics."""

    def __init__(self, manager: MonitorManager):
        self._manager = manager
        self._store = manager.store

    def get_latest(self, metric_name: str) -> tuple[float, Any] | None:
        """Get the latest value of a metric."""
        return self._store.get_latest(metric_name)

    def get_latest_value(self, metric_name: str, default: Any = 0.0) -> Any:
        """Get just the latest value (no timestamp), with a default."""
        entry = self._store.get_latest(metric_name)
        if entry is None:
            return default
        return entry[1]

    def query_range(self, metric_name: str, start: float, end: float) -> list[tuple[float, Any]]:
        """Get all entries of a metric in [start, end]."""
        return self._store.query_range(metric_name, start, end)

    def get_snapshot(self, metric_names: list[str] | None = None) -> dict[str, Any]:
        """Get the latest value of multiple metrics as a dict."""
        if metric_names is None:
            metric_names = self._store.get_all_metric_names()
        result = {}
        for name in metric_names:
            entry = self._store.get_latest(name)
            if entry is not None:
                result[name] = entry[1]
        return result

    def get_all_metric_names(self) -> list[str]:
        return self._store.get_all_metric_names()
