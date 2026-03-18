from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler


class AutoscalingAPI:
    """Public API for controller/RL to adjust autoscaling parameters."""

    def __init__(self, autoscaler: OpenWhiskPoolAutoscaler):
        self._autoscaler = autoscaler

    def get_idle_timeout(self, service_id: str) -> float:
        return self._autoscaler.get_idle_timeout(service_id)

    def set_idle_timeout(self, service_id: str, value: float) -> None:
        self._autoscaler.set_idle_timeout(service_id, value)

    def get_prewarm_count(self, service_id: str) -> int:
        return self._autoscaler.get_prewarm_count(service_id)

    def set_prewarm_count(self, service_id: str, count: int) -> None:
        self._autoscaler.set_prewarm_count(service_id, count)

    def trigger_reconcile(self) -> None:
        self._autoscaler.reconcile()
