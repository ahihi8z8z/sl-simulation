from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.autoscaling.autoscaler import OpenWhiskPoolAutoscaler


class AutoscalingAPI:
    """Public API for controller/RL to adjust autoscaling parameters."""

    def __init__(self, autoscaler: OpenWhiskPoolAutoscaler):
        self._autoscaler = autoscaler

    # -- min/max instances --

    def get_min_instances(self, service_id: str) -> int:
        return self._autoscaler.get_min_instances(service_id)

    def set_min_instances(self, service_id: str, value: int) -> None:
        self._autoscaler.set_min_instances(service_id, value)

    def get_max_instances(self, service_id: str) -> int:
        return self._autoscaler.get_max_instances(service_id)

    def set_max_instances(self, service_id: str, value: int) -> None:
        self._autoscaler.set_max_instances(service_id, value)

    # -- Per-state pool targets --

    def get_pool_target(self, service_id: str, state: str) -> int:
        return self._autoscaler.get_pool_target(service_id, state)

    def set_pool_target(self, service_id: str, state: str, count: int) -> None:
        self._autoscaler.set_pool_target(service_id, state, count)

    def get_all_pool_targets(self, service_id: str) -> dict[str, int]:
        return self._autoscaler.get_all_pool_targets(service_id)

    def batch_set_pool_targets(self, service_id: str, targets: dict[str, int]) -> None:
        self._autoscaler.batch_set_pool_targets(service_id, targets)

    # -- Idle timeout --

    def get_idle_timeout(self, service_id: str) -> float:
        return self._autoscaler.get_idle_timeout(service_id)

    def set_idle_timeout(self, service_id: str, value: float) -> None:
        self._autoscaler.set_idle_timeout(service_id, value)

    def trigger_reconcile(self) -> None:
        self._autoscaler.reconcile()
