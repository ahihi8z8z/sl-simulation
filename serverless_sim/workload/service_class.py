from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine


@dataclass
class ServiceClass:
    """Represents a function/service type with its own lifecycle profile."""

    service_id: str = ""
    display_name: str = ""
    arrival_mode: str = "poisson"
    arrival_rate: float = 1.0
    job_size: float = 1.0
    max_concurrency: int = 1
    min_instances: int = 0
    max_instances: int = 0  # 0 = unlimited

    # Per-service lifecycle (set after construction via build_lifecycle)
    _state_machine: OpenWhiskExtendedStateMachine | None = field(
        default=None, repr=False, compare=False,
    )

    @property
    def state_machine(self) -> OpenWhiskExtendedStateMachine:
        if self._state_machine is None:
            from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine
            self._state_machine = OpenWhiskExtendedStateMachine.default()
        return self._state_machine

    @state_machine.setter
    def state_machine(self, sm: OpenWhiskExtendedStateMachine) -> None:
        self._state_machine = sm

    @property
    def peak_memory(self) -> float:
        """Max memory across all states (for LoadBalancer capacity check)."""
        sm = self.state_machine
        return max(sd.memory for sd in sm.states.values())

    @property
    def peak_cpu(self) -> float:
        """Max CPU across all states."""
        sm = self.state_machine
        return max(sd.cpu for sd in sm.states.values())

    @classmethod
    def from_config(cls, cfg: dict) -> ServiceClass:
        """Build a ServiceClass from a service config dict."""
        from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine

        svc = cls(
            service_id=cfg["service_id"],
            display_name=cfg.get("display_name", cfg["service_id"]),
            arrival_mode=cfg.get("arrival_mode", "poisson"),
            arrival_rate=cfg["arrival_rate"],
            job_size=cfg["job_size"],
            max_concurrency=cfg["max_concurrency"],
            min_instances=cfg.get("min_instances", 0),
            max_instances=cfg.get("max_instances", 0),
        )

        # Build per-service lifecycle
        lifecycle_cfg = cfg.get("lifecycle")
        if lifecycle_cfg:
            svc.state_machine = OpenWhiskExtendedStateMachine.from_lifecycle_config(lifecycle_cfg)
        else:
            svc.state_machine = OpenWhiskExtendedStateMachine.default()

        return svc
