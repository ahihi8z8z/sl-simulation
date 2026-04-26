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
    min_instances: int = 0
    max_instances: int = 0  # 0 = unlimited

    # Resource request per container (for capacity checks)
    # If set, used instead of peak_memory/peak_cpu for scheduling decisions
    request_cpu: float = 0.0
    request_memory: float = 0.0

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
        """Memory for capacity checks. Uses request_memory if set, else max across states."""
        if self.request_memory > 0:
            return self.request_memory
        sm = self.state_machine
        return max(sd.memory for sd in sm.states.values())

    @property
    def peak_cpu(self) -> float:
        """CPU for capacity checks. Uses request_cpu if set, else max across states."""
        if self.request_cpu > 0:
            return self.request_cpu
        sm = self.state_machine
        return max(sd.cpu for sd in sm.states.values())

    @classmethod
    def from_config(cls, cfg: dict) -> ServiceClass:
        """Build a ServiceClass from a service config dict."""
        from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine

        svc = cls(
            service_id=cfg["service_id"],
            display_name=cfg.get("display_name", cfg["service_id"]),
            min_instances=cfg.get("min_instances", 0),
            max_instances=cfg.get("max_instances", 0),
            request_cpu=cfg.get("request_cpu", 0.0),
            request_memory=cfg.get("request_memory", 0.0),
        )

        # Build per-service lifecycle
        lifecycle_cfg = cfg.get("lifecycle")
        if lifecycle_cfg:
            svc.state_machine = OpenWhiskExtendedStateMachine.from_lifecycle_config(lifecycle_cfg)
        else:
            svc.state_machine = OpenWhiskExtendedStateMachine.default()

        return svc
