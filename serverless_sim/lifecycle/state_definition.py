from dataclasses import dataclass


@dataclass
class StateDefinition:
    """Defines a lifecycle state."""

    state_name: str = ""
    category: str = "stable"  # "stable" or "transient"
    steady_cpu: float = 0.0
    steady_memory: float = 0.0
    service_bound: bool = False
    reusable: bool = True

    @property
    def is_stable(self) -> bool:
        return self.category == "stable"

    @property
    def is_transient(self) -> bool:
        return self.category == "transient"
