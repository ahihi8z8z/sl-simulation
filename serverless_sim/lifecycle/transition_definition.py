from dataclasses import dataclass
from typing import Optional


@dataclass
class TransitionDefinition:
    """Defines a transition between lifecycle states."""

    from_state: str = ""
    to_state: str = ""
    transition_time: float = 0.0
    transition_cpu: float = 0.0
    transition_memory: float = 0.0
    allowed_services: Optional[list] = None
