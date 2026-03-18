from dataclasses import dataclass


@dataclass
class ResourceProfile:
    """CPU/memory resource pair."""

    cpu: float = 0.0
    memory: float = 0.0
