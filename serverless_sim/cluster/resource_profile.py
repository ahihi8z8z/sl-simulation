from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceProfile:
    """CPU/memory resource pair with arithmetic operations."""

    cpu: float = 0.0
    memory: float = 0.0

    def add(self, other: ResourceProfile) -> ResourceProfile:
        """Return a new profile with resources summed."""
        return ResourceProfile(
            cpu=self.cpu + other.cpu,
            memory=self.memory + other.memory,
        )

    def subtract(self, other: ResourceProfile) -> ResourceProfile:
        """Return a new profile with *other* subtracted."""
        return ResourceProfile(
            cpu=self.cpu - other.cpu,
            memory=self.memory - other.memory,
        )

    def can_fit(self, request: ResourceProfile) -> bool:
        """Return True if *request* fits within this profile."""
        return self.cpu >= request.cpu and self.memory >= request.memory

    def is_zero(self) -> bool:
        return self.cpu <= 0.0 and self.memory <= 0.0

    def __repr__(self) -> str:
        return f"ResourceProfile(cpu={self.cpu:.2f}, memory={self.memory:.1f})"
