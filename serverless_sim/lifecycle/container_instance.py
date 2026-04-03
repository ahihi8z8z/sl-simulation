from __future__ import annotations

import itertools
from typing import Optional

import simpy


_id_counter = itertools.count(1)


class ContainerInstance:
    """Container instance with simpy.Resource for concurrency slots."""

    def __init__(
        self,
        env: simpy.Environment,
        service_id: str,
        node_id: str,
        max_concurrency: int = 1,
    ):
        self.env = env
        self.instance_id: str = f"inst-{next(_id_counter)}"
        self.service_id: str = service_id
        self.node_id: str = node_id
        self.max_concurrency: int = max_concurrency

        # SimPy resource for concurrency control
        self.slots: simpy.Resource = simpy.Resource(env, capacity=max_concurrency)

        # State tracking
        self.state: str = "null"
        self.target_state: Optional[str] = None
        self.cold_start: bool = True  # True until first request completes

        # Timing
        self.created_at: float = env.now
        self.last_used_at: float = env.now
        self.state_entered_at: float = env.now

        # Request tracking
        self.active_requests: int = 0

        # Currently allocated resources on node (for correct release)
        self.allocated_cpu: float = 0.0
        self.allocated_memory: float = 0.0

        # Eviction flag — checked by in-flight _cold_start processes
        self.evicted: bool = False

    @property
    def is_idle(self) -> bool:
        return self.active_requests == 0

    @property
    def available_slots(self) -> int:
        return self.max_concurrency - self.slots.count

    def __repr__(self) -> str:
        return (
            f"ContainerInstance(id={self.instance_id}, service={self.service_id}, "
            f"state={self.state}, active={self.active_requests}/{self.max_concurrency})"
        )
