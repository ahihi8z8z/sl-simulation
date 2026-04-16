from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import simpy

from serverless_sim.cluster.resource_profile import ResourceProfile

if TYPE_CHECKING:
    from serverless_sim.cluster.compute_class import ComputeClass
    from serverless_sim.core.simulation.sim_context import SimContext


class Node:
    """Worker node — container host with resource tracking.

    Nodes hold containers and track resource usage. All scheduling
    decisions (reuse, promote, cold start) are made by the central
    load balancer, not by the node.
    """

    def __init__(
        self,
        env: simpy.Environment,
        node_id: str,
        capacity: ResourceProfile,
        compute_class: ComputeClass,
        logger: logging.Logger | None = None,
    ):
        self.env = env
        self.node_id = node_id
        self.capacity = capacity
        self.compute_class = compute_class
        self.logger = logger or logging.getLogger(__name__)

        # System reserved resources (applied only when node has pods)
        self.reserved = ResourceProfile(
            cpu=compute_class.reserved_cpu,
            memory=compute_class.reserved_memory,
        )
        self._reserved_applied = False

        # Resource accounting — starts at full capacity (reserved applied on first allocate)
        self.allocated = ResourceProfile(cpu=0.0, memory=0.0)
        self.available = ResourceProfile(cpu=capacity.cpu, memory=capacity.memory)

        # Flavor accounting — fixed reservation for placement decisions
        # Initialized with reserved resources (always counted toward flavor)
        self.flavor_cpu_used = compute_class.reserved_cpu
        self.flavor_memory_used = compute_class.reserved_memory

        self.enabled = True
        self._ctx: SimContext | None = None  # set by ClusterManager when ctx is available

    # ------------------------------------------------------------------
    # Resource accounting
    # ------------------------------------------------------------------

    def allocate(self, request: ResourceProfile) -> bool:
        """Try to allocate resources. Returns True on success."""
        if not self._reserved_applied and (self.reserved.cpu > 0 or self.reserved.memory > 0):
            self.available = self.available.subtract(self.reserved)
            self._reserved_applied = True
        if not self.available.can_fit(request):
            return False
        self.allocated = self.allocated.add(request)
        self.available = self.available.subtract(request)
        return True

    def release(self, request: ResourceProfile) -> None:
        """Release previously allocated resources."""
        self.allocated = self.allocated.subtract(request)
        self.available = self.available.add(request)
        self.allocated = ResourceProfile(
            cpu=max(0.0, self.allocated.cpu),
            memory=max(0.0, self.allocated.memory),
        )
        max_avail_cpu = self.capacity.cpu - (self.reserved.cpu if self._reserved_applied else 0.0)
        max_avail_mem = self.capacity.memory - (self.reserved.memory if self._reserved_applied else 0.0)
        self.available = ResourceProfile(
            cpu=min(max_avail_cpu, self.available.cpu),
            memory=min(max_avail_mem, self.available.memory),
        )
        if self._reserved_applied and self.allocated.cpu <= 0 and self.allocated.memory <= 0:
            self.available = self.available.add(self.reserved)
            self._reserved_applied = False

    def can_fit(self, request: ResourceProfile) -> bool:
        """Check if resources can be allocated without modifying state."""
        return self.available.can_fit(request)

    def can_fit_flavor(self, cpu: float, memory: float) -> bool:
        """Check if node has enough flavor capacity for a new container."""
        return (self.capacity.cpu - self.flavor_cpu_used >= cpu
                and self.capacity.memory - self.flavor_memory_used >= memory)

    def reserve_flavor(self, cpu: float, memory: float) -> None:
        """Reserve flavor resources when creating a container."""
        self.flavor_cpu_used += cpu
        self.flavor_memory_used += memory

    def release_flavor(self, cpu: float, memory: float) -> None:
        """Release flavor resources when evicting a container."""
        self.flavor_cpu_used -= cpu
        self.flavor_memory_used -= memory
        self.flavor_cpu_used = max(0.0, self.flavor_cpu_used)
        self.flavor_memory_used = max(0.0, self.flavor_memory_used)

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, capacity={self.capacity}, "
            f"allocated={self.allocated}, available={self.available})"
        )
