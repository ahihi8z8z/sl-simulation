from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import simpy

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.cluster.serving_model import BaseServingModel

if TYPE_CHECKING:
    from serverless_sim.cluster.compute_class import ComputeClass


class Node:
    """Represents a single worker node with a simpy.Store request queue."""

    def __init__(
        self,
        env: simpy.Environment,
        node_id: str,
        capacity: ResourceProfile,
        compute_class: ComputeClass,
        serving_model: BaseServingModel,
        logger: logging.Logger | None = None,
    ):
        self.env = env
        self.node_id = node_id
        self.capacity = capacity
        self.compute_class = compute_class
        self.serving_model = serving_model
        self.logger = logger or logging.getLogger(__name__)

        # Resource accounting
        self.allocated = ResourceProfile(cpu=0.0, memory=0.0)
        self.available = ResourceProfile(cpu=capacity.cpu, memory=capacity.memory)

        # Request queue (unbounded Store)
        self.queue: simpy.Store = simpy.Store(env)

        self.enabled = True

    # ------------------------------------------------------------------
    # Resource accounting
    # ------------------------------------------------------------------

    def allocate(self, request: ResourceProfile) -> bool:
        """Try to allocate resources. Returns True on success."""
        if not self.available.can_fit(request):
            return False
        self.allocated = self.allocated.add(request)
        self.available = self.available.subtract(request)
        return True

    def release(self, request: ResourceProfile) -> None:
        """Release previously allocated resources."""
        self.allocated = self.allocated.subtract(request)
        self.available = self.available.add(request)

    def can_fit(self, request: ResourceProfile) -> bool:
        """Check if resources can be allocated without modifying state."""
        return self.available.can_fit(request)

    # ------------------------------------------------------------------
    # Pull loop — processes requests from the queue
    # ------------------------------------------------------------------

    def start_pull_loop(self) -> None:
        """Start the SimPy process that pulls requests from the queue."""
        self.env.process(self._pull_loop())

    def _pull_loop(self):
        """Pull requests from the queue and log them.

        In later steps the lifecycle_manager will handle actual execution.
        """
        while True:
            invocation = yield self.queue.get()
            self.logger.debug(
                "t=%.3f | %s | received request %s (service=%s)",
                self.env.now,
                self.node_id,
                invocation.request_id,
                invocation.service_id,
            )

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, capacity={self.capacity}, "
            f"allocated={self.allocated}, available={self.available})"
        )
