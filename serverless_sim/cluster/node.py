from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import simpy

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.cluster.serving_model import BaseServingModel

if TYPE_CHECKING:
    from serverless_sim.cluster.compute_class import ComputeClass
    from serverless_sim.core.simulation.sim_context import SimContext


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
        self.max_queue_depth = compute_class.max_queue_depth  # 0 = unlimited
        self._ctx: SimContext | None = None  # set by ClusterManager when ctx is available

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

    @property
    def queue_depth(self) -> int:
        """Current number of requests waiting in the queue."""
        return len(self.queue.items)

    @property
    def queue_is_full(self) -> bool:
        """True if queue has reached max_queue_depth (0 = never full)."""
        if self.max_queue_depth <= 0:
            return False
        return self.queue_depth >= self.max_queue_depth

    # ------------------------------------------------------------------
    # Pull loop — processes requests from the queue
    # ------------------------------------------------------------------

    def start_pull_loop(self) -> None:
        """Start the SimPy process that pulls requests from the queue."""
        self.env.process(self._pull_loop())

    def _pull_loop(self):
        """Pull requests from the queue and process them via lifecycle manager."""
        while True:
            invocation = yield self.queue.get()
            self.logger.debug(
                "t=%.3f | %s | received request %s (service=%s)",
                self.env.now,
                self.node_id,
                invocation.request_id,
                invocation.service_id,
            )
            # If lifecycle manager is available, process the request
            if self._ctx and self._ctx.lifecycle_manager:
                self.env.process(self._process_request(invocation))

    def _process_request(self, invocation):
        """SimPy process: find/create instance → execute → complete."""
        lm = self._ctx.lifecycle_manager

        # Try to find a warm reusable instance
        instance = lm.find_reusable_instance(self, invocation.service_id)
        cold_start = instance is None

        if instance is None:
            # Try to promote an intermediate instance (faster than full cold start)
            promotable = lm.find_promotable_instance(self, invocation.service_id)
            if promotable is not None:
                promote_proc = lm.promote_instance(self, promotable)
                instance = yield promote_proc
            else:
                # Full cold start: create new instance from null
                cold_start_proc = lm.prepare_instance_for_service(self, invocation.service_id)
                instance = yield cold_start_proc

        # Acquire concurrency slot
        req = instance.slots.request()
        yield req

        # Start execution
        lm.start_execution(instance, invocation)
        if cold_start:
            invocation.cold_start = True

        # Compute service time
        service_time = self.serving_model.estimate_service_time(
            invocation.job_size, self
        )
        yield self.env.timeout(service_time)

        # Release resources
        lm.finish_execution(instance, invocation)
        instance.slots.release(req)

        invocation.status = "completed"
        self._ctx.request_table.finalize(invocation)
        self.logger.debug(
            "t=%.3f | %s | completed %s (cold=%s, duration=%.3f)",
            self.env.now,
            self.node_id,
            invocation.request_id,
            cold_start,
            service_time,
        )

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, capacity={self.capacity}, "
            f"allocated={self.allocated}, available={self.available})"
        )
