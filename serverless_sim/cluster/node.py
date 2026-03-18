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
        """SimPy process: find/create instance → execute → complete.

        Races execution against a timeout measured from arrival_time.
        """
        lm = self._ctx.lifecycle_manager

        # Check if already timed out before we even start
        remaining = invocation.timeout - (self.env.now - invocation.arrival_time)
        if remaining <= 0:
            self._timeout_invocation(invocation, lm)
            return

        # Try to find a warm reusable instance
        instance = lm.find_reusable_instance(self, invocation.service_id)
        cold_start = instance is None

        if instance is None:
            # Cold start: create new instance and wait for it to be ready
            # Race cold start against timeout
            cold_start_proc = lm.prepare_instance_for_service(self, invocation.service_id)
            remaining = invocation.timeout - (self.env.now - invocation.arrival_time)
            timeout_event = self.env.timeout(max(remaining, 0))
            result = yield cold_start_proc | timeout_event
            if cold_start_proc not in result:
                # Timeout during cold start
                self._timeout_invocation(invocation, lm)
                return
            instance = result[cold_start_proc]

        # Check timeout again after cold start
        remaining = invocation.timeout - (self.env.now - invocation.arrival_time)
        if remaining <= 0:
            self._timeout_invocation(invocation, lm)
            return

        # Acquire concurrency slot
        req = instance.slots.request()
        yield req

        # Start execution
        lm.start_execution(instance, invocation)
        if cold_start:
            invocation.cold_start = True

        # Compute service time and race against timeout
        service_time = self.serving_model.estimate_service_time(
            invocation.job_size, self
        )
        remaining = invocation.timeout - (self.env.now - invocation.arrival_time)

        exec_event = self.env.timeout(service_time)
        timeout_event = self.env.timeout(max(remaining, 0))

        result = yield exec_event | timeout_event
        if exec_event in result:
            # Normal completion
            lm.finish_execution(instance, invocation)
            instance.slots.release(req)
            self.logger.debug(
                "t=%.3f | %s | completed %s (cold=%s, duration=%.3f)",
                self.env.now,
                self.node_id,
                invocation.request_id,
                cold_start,
                service_time,
            )
        else:
            # Timeout during execution — abort and release resources
            lm.finish_execution(instance, invocation)
            instance.slots.release(req)
            invocation.timed_out = True
            invocation.drop_reason = "timeout"
            invocation.status = "timed_out"
            self.logger.debug(
                "t=%.3f | %s | TIMEOUT %s (elapsed=%.3f, limit=%.3f)",
                self.env.now,
                self.node_id,
                invocation.request_id,
                self.env.now - invocation.arrival_time,
                invocation.timeout,
            )

    def _timeout_invocation(self, invocation, lm):
        """Mark invocation as timed out before execution started."""
        invocation.timed_out = True
        invocation.dropped = True
        invocation.drop_reason = "timeout"
        invocation.status = "timed_out"
        invocation.completion_time = self.env.now
        self.logger.debug(
            "t=%.3f | %s | TIMEOUT (pre-exec) %s",
            self.env.now,
            self.node_id,
            invocation.request_id,
        )

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, capacity={self.capacity}, "
            f"allocated={self.allocated}, available={self.available})"
        )
