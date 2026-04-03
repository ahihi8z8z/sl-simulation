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

        # System reserved resources (applied only when node has pods)
        self.reserved = ResourceProfile(
            cpu=compute_class.reserved_cpu,
            memory=compute_class.reserved_memory,
        )
        self._reserved_applied = False

        # Resource accounting — starts at full capacity (reserved applied on first allocate)
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
        # Apply reserved on first allocation (node gets a pod)
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
        # Clamp to avoid float precision artifacts
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
        # Release reserved when no more containers
        if self._reserved_applied and self.allocated.cpu <= 0 and self.allocated.memory <= 0:
            self.available = self.available.add(self.reserved)
            self._reserved_applied = False

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
        if not self._ctx or not self._ctx.lifecycle_manager:
            return
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
                # Check max_instances before cold start
                if self._ctx.autoscaling_manager:
                    max_inst = self._ctx.autoscaling_manager.get_max_instances(invocation.service_id)
                    if max_inst > 0:
                        total = self._ctx.autoscaling_manager._count_total_instances(invocation.service_id)
                        if total >= max_inst:
                            invocation.dropped = True
                            invocation.drop_reason = "max_instances"
                            invocation.status = "dropped"
                            invocation.completion_time = self.env.now
                            self._ctx.request_table.finalize(invocation)
                            self.logger.debug(
                                "t=%.3f | %s | DROP %s (max_instances=%d)",
                                self.env.now, self.node_id,
                                invocation.request_id, max_inst,
                            )
                            return

                # Check if node has enough memory for a new container
                service = self._ctx.workload_manager.services[invocation.service_id]
                peak_mem = ResourceProfile(cpu=0.0, memory=service.peak_memory)
                if not self.available.can_fit(peak_mem):
                    invocation.dropped = True
                    invocation.drop_reason = "no_resources"
                    invocation.status = "dropped"
                    invocation.completion_time = self.env.now
                    self._ctx.request_table.finalize(invocation)
                    self.logger.debug(
                        "t=%.3f | %s | DROP %s (no_resources, avail_mem=%.0f, need=%.0f)",
                        self.env.now, self.node_id,
                        invocation.request_id,
                        self.available.memory, service.peak_memory,
                    )
                    return

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
            invocation.job_size, self, service_time=invocation.service_time,
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
