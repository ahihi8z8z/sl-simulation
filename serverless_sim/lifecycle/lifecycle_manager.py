from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import simpy

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.lifecycle.container_instance import ContainerInstance
from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine

if TYPE_CHECKING:
    from serverless_sim.cluster.node import Node
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.workload.invocation import Invocation


class LifecycleManager:
    """Manages container instance lifecycle on all nodes."""

    def __init__(self, ctx: SimContext, state_machine: OpenWhiskExtendedStateMachine | None = None):
        self.ctx = ctx
        self.logger = ctx.logger
        self.sm = state_machine or OpenWhiskExtendedStateMachine.default()

        # node_id → list of ContainerInstance (only alive instances)
        self.instances: dict[str, list[ContainerInstance]] = {}
        self._evicted_count: int = 0

    def _get_instances(self, node_id: str) -> list[ContainerInstance]:
        return self.instances.setdefault(node_id, [])

    # ------------------------------------------------------------------
    # Find / prepare instances
    # ------------------------------------------------------------------

    def find_reusable_instance(self, node: Node, service_id: str) -> ContainerInstance | None:
        """Find a warm instance for this service with available slots."""
        for inst in self._get_instances(node.node_id):
            if (
                inst.service_id == service_id
                and inst.state == "warm"
                and inst.available_slots > 0
            ):
                return inst
        return None

    def prepare_instance_for_service(
        self, node: Node, service_id: str, target_state: str = "warm",
    ) -> simpy.events.Event:
        """Create a new instance and transition it to *target_state*.

        *target_state* must be a state in the cold-start chain (e.g.
        ``"prewarm"``, ``"code_loaded"``, ``"warm"``).  Defaults to
        ``"warm"`` for a full cold start.

        Returns a SimPy process that yields until the instance is ready.
        """
        return self.ctx.env.process(self._cold_start(node, service_id, target_state))

    def _cold_start(self, node: Node, service_id: str, target_state: str = "warm"):
        """SimPy process: follow chain from null → target_state."""
        service = self.ctx.workload_manager.services[service_id]

        inst = ContainerInstance(
            env=self.ctx.env,
            service_id=service_id,
            node_id=node.node_id,
            max_concurrency=service.max_concurrency,
            memory=service.memory,
            cpu=service.cpu,
        )
        self._get_instances(node.node_id).append(inst)

        # Allocate memory on the node for this container
        mem_req = ResourceProfile(cpu=0.0, memory=service.memory)
        node.allocate(mem_req)

        # Find path from null to target_state
        path = self.sm.find_path("null", target_state)
        if path is None:
            path = ["null", target_state]  # fallback

        # Walk the path, applying transitions
        for i in range(len(path) - 1):
            from_state = path[i]
            to_state = path[i + 1]
            td = self.sm.get_transition(from_state, to_state)

            inst.state = from_state
            inst.target_state = to_state
            inst.state_entered_at = self.ctx.env.now

            if td:
                # Allocate transition resources
                if td.transition_cpu > 0 or td.transition_memory > 0:
                    trans_res = ResourceProfile(cpu=td.transition_cpu, memory=td.transition_memory)
                    node.allocate(trans_res)

                # Wait for transition
                if td.transition_time > 0:
                    yield self.ctx.env.timeout(td.transition_time)

                # Release transition resources
                if td.transition_cpu > 0 or td.transition_memory > 0:
                    node.release(trans_res)

        inst.state = target_state
        inst.target_state = None
        inst.state_entered_at = self.ctx.env.now
        inst.service_id = service_id

        self.logger.debug(
            "t=%.3f | COLD_START | %s for %s on %s (%.3fs, path=%s)",
            self.ctx.env.now,
            inst.instance_id,
            service_id,
            node.node_id,
            self.ctx.env.now - inst.created_at,
            "→".join(path),
        )
        return inst

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def start_execution(self, instance: ContainerInstance, invocation: Invocation) -> None:
        """Mark instance as running, allocate per-request CPU."""
        instance.state = "running"
        instance.state_entered_at = self.ctx.env.now
        instance.active_requests += 1

        # Per-request CPU allocation on node
        service = self.ctx.workload_manager.services[invocation.service_id]
        per_req_cpu = service.cpu
        node = self.ctx.cluster_manager.get_node(instance.node_id)
        node.allocate(ResourceProfile(cpu=per_req_cpu, memory=0.0))
        instance.allocated_request_cpu += per_req_cpu

        invocation.execution_start_time = self.ctx.env.now
        invocation.assigned_instance_id = instance.instance_id

    def finish_execution(self, instance: ContainerInstance, invocation: Invocation) -> None:
        """Release per-request CPU, transition back to warm if idle.

        This only handles resource cleanup.  The caller is responsible
        for setting the final ``invocation.status`` and calling
        ``request_table.finalize()``.
        """
        instance.active_requests -= 1
        instance.last_used_at = self.ctx.env.now

        # Release per-request CPU
        service = self.ctx.workload_manager.services[invocation.service_id]
        per_req_cpu = service.cpu
        node = self.ctx.cluster_manager.get_node(instance.node_id)
        node.release(ResourceProfile(cpu=per_req_cpu, memory=0.0))
        instance.allocated_request_cpu -= per_req_cpu

        invocation.execution_end_time = self.ctx.env.now
        invocation.completion_time = self.ctx.env.now

        # Mark cold start on invocation
        if instance.cold_start:
            invocation.cold_start = True
            instance.cold_start = False

        # If no more active requests, go back to warm
        if instance.active_requests == 0:
            instance.state = "warm"
            instance.state_entered_at = self.ctx.env.now

    # ------------------------------------------------------------------
    # Eviction (used by autoscaler later)
    # ------------------------------------------------------------------

    def evict_instance(self, instance: ContainerInstance) -> None:
        """Evict an idle instance, releasing node memory and removing it.

        Like OpenWhisk, evicted containers are removed entirely from
        the instance list — no lingering references.
        """
        if instance.active_requests > 0:
            return  # Cannot evict active instance

        node = self.ctx.cluster_manager.get_node(instance.node_id)
        node.release(ResourceProfile(cpu=0.0, memory=instance.memory))

        # Remove from instance list (like docker rm)
        node_instances = self._get_instances(instance.node_id)
        try:
            node_instances.remove(instance)
        except ValueError:
            pass

        self._evicted_count += 1

        self.logger.debug(
            "t=%.3f | EVICT | %s from %s (total_evicted=%d)",
            self.ctx.env.now,
            instance.instance_id,
            instance.node_id,
            self._evicted_count,
        )

        # Notify autoscaler to replenish pool reactively
        if self.ctx.autoscaling_manager is not None:
            self.ctx.autoscaling_manager.notify_pool_change(
                instance.node_id, instance.service_id,
            )

    def get_instances_for_node(self, node_id: str, state: str | None = None) -> list[ContainerInstance]:
        """Get instances on a node, optionally filtered by state."""
        instances = self._get_instances(node_id)
        if state is not None:
            return [i for i in instances if i.state == state]
        return list(instances)
