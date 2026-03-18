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

        # node_id → list of ContainerInstance
        self.instances: dict[str, list[ContainerInstance]] = {}

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
        self, node: Node, service_id: str
    ) -> simpy.events.Event:
        """Create a new instance and transition it to warm (cold start).

        Returns a SimPy process that yields until the instance is ready.
        """
        return self.ctx.env.process(self._cold_start(node, service_id))

    def _cold_start(self, node: Node, service_id: str):
        """SimPy process: null → prewarm → warm."""
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

        # Transition null → prewarm
        t1 = self.sm.get_transition("null", "prewarm")
        if t1 and t1.transition_time > 0:
            inst.state = "prewarm"
            inst.state_entered_at = self.ctx.env.now
            yield self.ctx.env.timeout(t1.transition_time)

        # Transition prewarm → warm
        t2 = self.sm.get_transition("prewarm", "warm")
        if t2 and t2.transition_time > 0:
            inst.state = "prewarm"
            inst.state_entered_at = self.ctx.env.now
            yield self.ctx.env.timeout(t2.transition_time)

        inst.state = "warm"
        inst.state_entered_at = self.ctx.env.now
        inst.service_id = service_id

        self.logger.debug(
            "t=%.3f | COLD_START | %s for %s on %s (%.3fs)",
            self.ctx.env.now,
            inst.instance_id,
            service_id,
            node.node_id,
            self.ctx.env.now - inst.created_at,
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
        """Release per-request CPU, transition back to warm if idle."""
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

        invocation.status = "completed"

        # If no more active requests, go back to warm
        if instance.active_requests == 0:
            instance.state = "warm"
            instance.state_entered_at = self.ctx.env.now

    # ------------------------------------------------------------------
    # Eviction (used by autoscaler later)
    # ------------------------------------------------------------------

    def evict_instance(self, instance: ContainerInstance) -> None:
        """Evict an idle instance, releasing node memory."""
        if instance.active_requests > 0:
            return  # Cannot evict active instance

        node = self.ctx.cluster_manager.get_node(instance.node_id)
        node.release(ResourceProfile(cpu=0.0, memory=instance.memory))

        instance.state = "evicted"
        instance.state_entered_at = self.ctx.env.now

        self.logger.debug(
            "t=%.3f | EVICT | %s from %s",
            self.ctx.env.now,
            instance.instance_id,
            instance.node_id,
        )

    def get_instances_for_node(self, node_id: str, state: str | None = None) -> list[ContainerInstance]:
        """Get instances on a node, optionally filtered by state."""
        instances = self._get_instances(node_id)
        if state is not None:
            return [i for i in instances if i.state == state]
        return list(instances)
