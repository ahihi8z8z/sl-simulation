from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import simpy

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.lifecycle.container_instance import ContainerInstance

if TYPE_CHECKING:
    from serverless_sim.cluster.node import Node
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.lifecycle.state_machine import OpenWhiskExtendedStateMachine
    from serverless_sim.workload.invocation import Invocation


class LifecycleManager:
    """Manages container instance lifecycle on all nodes.

    Each service has its own state machine (lifecycle profile).  Resource
    accounting is state-based: entering a state allocates that state's
    cpu/memory on the node; leaving it releases them.
    """

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.logger = ctx.logger

        # node_id → list of ContainerInstance (only alive instances)
        self.instances: dict[str, list[ContainerInstance]] = {}
        self._evicted_count: int = 0

        # Cumulative resource-seconds (for effective ratio)
        self.total_cpu_seconds: float = 0.0
        self.total_memory_seconds: float = 0.0
        self.running_cpu_seconds: float = 0.0
        self.running_memory_seconds: float = 0.0

    def _get_sm(self, service_id: str) -> OpenWhiskExtendedStateMachine:
        """Get the state machine for a service."""
        return self.ctx.workload_manager.services[service_id].state_machine

    def _get_instances(self, node_id: str) -> list[ContainerInstance]:
        return self.instances.setdefault(node_id, [])

    def _state_resources(self, service_id: str, state: str) -> ResourceProfile:
        """Get the resource profile for a service in a given state.

        If the state machine has a state_resource_model, sample from it.
        Otherwise use the fixed values from StateDefinition.
        """
        sm = self._get_sm(service_id)
        if sm.state_resource_model is not None:
            sample = sm.state_resource_model.sample(state, self.ctx.rng)
            return ResourceProfile(cpu=sample.cpu, memory=sample.memory)
        sd = sm.get_state(state)
        if sd is None:
            return ResourceProfile(cpu=0.0, memory=0.0)
        return ResourceProfile(cpu=sd.cpu, memory=sd.memory)

    # ------------------------------------------------------------------
    # Find / prepare instances
    # ------------------------------------------------------------------

    def find_reusable_instance(self, node: Node, service_id: str) -> ContainerInstance | None:
        """Find a reusable instance for this service with available slots.

        Prefers warm instances (no transition cost) but also considers running
        instances that still have free concurrency slots (max_concurrency > 1).
        """
        running_candidate: ContainerInstance | None = None
        for inst in self._get_instances(node.node_id):
            if inst.service_id != service_id:
                continue
            has_free_slot = inst.active_requests < inst.max_concurrency
            if not has_free_slot:
                continue
            if inst.state == "warm":
                return inst
            if inst.state == "running" and running_candidate is None:
                running_candidate = inst
        return running_candidate

    def find_promotable_instance(self, node: Node, service_id: str) -> ContainerInstance | None:
        """Find the deepest intermediate instance that can be promoted to warm."""
        sm = self._get_sm(service_id)
        chain = sm.get_cold_start_path()
        intermediate = chain[1:-1]  # e.g. ["prewarm", "code_loaded"]
        if not intermediate:
            return None

        # Single scan — find deepest state match
        intermediate_set = set(intermediate)
        best: ContainerInstance | None = None
        best_depth = -1
        for inst in self._get_instances(node.node_id):
            if (inst.service_id == service_id
                    and inst.state in intermediate_set
                    and inst.target_state is None
                    and inst.is_idle):
                depth = intermediate.index(inst.state)
                if depth > best_depth:
                    best = inst
                    best_depth = depth
        return best

    def find_reusable_or_promotable(self, node: Node, service_id: str
                                     ) -> tuple[ContainerInstance | None, ContainerInstance | None]:
        """Single-scan combined search for reusable and promotable instances."""
        sm = self._get_sm(service_id)
        chain = sm.get_cold_start_path()
        intermediate = chain[1:-1]
        intermediate_set = set(intermediate)

        reusable: ContainerInstance | None = None
        running_candidate: ContainerInstance | None = None
        promotable: ContainerInstance | None = None
        best_depth = -1

        for inst in self._get_instances(node.node_id):
            if inst.service_id != service_id:
                continue
            # Reusable check
            if inst.active_requests < inst.max_concurrency:
                if inst.state == "warm":
                    reusable = inst
                    break  # Best possible — stop early
                if inst.state == "running" and running_candidate is None:
                    running_candidate = inst
            # Promotable check
            if (inst.state in intermediate_set
                    and inst.target_state is None
                    and inst.is_idle):
                depth = intermediate.index(inst.state)
                if depth > best_depth:
                    promotable = inst
                    best_depth = depth

        return (reusable or running_candidate, promotable)

    def promote_instance(
        self, node: Node, instance: ContainerInstance,
    ) -> simpy.events.Event:
        """Promote an intermediate instance to warm."""
        return self.ctx.env.process(self._promote(node, instance))

    def _promote(self, node: Node, instance: ContainerInstance):
        """SimPy process: transition instance from current state → warm."""
        from_state = instance.state
        sm = self._get_sm(instance.service_id)
        path = sm.find_path(from_state, "warm")
        if path is None or len(path) < 2:
            self._transition_state(node, instance, "warm")
            return instance

        rng = self.ctx.rng

        for i in range(len(path) - 1):
            s_from = path[i]
            s_to = path[i + 1]

            instance.target_state = s_to

            sample = sm.transition_model.sample(s_from, s_to, rng)
            if sample.cpu > 0 or sample.memory > 0:
                trans_res = ResourceProfile(cpu=sample.cpu, memory=sample.memory)
                node.allocate(trans_res)
            if sample.time > 0:
                yield self.ctx.env.timeout(sample.time)
            if sample.cpu > 0 or sample.memory > 0:
                node.release(trans_res)

            self._transition_state(node, instance, s_to)

        self.logger.debug(
            "t=%.3f | PROMOTE | %s %s→warm on %s",
            self.ctx.env.now, instance.instance_id, from_state, node.node_id,
        )

        return instance

    def prepare_instance_for_service(
        self, node: Node, service_id: str, target_state: str = "warm",
        pool_state: str | None = None,
    ) -> simpy.events.Event:
        """Create a new instance and transition it to *target_state*."""
        return self.ctx.env.process(self._cold_start(node, service_id, target_state, pool_state))

    def _cold_start(self, node: Node, service_id: str, target_state: str = "warm",
                     pool_state: str | None = None):
        """SimPy process: follow chain from null → target_state."""
        service = self.ctx.workload_manager.services[service_id]
        sm = self._get_sm(service_id)

        inst = ContainerInstance(
            env=self.ctx.env,
            service_id=service_id,
            node_id=node.node_id,
            max_concurrency=service.max_concurrency,
        )
        inst.target_state = target_state  # Set immediately for eviction matching
        inst.pool_state = pool_state
        self._get_instances(node.node_id).append(inst)

        # Allocate null state resources (typically 0)
        null_res = self._state_resources(service_id, "null")
        if null_res.cpu > 0 or null_res.memory > 0:
            node.allocate(null_res)
        inst.allocated_cpu = null_res.cpu
        inst.allocated_memory = null_res.memory

        # Find path from null to target_state
        path = sm.find_path("null", target_state)
        if path is None:
            path = ["null", target_state]

        # Walk the path, applying transitions and state resource changes
        rng = self.ctx.rng
        for i in range(len(path) - 1):
            from_state = path[i]
            to_state = path[i + 1]

            # Check if evicted while in-flight
            if inst.evicted:
                return inst

            inst.state = from_state
            inst.target_state = to_state
            inst.state_entered_at = self.ctx.env.now

            # Sample transition parameters from the model
            sample = sm.transition_model.sample(from_state, to_state, rng)

            if sample.cpu > 0 or sample.memory > 0:
                trans_res = ResourceProfile(cpu=sample.cpu, memory=sample.memory)
                node.allocate(trans_res)

            if sample.time > 0:
                yield self.ctx.env.timeout(sample.time)

            # Re-check after yield — may have been evicted during wait
            if inst.evicted:
                if sample.cpu > 0 or sample.memory > 0:
                    node.release(trans_res)
                return inst

            if sample.cpu > 0 or sample.memory > 0:
                node.release(trans_res)

            # Transition state resources: release old, allocate new
            self._transition_state(node, inst, to_state)

        inst.target_state = None
        inst.service_id = service_id

        self.logger.debug(
            "t=%.3f | COLD_START | %s for %s on %s (%.3fs, path=%s)",
            self.ctx.env.now, inst.instance_id, service_id, node.node_id,
            self.ctx.env.now - inst.created_at, "→".join(path),
        )
        return inst

    def _transition_state(
        self, node: Node, instance: ContainerInstance, new_state: str,
    ) -> None:
        """Change instance state and adjust node resources accordingly.

        Releases exactly what was previously allocated (tracked on instance),
        then samples and allocates new state resources.
        """
        # Accumulate resource-seconds for old state
        time_in_state = self.ctx.env.now - instance.state_entered_at
        if time_in_state > 0:
            self.total_cpu_seconds += time_in_state * instance.allocated_cpu
            self.total_memory_seconds += time_in_state * instance.allocated_memory
            if instance.state == "running":
                self.running_cpu_seconds += time_in_state * instance.allocated_cpu
                self.running_memory_seconds += time_in_state * instance.allocated_memory

        # Release what was actually allocated (not re-sampled)
        if instance.allocated_cpu > 0 or instance.allocated_memory > 0:
            node.release(ResourceProfile(cpu=instance.allocated_cpu, memory=instance.allocated_memory))

        # Sample and allocate new state resources
        new_res = self._state_resources(instance.service_id, new_state)
        if new_res.cpu > 0 or new_res.memory > 0:
            node.allocate(new_res)

        # Track what we allocated
        instance.allocated_cpu = new_res.cpu
        instance.allocated_memory = new_res.memory

        instance.state = new_state
        instance.target_state = None
        instance.state_entered_at = self.ctx.env.now

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def start_execution(self, instance: ContainerInstance, invocation: Invocation) -> None:
        """Mark instance as running, transition resources warm → running."""
        node = self.ctx.cluster_manager.get_node(instance.node_id)

        # Only transition resources on first concurrent request
        if instance.active_requests == 0:
            self._transition_state(node, instance, "running")

        instance.active_requests += 1
        invocation.execution_start_time = self.ctx.env.now
        invocation.assigned_instance_id = instance.instance_id

    def finish_execution(self, instance: ContainerInstance, invocation: Invocation) -> None:
        """Release execution, transition back to warm if idle."""
        instance.active_requests -= 1
        instance.last_used_at = self.ctx.env.now

        invocation.execution_end_time = self.ctx.env.now
        invocation.completion_time = self.ctx.env.now

        # Mark cold start on invocation
        if instance.cold_start:
            invocation.cold_start = True
            instance.cold_start = False

        # If no more active requests, go back to warm
        if instance.active_requests == 0:
            node = self.ctx.cluster_manager.get_node(instance.node_id)
            self._transition_state(node, instance, "warm")

    # ------------------------------------------------------------------
    # Pool demotion
    # ------------------------------------------------------------------

    def demote_to_pool_state(self, instance: ContainerInstance) -> None:
        """Demote a pool container from warm back to its pool_state.

        Releases warm-level resources and allocates pool_state-level resources.
        Instant transition (no time cost).
        """
        if instance.pool_state is None or instance.pool_state == instance.state:
            return
        node = self.ctx.cluster_manager.get_node(instance.node_id)
        self._transition_state(node, instance, instance.pool_state)
        self.logger.debug(
            "t=%.3f | DEMOTE | %s warm→%s on %s",
            self.ctx.env.now, instance.instance_id, instance.pool_state, node.node_id,
        )

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_instance(self, instance: ContainerInstance) -> None:
        """Evict instance, releasing state resources and removing it."""
        instance.evicted = True
        if instance.active_requests > 0:
            return

        node = self.ctx.cluster_manager.get_node(instance.node_id)

        # Accumulate resource-seconds for final state
        time_in_state = self.ctx.env.now - instance.state_entered_at
        if time_in_state > 0:
            self.total_cpu_seconds += time_in_state * instance.allocated_cpu
            self.total_memory_seconds += time_in_state * instance.allocated_memory
            if instance.state == "running":
                self.running_cpu_seconds += time_in_state * instance.allocated_cpu
                self.running_memory_seconds += time_in_state * instance.allocated_memory

        # Release what was actually allocated (tracked on instance)
        if instance.allocated_cpu > 0 or instance.allocated_memory > 0:
            node.release(ResourceProfile(cpu=instance.allocated_cpu, memory=instance.allocated_memory))
            instance.allocated_cpu = 0.0
            instance.allocated_memory = 0.0

        # Remove from instance list
        node_instances = self._get_instances(instance.node_id)
        try:
            node_instances.remove(instance)
        except ValueError:
            pass

        self._evicted_count += 1

        self.logger.debug(
            "t=%.3f | EVICT | %s from %s (total_evicted=%d)",
            self.ctx.env.now, instance.instance_id, instance.node_id, self._evicted_count,
        )

    def get_instances_for_node(self, node_id: str, state: str | None = None) -> list[ContainerInstance]:
        """Get instances on a node, optionally filtered by state."""
        instances = self._get_instances(node_id)
        if state is not None:
            return [i for i in instances if i.state == state]
        return list(instances)
