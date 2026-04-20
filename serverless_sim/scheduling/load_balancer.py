"""Pluggable load balancer strategies.

All scheduling logic is centralized here. Strategies only decide
node ordering via ``_get_candidate_nodes()``. The base class handles:
  1. Find reusable instance → execute
  2. Find promotable instance → promote → execute
  3. Check max_instances
  4. Reserve flavor → cold start → execute
  5. Drop if nothing works

Available strategies:
- ``HashRingBalancer`` — consistent hash ring (OpenWhisk-style, default)
- ``RoundRobinBalancer`` — cycle through nodes
- ``LeastLoadedBalancer`` — pick node with most flavor capacity
- ``PowerOfTwoChoicesBalancer`` — random 2 nodes, pick least loaded
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.cluster.node import Node
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.lifecycle.container_instance import ContainerInstance
    from serverless_sim.workload.invocation import Invocation


class BaseLoadBalancer:
    """Centralized request dispatcher."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.logger = ctx.logger

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _get_candidate_nodes(self, invocation: Invocation) -> list[Node]:
        """Return nodes in preference order. Subclasses implement this."""
        return self.ctx.cluster_manager.get_enabled_nodes()

    # ------------------------------------------------------------------
    # Unified dispatch (not overridden by subclasses)
    # ------------------------------------------------------------------

    def dispatch(self, invocation: Invocation) -> bool:
        """Central dispatch: reuse → promote → cold start → drop."""
        lm = self.ctx.lifecycle_manager
        if lm is None:
            self._drop(invocation, "no_lifecycle_manager")
            return False

        nodes = self._get_candidate_nodes(invocation)
        if not nodes:
            self._drop(invocation, "no_nodes")
            return False

        service_id = invocation.service_id
        service = self.ctx.workload_manager.services.get(service_id)
        if service is None:
            self._drop(invocation, "unknown_service")
            return False

        # 1. Find reusable warm/running instance with free slot
        for node in nodes:
            instance = lm.find_reusable_instance(node, service_id)
            if instance is not None:
                self._dispatch_to_instance(invocation, node, instance)
                return True

        # 2. Find promotable intermediate instance
        for node in nodes:
            promotable = lm.find_promotable_instance(node, service_id)
            if promotable is not None:
                self._dispatch_promote(invocation, node, promotable)
                return True

        # 3. Check max_instances
        if self.ctx.autoscaling_manager:
            max_inst = self.ctx.autoscaling_manager.get_max_instances(service_id)
            pending = self.ctx.autoscaling_manager._pending.get(service_id, 0)
            if max_inst > 0:
                total = self.ctx.autoscaling_manager._count_total_instances(service_id) + pending
                if total >= max_inst:
                    self._drop(invocation, "max_instances")
                    return False

        # 4. Cold start on first node with flavor capacity
        for node in nodes:
            if node.can_fit_flavor(service.peak_cpu, service.peak_memory):
                node.reserve_flavor(service.peak_cpu, service.peak_memory)
                self._dispatch_cold_start(invocation, node, service)
                return True

        self._drop(invocation, "no_capacity")
        return False

    # ------------------------------------------------------------------
    # Dispatch helpers (schedule SimPy processes)
    # ------------------------------------------------------------------

    def _dispatch_to_instance(self, invocation: Invocation, node: Node,
                              instance: ContainerInstance) -> None:
        """Dispatch to an existing reusable instance."""
        invocation.assigned_node_id = node.node_id
        invocation.dispatch_time = self.ctx.env.now
        invocation.status = "dispatched"
        self.ctx.env.process(self._execute(invocation, node, instance))

    def _dispatch_promote(self, invocation: Invocation, node: Node,
                          instance: ContainerInstance) -> None:
        """Promote intermediate instance then execute."""
        invocation.assigned_node_id = node.node_id
        invocation.dispatch_time = self.ctx.env.now
        invocation.status = "dispatched"
        self.ctx.env.process(self._promote_and_execute(invocation, node, instance))

    def _dispatch_cold_start(self, invocation: Invocation, node: Node, service) -> None:
        """Cold start a new container then execute. Flavor already reserved."""
        invocation.assigned_node_id = node.node_id
        invocation.dispatch_time = self.ctx.env.now
        invocation.status = "dispatched"
        self.ctx.env.process(self._cold_start_and_execute(invocation, node, service))

    def _drop(self, invocation: Invocation, reason: str) -> None:
        invocation.dropped = True
        invocation.drop_reason = reason
        invocation.status = "dropped"
        invocation.completion_time = self.ctx.env.now
        self.ctx.request_table.finalize(invocation)
        self.logger.debug(
            "t=%.3f | DROP | %s reason=%s",
            self.ctx.env.now, invocation.request_id, reason,
        )

    # ------------------------------------------------------------------
    # SimPy processes
    # ------------------------------------------------------------------

    def _execute(self, invocation: Invocation, node: Node,
                 instance: ContainerInstance):
        """SimPy process: acquire slot → execute → release."""
        lm = self.ctx.lifecycle_manager

        req = instance.slots.request()
        yield req

        lm.start_execution(instance, invocation)
        invocation.cold_start = False
        yield self.ctx.env.timeout(invocation.service_time)

        lm.finish_execution(instance, invocation)
        instance.slots.release(req)

        invocation.status = "completed"
        self.ctx.request_table.finalize(invocation)
        self.logger.debug(
            "t=%.3f | COMPLETED | %s on %s (reuse)",
            self.ctx.env.now, invocation.request_id, node.node_id,
        )

    def _promote_and_execute(self, invocation: Invocation, node: Node,
                             instance: ContainerInstance):
        """SimPy process: promote → acquire slot → execute → release.

        Hitting a pool-provisioned intermediate instance (e.g. prewarm) does
        NOT count as a cold start — the pool already absorbed that cost.
        """
        lm = self.ctx.lifecycle_manager

        promote_proc = lm.promote_instance(node, instance)
        instance = yield promote_proc

        req = instance.slots.request()
        yield req

        lm.start_execution(instance, invocation)
        invocation.cold_start = False
        yield self.ctx.env.timeout(invocation.service_time)

        lm.finish_execution(instance, invocation)
        instance.slots.release(req)

        invocation.status = "completed"
        self.ctx.request_table.finalize(invocation)
        self.logger.debug(
            "t=%.3f | COMPLETED | %s on %s (promoted)",
            self.ctx.env.now, invocation.request_id, node.node_id,
        )

    def _cold_start_and_execute(self, invocation: Invocation, node: Node, service):
        """SimPy process: cold start (flavor already reserved) → execute."""
        lm = self.ctx.lifecycle_manager

        cold_start_proc = lm.prepare_instance_for_service(node, invocation.service_id)
        instance = yield cold_start_proc

        req = instance.slots.request()
        yield req

        lm.start_execution(instance, invocation)
        invocation.cold_start = True
        yield self.ctx.env.timeout(invocation.service_time)

        lm.finish_execution(instance, invocation)
        instance.slots.release(req)

        invocation.status = "completed"
        self.ctx.request_table.finalize(invocation)
        self.logger.debug(
            "t=%.3f | COMPLETED | %s on %s (cold_start)",
            self.ctx.env.now, invocation.request_id, node.node_id,
        )


# ------------------------------------------------------------------
# Strategies: only implement _get_candidate_nodes
# ------------------------------------------------------------------

class HashRingBalancer(BaseLoadBalancer):
    """Consistent-hashing: hash service_id to primary, walk ring."""

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        self._hash_cache: dict[str, int] = {}

    def _get_candidate_nodes(self, invocation: Invocation) -> list[Node]:
        nodes = self.ctx.cluster_manager.get_enabled_nodes()
        if not nodes:
            return []
        n = len(nodes)
        primary_idx = self._hash_to_index(invocation.service_id, n)
        return [nodes[(primary_idx + i) % n] for i in range(n)]

    def _hash_to_index(self, service_id: str, n: int) -> int:
        h = self._hash_cache.get(service_id)
        if h is None:
            import hashlib
            h = int(hashlib.md5(service_id.encode()).hexdigest(), 16)
            self._hash_cache[service_id] = h
        return h % n


ShardingContainerPoolBalancer = HashRingBalancer


class RoundRobinBalancer(BaseLoadBalancer):
    """Round-robin across enabled nodes."""

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        self._counter = 0

    def _get_candidate_nodes(self, invocation: Invocation) -> list[Node]:
        nodes = self.ctx.cluster_manager.get_enabled_nodes()
        if not nodes:
            return []
        n = len(nodes)
        ordered = [nodes[(self._counter + i) % n] for i in range(n)]
        self._counter = (self._counter + 1) % n
        return ordered


class LeastLoadedBalancer(BaseLoadBalancer):
    """Pick node with most flavor capacity first."""

    def _get_candidate_nodes(self, invocation: Invocation) -> list[Node]:
        nodes = self.ctx.cluster_manager.get_enabled_nodes()
        return sorted(nodes, key=lambda n: n.capacity.memory - n.flavor_memory_used, reverse=True)


class PowerOfTwoChoicesBalancer(BaseLoadBalancer):
    """Random 2 nodes, prefer least loaded. Fallback to all."""

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        import numpy as np
        self._rng = np.random.default_rng(ctx.rng.spawn(1)[0])

    def _get_candidate_nodes(self, invocation: Invocation) -> list[Node]:
        nodes = self.ctx.cluster_manager.get_enabled_nodes()
        if len(nodes) <= 2:
            return nodes

        rng = self._rng
        indices = rng.choice(len(nodes), size=2, replace=False)
        picked = [nodes[i] for i in indices]
        # Sort by flavor remaining (most free first)
        picked.sort(key=lambda n: n.capacity.memory - n.flavor_memory_used, reverse=True)

        # Append remaining nodes as fallback
        picked_set = set(id(n) for n in picked)
        rest = [n for n in nodes if id(n) not in picked_set]
        return picked + rest


# Registry
LOAD_BALANCER_REGISTRY: dict[str, type[BaseLoadBalancer]] = {
    "hash_ring": HashRingBalancer,
    "round_robin": RoundRobinBalancer,
    "least_loaded": LeastLoadedBalancer,
    "power_of_two_choices": PowerOfTwoChoicesBalancer,
}


def create_load_balancer(ctx: SimContext) -> BaseLoadBalancer:
    strategy = ctx.config.get("scheduling", {}).get("strategy", "hash_ring")
    cls = LOAD_BALANCER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown scheduling strategy '{strategy}'. "
            f"Available: {sorted(LOAD_BALANCER_REGISTRY.keys())}"
        )
    return cls(ctx)
