"""Pluggable load balancer strategies.

All strategies implement ``BaseLoadBalancer.dispatch()`` which routes
an invocation to a node queue or drops it.

Available strategies:
- ``HashRingBalancer`` — consistent hash ring (OpenWhisk-style, default)
- ``RoundRobinBalancer`` — cycle through nodes
- ``LeastLoadedBalancer`` — pick node with most available memory
- ``PowerOfTwoChoicesBalancer`` — random 2 nodes, pick least loaded (Knative-style)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from serverless_sim.cluster.resource_profile import ResourceProfile

if TYPE_CHECKING:
    from serverless_sim.cluster.node import Node
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.workload.invocation import Invocation


class BaseLoadBalancer:
    """Interface for load balancing strategies."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.logger = ctx.logger

    def dispatch(self, invocation: Invocation) -> bool:
        """Route invocation to a node. Returns True if dispatched, False if dropped."""
        raise NotImplementedError

    def _get_enabled_nodes(self) -> list[Node]:
        """Refresh and return enabled nodes."""
        return self.ctx.cluster_manager.get_enabled_nodes()

    def _can_accept(self, node: Node, mem_required: float) -> bool:
        """Check if node can accept a request (queue not full + enough memory)."""
        if node.queue_is_full:
            return False
        resource_req = ResourceProfile(cpu=0.0, memory=mem_required)
        return node.available.can_fit(resource_req)

    def _get_mem_required(self, invocation: Invocation) -> float:
        """Get peak memory requirement for the invocation's service."""
        service = self.ctx.workload_manager.services.get(invocation.service_id)
        return service.peak_memory if service else 0.0

    def _assign(self, invocation: Invocation, node: Node) -> None:
        """Assign invocation to node and put it in the queue."""
        invocation.assigned_node_id = node.node_id
        invocation.dispatch_time = self.ctx.env.now
        invocation.queue_enter_time = self.ctx.env.now
        invocation.status = "queued"
        node.queue.put(invocation)

    def _drop(self, invocation: Invocation, reason: str) -> None:
        """Drop an invocation."""
        invocation.dropped = True
        invocation.drop_reason = reason
        invocation.status = "dropped"
        invocation.completion_time = self.ctx.env.now
        self.ctx.request_table.finalize(invocation)
        self.logger.debug(
            "t=%.3f | DROP | %s reason=%s",
            self.ctx.env.now,
            invocation.request_id,
            reason,
        )


class HashRingBalancer(BaseLoadBalancer):
    """Consistent-hashing load balancer (OpenWhisk-style).

    Hash ``service_id`` to pick a primary node, then walk the ring
    looking for a node with capacity.
    """

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        self._hash_cache: dict[str, int] = {}

    def dispatch(self, invocation: Invocation) -> bool:
        nodes = self._get_enabled_nodes()
        if not nodes:
            self._drop(invocation, "no_nodes")
            return False

        n = len(nodes)
        primary_idx = self._hash_to_index(invocation.service_id, n)
        mem_required = self._get_mem_required(invocation)

        for offset in range(n):
            idx = (primary_idx + offset) % n
            node = nodes[idx]
            if self._can_accept(node, mem_required):
                self._assign(invocation, node)
                self.logger.debug(
                    "t=%.3f | DISPATCH | %s → %s (hash, offset=%d)",
                    self.ctx.env.now, invocation.request_id, node.node_id, offset,
                )
                return True

        self._drop(invocation, "no_capacity")
        return False

    def _hash_to_index(self, service_id: str, n: int) -> int:
        h = self._hash_cache.get(service_id)
        if h is None:
            # Use hashlib for deterministic hashing (Python hash() varies per run)
            import hashlib
            h = int(hashlib.md5(service_id.encode()).hexdigest(), 16)
            self._hash_cache[service_id] = h
        return h % n


# Backward-compatible alias
ShardingContainerPoolBalancer = HashRingBalancer


class RoundRobinBalancer(BaseLoadBalancer):
    """Simple round-robin across enabled nodes."""

    def __init__(self, ctx: SimContext):
        super().__init__(ctx)
        self._counter = 0

    def dispatch(self, invocation: Invocation) -> bool:
        nodes = self._get_enabled_nodes()
        if not nodes:
            self._drop(invocation, "no_nodes")
            return False

        n = len(nodes)
        mem_required = self._get_mem_required(invocation)

        for offset in range(n):
            idx = (self._counter + offset) % n
            node = nodes[idx]
            if self._can_accept(node, mem_required):
                self._assign(invocation, node)
                self._counter = (idx + 1) % n
                self.logger.debug(
                    "t=%.3f | DISPATCH | %s → %s (round_robin)",
                    self.ctx.env.now, invocation.request_id, node.node_id,
                )
                return True

        self._drop(invocation, "no_capacity")
        return False


class LeastLoadedBalancer(BaseLoadBalancer):
    """Pick the node with the most available memory."""

    def dispatch(self, invocation: Invocation) -> bool:
        nodes = self._get_enabled_nodes()
        if not nodes:
            self._drop(invocation, "no_nodes")
            return False

        mem_required = self._get_mem_required(invocation)

        # Filter nodes that can accept, pick the one with most available memory
        candidates = [n for n in nodes if self._can_accept(n, mem_required)]
        if not candidates:
            self._drop(invocation, "no_capacity")
            return False

        best = max(candidates, key=lambda n: n.available.memory)
        self._assign(invocation, best)
        self.logger.debug(
            "t=%.3f | DISPATCH | %s → %s (least_loaded, avail_mem=%.0f)",
            self.ctx.env.now, invocation.request_id, best.node_id, best.available.memory,
        )
        return True


class PowerOfTwoChoicesBalancer(BaseLoadBalancer):
    """Random 2 nodes, pick the one with shorter queue (Knative-style).

    Falls back to scanning all nodes if both random choices are full.
    """

    def dispatch(self, invocation: Invocation) -> bool:
        nodes = self._get_enabled_nodes()
        if not nodes:
            self._drop(invocation, "no_nodes")
            return False

        mem_required = self._get_mem_required(invocation)
        n = len(nodes)

        if n == 1:
            if self._can_accept(nodes[0], mem_required):
                self._assign(invocation, nodes[0])
                return True
            self._drop(invocation, "no_capacity")
            return False

        # Pick 2 random nodes
        rng = self.ctx.rng
        indices = rng.choice(n, size=min(2, n), replace=False)
        candidates = [nodes[i] for i in indices]

        # Filter acceptable candidates
        acceptable = [nd for nd in candidates if self._can_accept(nd, mem_required)]

        if acceptable:
            # Pick the one with shorter queue
            best = min(acceptable, key=lambda nd: nd.queue_depth)
            self._assign(invocation, best)
            self.logger.debug(
                "t=%.3f | DISPATCH | %s → %s (p2c, queue=%d)",
                self.ctx.env.now, invocation.request_id, best.node_id, best.queue_depth,
            )
            return True

        # Fallback: scan all nodes
        for node in nodes:
            if self._can_accept(node, mem_required):
                self._assign(invocation, node)
                self.logger.debug(
                    "t=%.3f | DISPATCH | %s → %s (p2c_fallback)",
                    self.ctx.env.now, invocation.request_id, node.node_id,
                )
                return True

        self._drop(invocation, "no_capacity")
        return False


# Registry for config-driven selection
LOAD_BALANCER_REGISTRY: dict[str, type[BaseLoadBalancer]] = {
    "hash_ring": HashRingBalancer,
    "round_robin": RoundRobinBalancer,
    "least_loaded": LeastLoadedBalancer,
    "power_of_two_choices": PowerOfTwoChoicesBalancer,
}


def create_load_balancer(ctx: SimContext) -> BaseLoadBalancer:
    """Create a load balancer from config.

    Config key: ``scheduling.strategy`` (default: ``"hash_ring"``).
    """
    strategy = ctx.config.get("scheduling", {}).get("strategy", "hash_ring")
    cls = LOAD_BALANCER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown scheduling strategy '{strategy}'. "
            f"Available: {sorted(LOAD_BALANCER_REGISTRY.keys())}"
        )
    return cls(ctx)
