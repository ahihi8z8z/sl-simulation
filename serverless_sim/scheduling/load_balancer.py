from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from serverless_sim.cluster.resource_profile import ResourceProfile

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext
    from serverless_sim.workload.invocation import Invocation


class ShardingContainerPoolBalancer:
    """Consistent-hashing load balancer inspired by OpenWhisk.

    Routes requests to nodes via a hash ring based on service_id.
    Falls back by walking the ring when the primary node lacks memory.
    Drops the request if no node has capacity.
    """

    def __init__(self, ctx: SimContext):
        self.ctx = ctx
        self.logger = ctx.logger
        self._nodes = ctx.cluster_manager.get_enabled_nodes()
        self._hash_cache: dict[str, int] = {}  # service_id → hash()

    def dispatch(self, invocation: Invocation) -> bool:
        """Dispatch an invocation to a node queue.

        Returns True if dispatched, False if dropped (no capacity).
        """
        if not self._nodes:
            self._drop(invocation, "no_nodes")
            return False

        # Refresh enabled nodes each dispatch (nodes may be disabled)
        self._nodes = self.ctx.cluster_manager.get_enabled_nodes()
        if not self._nodes:
            self._drop(invocation, "no_nodes")
            return False

        n = len(self._nodes)
        primary_idx = self._hash_to_index(invocation.service_id, n)

        # Find the peak memory requirement for this service
        service = self.ctx.workload_manager.services.get(invocation.service_id)
        mem_required = service.peak_memory if service else 0.0
        resource_req = ResourceProfile(cpu=0.0, memory=mem_required)

        # Try primary, then walk the ring
        for offset in range(n):
            idx = (primary_idx + offset) % n
            node = self._nodes[idx]
            if node.queue_is_full:
                continue
            if node.available.can_fit(resource_req):
                invocation.assigned_node_id = node.node_id
                invocation.dispatch_time = self.ctx.env.now
                invocation.queue_enter_time = self.ctx.env.now
                invocation.status = "queued"
                node.queue.put(invocation)
                self.logger.debug(
                    "t=%.3f | DISPATCH | %s → %s (offset=%d)",
                    self.ctx.env.now,
                    invocation.request_id,
                    node.node_id,
                    offset,
                )
                return True

        # No node has capacity
        self._drop(invocation, "no_capacity")
        return False

    def _drop(self, invocation: Invocation, reason: str) -> None:
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

    def _hash_to_index(self, service_id: str, n: int) -> int:
        """Consistent hash of service_id to a node index (cached)."""
        h = self._hash_cache.get(service_id)
        if h is None:
            h = hash(service_id)
            self._hash_cache[service_id] = h
        return h % n
