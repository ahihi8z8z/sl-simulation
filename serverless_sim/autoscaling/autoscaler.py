from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.lifecycle.container_instance import ContainerInstance

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class OpenWhiskPoolAutoscaler:
    """Memory-bounded pool autoscaler with per-state pool targets.

    Periodic reconcile loop:
    1. Evict idle containers past idle_timeout
    2. Evict LRU idle containers when node is over memory
    3. Top-up pool targets per state along the cold-start chain

    Each intermediate state in the cold-start chain has an independent
    pool with its own target count.  Containers are counted **exactly**
    at their current state — a warm container does NOT count toward the
    prewarm pool.

    For chain ``[null, prewarm, code_loaded, warm]``:

    - ``pool_target("svc-a", "prewarm") = 3`` → keep 3 containers
      exactly at ``prewarm``
    - ``pool_target("svc-a", "warm") = 1`` → keep 1 idle container
      exactly at ``warm``
    """

    def __init__(self, ctx: SimContext, reconcile_interval: float = 5.0):
        self.ctx = ctx
        self.logger = ctx.logger
        self.reconcile_interval = reconcile_interval

        # Evictable states from state machine (stable, not null/evicted)
        self._evictable_states = ctx.lifecycle_manager.sm.get_evictable_states()

        # Cold-start chain (excluding "null"): states from shallowest to deepest
        chain = ctx.lifecycle_manager.sm.get_cold_start_path()
        self._pool_states = [s for s in chain if s != "null"]  # e.g. ["prewarm", "code_loaded", "warm"]

        # Per-service parameters
        self._idle_timeout: dict[str, float] = {}
        # Per-service per-state pool targets: {svc_id: {state: count}}
        self._pool_targets: dict[str, dict[str, int]] = {}

        # Initialize from config
        for svc in ctx.workload_manager.services.values():
            self._idle_timeout[svc.service_id] = svc.idle_timeout
            self._pool_targets[svc.service_id] = {}

            # Per-state targets from config (explicit)
            if svc.pool_targets:
                self._pool_targets[svc.service_id].update(svc.pool_targets)

            # Backward compat: prewarm_count sets target for first pool state
            # (only if pool_targets didn't already set it)
            if svc.prewarm_count > 0:
                first_state = self._pool_states[0] if self._pool_states else "prewarm"
                self._pool_targets[svc.service_id].setdefault(first_state, svc.prewarm_count)

    def start(self) -> None:
        self.ctx.env.process(self._reconcile_loop())

    def _reconcile_loop(self):
        while True:
            yield self.ctx.env.timeout(self.reconcile_interval)
            self.reconcile()

    def reconcile(self) -> None:
        """Run one reconcile cycle across all nodes."""
        now = self.ctx.env.now
        lm = self.ctx.lifecycle_manager

        for node in self.ctx.cluster_manager.get_enabled_nodes():
            instances = lm.get_instances_for_node(node.node_id)

            # 1. Evict idle containers past idle_timeout
            for inst in instances:
                if inst.state in self._evictable_states and inst.is_idle:
                    timeout = self._idle_timeout.get(inst.service_id, 60.0)
                    if (now - inst.last_used_at) >= timeout:
                        lm.evict_instance(inst)

            # 2. LRU eviction when node memory is overcommitted
            instances = lm.get_instances_for_node(node.node_id)
            while node.available.memory < 0:
                candidates = [
                    i for i in instances
                    if i.state in self._evictable_states and i.is_idle
                ]
                if not candidates:
                    break
                candidates.sort(key=lambda i: i.last_used_at)
                lm.evict_instance(candidates[0])
                instances = lm.get_instances_for_node(node.node_id)

            # 3. Per-state pool top-up
            # Each pool is independent — count containers exactly at that state.
            # Walk from shallowest to deepest so cheaper containers are created first.
            for svc_id, targets in self._pool_targets.items():
                for state in self._pool_states:
                    target = targets.get(state, 0)
                    if target <= 0:
                        continue

                    # Count instances exactly at this state
                    at_state = [
                        i for i in lm.get_instances_for_node(node.node_id)
                        if i.service_id == svc_id and i.state == state
                    ]
                    deficit = target - len(at_state)
                    for _ in range(deficit):
                        service = self.ctx.workload_manager.services[svc_id]
                        mem_req = ResourceProfile(cpu=0.0, memory=service.memory)
                        if node.available.can_fit(mem_req):
                            lm.prepare_instance_for_service(node, svc_id, target_state=state)

    # ------------------------------------------------------------------
    # Per-state pool target access
    # ------------------------------------------------------------------

    def get_pool_target(self, service_id: str, state: str) -> int:
        """Get pool target for a specific state."""
        return self._pool_targets.get(service_id, {}).get(state, 0)

    def set_pool_target(self, service_id: str, state: str, count: int) -> None:
        """Set pool target for a specific state."""
        self._pool_targets.setdefault(service_id, {})[state] = count

    def get_all_pool_targets(self, service_id: str) -> dict[str, int]:
        """Get all pool targets for a service."""
        return dict(self._pool_targets.get(service_id, {}))

    # ------------------------------------------------------------------
    # Backward-compatible prewarm_count (alias for first pool state)
    # ------------------------------------------------------------------

    def get_prewarm_count(self, service_id: str) -> int:
        first = self._pool_states[0] if self._pool_states else "prewarm"
        return self.get_pool_target(service_id, first)

    def set_prewarm_count(self, service_id: str, count: int) -> None:
        first = self._pool_states[0] if self._pool_states else "prewarm"
        self.set_pool_target(service_id, first, count)

    # ------------------------------------------------------------------
    # Idle timeout
    # ------------------------------------------------------------------

    def get_idle_timeout(self, service_id: str) -> float:
        return self._idle_timeout.get(service_id, 60.0)

    def set_idle_timeout(self, service_id: str, value: float) -> None:
        self._idle_timeout[service_id] = value
