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

    Pool targets apply only to intermediate states between ``null``
    and ``warm`` (exclusive).  Warm containers are created naturally
    by request processing and kept alive by idle_timeout — like
    OpenWhisk, there is no limit on warm container count (only
    bounded by node memory).

    For chain ``[null, prewarm, code_loaded, warm]``:

    - ``pool_target("svc-a", "prewarm") = 3`` → keep 3 stem-cell
      containers at ``prewarm``
    - ``pool_target("svc-a", "code_loaded") = 1`` → keep 1 container
      pre-warmed to ``code_loaded``
    - ``warm`` is NOT a valid pool target state
    """

    def __init__(self, ctx: SimContext, reconcile_interval: float = 5.0):
        self.ctx = ctx
        self.logger = ctx.logger
        self.reconcile_interval = reconcile_interval

        # Evictable states from state machine (stable, not null/evicted)
        self._evictable_states = ctx.lifecycle_manager.sm.get_evictable_states()

        # Pool states: intermediate states between null and warm (exclusive)
        # Warm containers are created on-demand by requests, not by pool top-up.
        chain = ctx.lifecycle_manager.sm.get_cold_start_path()
        self._pool_states = [s for s in chain if s not in ("null", "warm")]  # e.g. ["prewarm", "code_loaded"]

        # Per-service parameters
        self._idle_timeout: dict[str, float] = {}
        # Per-service per-state pool targets: {svc_id: {state: count}}
        self._pool_targets: dict[str, dict[str, int]] = {}

        # Initialize from config
        for svc in ctx.workload_manager.services.values():
            self._idle_timeout[svc.service_id] = svc.idle_timeout
            self._pool_targets[svc.service_id] = {}

            # Per-state targets from config (only intermediate states)
            if svc.pool_targets:
                for state, count in svc.pool_targets.items():
                    if state in self._pool_states:
                        self._pool_targets[svc.service_id][state] = count

            # Backward compat: prewarm_count sets target for first pool state
            # (only if pool_targets didn't already set it)
            if svc.prewarm_count > 0:
                first_state = self._pool_states[0] if self._pool_states else "prewarm"
                self._pool_targets[svc.service_id].setdefault(first_state, svc.prewarm_count)

    def start(self) -> None:
        self.initial_fill()
        self.ctx.env.process(self._reconcile_loop())

    def _reconcile_loop(self):
        while True:
            yield self.ctx.env.timeout(self.reconcile_interval)
            self.reconcile()

    def reconcile(self) -> None:
        """Run one reconcile cycle across all nodes.

        Reconcile handles eviction only (idle timeout + LRU).
        Pool top-up is reactive — triggered immediately when an instance
        is consumed or evicted via ``notify_pool_change()``.
        """
        now = self.ctx.env.now
        lm = self.ctx.lifecycle_manager

        for node in self.ctx.cluster_manager.get_enabled_nodes():
            instances = lm.get_instances_for_node(node.node_id)

            # 1. Evict idle containers past idle_timeout
            #    Skip pool containers that have never served a request —
            #    like OpenWhisk, prewarm stem cells persist until consumed.
            for inst in instances:
                if inst.state in self._evictable_states and inst.is_idle and inst.has_served:
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

    # ------------------------------------------------------------------
    # Reactive pool top-up
    # ------------------------------------------------------------------

    def notify_pool_change(self, node_id: str, service_id: str) -> None:
        """Reactively replenish pool targets for *service_id* on *node_id*.

        Called when an instance is consumed (moved to warm/running) or
        evicted.  Checks each pool state and creates replacement
        instances if below target.
        """
        targets = self._pool_targets.get(service_id, {})
        if not targets:
            return

        lm = self.ctx.lifecycle_manager
        node = self.ctx.cluster_manager.get_node(node_id)

        for state in self._pool_states:
            target = targets.get(state, 0)
            if target <= 0:
                continue

            at_state = [
                i for i in lm.get_instances_for_node(node_id)
                if i.service_id == service_id and i.state == state
            ]
            deficit = target - len(at_state)
            for _ in range(deficit):
                service = self.ctx.workload_manager.services[service_id]
                mem_req = ResourceProfile(cpu=0.0, memory=service.memory)
                if node.available.can_fit(mem_req):
                    lm.prepare_instance_for_service(node, service_id, target_state=state)

    def initial_fill(self) -> None:
        """Fill all pools to their targets on startup."""
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for svc_id in self._pool_targets:
                self.notify_pool_change(node.node_id, svc_id)

    # ------------------------------------------------------------------
    # Per-state pool target access
    # ------------------------------------------------------------------

    def get_pool_target(self, service_id: str, state: str) -> int:
        """Get pool target for a specific state."""
        return self._pool_targets.get(service_id, {}).get(state, 0)

    def set_pool_target(self, service_id: str, state: str, count: int) -> None:
        """Set pool target for a specific state (must be an intermediate state).

        Triggers reactive top-up on all nodes immediately.
        """
        if state not in self._pool_states:
            self.logger.warning(
                "Ignoring pool target for '%s' — only intermediate states %s are valid",
                state, self._pool_states,
            )
            return
        self._pool_targets.setdefault(service_id, {})[state] = count
        # Reactively fill on all nodes
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            self.notify_pool_change(node.node_id, service_id)

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
