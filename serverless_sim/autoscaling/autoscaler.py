from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.lifecycle.container_instance import ContainerInstance

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class OpenWhiskPoolAutoscaler:
    """Memory-bounded pool autoscaler with min/max instances and per-state pool targets.

    Periodic reconcile loop:
    1. Evict idle containers past idle_timeout (respects min_instances)
    2. Evict LRU idle containers when node is over memory (can violate min_instances)
    3. Top-up pool targets per state along the cold-start chain

    min_instances / max_instances are user-facing service config.
    pool_targets and idle_timeout are controller/policy-managed parameters.
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

        # Per-service min/max instances (from service config)
        self._min_instances: dict[str, int] = {}
        self._max_instances: dict[str, int] = {}

        # Per-service parameters (controller/policy-managed, not from service config)
        self._idle_timeout: dict[str, float] = {}
        # Per-service per-state pool targets: {svc_id: {state: count}}
        self._pool_targets: dict[str, dict[str, int]] = {}

        # Track creates within the same SimPy timestep to avoid double-creating.
        # Reset when env.now advances.
        self._pending_creates: dict[str, int] = {}
        self._pending_time: float = -1.0

        # Initialize from config
        for svc in ctx.workload_manager.services.values():
            self._min_instances[svc.service_id] = svc.min_instances
            self._max_instances[svc.service_id] = svc.max_instances
            self._idle_timeout[svc.service_id] = 60.0
            self._pool_targets[svc.service_id] = {}

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
        Pool top-up is reactive -- triggered immediately when an instance
        is consumed or evicted via ``notify_pool_change()``.
        """
        now = self.ctx.env.now
        lm = self.ctx.lifecycle_manager

        for node in self.ctx.cluster_manager.get_enabled_nodes():
            instances = lm.get_instances_for_node(node.node_id)

            # 1. Evict idle warm containers past idle_timeout
            #    Only warm containers are subject to idle timeout.
            #    Pool containers at intermediate states (prewarm, code_loaded, ...)
            #    persist until consumed or LRU-evicted -- like OpenWhisk stem cells.
            #    Respect min_instances: don't evict below min.
            for inst in instances:
                if inst.state == "warm" and inst.is_idle:
                    timeout = self._idle_timeout.get(inst.service_id, 60.0)
                    if (now - inst.last_used_at) >= timeout:
                        # Check min_instances before evicting (alive = warm + running)
                        alive_count = self._count_alive_instances(inst.service_id)
                        min_inst = self._min_instances.get(inst.service_id, 0)
                        if alive_count <= min_inst:
                            continue  # don't evict below min_instances
                        lm.evict_instance(inst)

            # 2. LRU eviction when node memory is overcommitted
            #    CAN violate min_instances (soft guarantee)
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
        evicted.  Respects max_instances budget, fills min_instances
        (warm) first, then pool_targets for intermediate states.
        """
        lm = self.ctx.lifecycle_manager
        node = self.ctx.cluster_manager.get_node(node_id)
        service = self.ctx.workload_manager.services[service_id]
        mem_req = ResourceProfile(cpu=0.0, memory=service.memory)

        max_inst = self._max_instances.get(service_id, 0)
        min_inst = self._min_instances.get(service_id, 0)

        # Track creates within the same SimPy timestep to avoid double-creating.
        # When env.now advances, previous pending creates have materialized.
        now = self.ctx.env.now
        if now != self._pending_time:
            self._pending_creates.clear()
            self._pending_time = now

        actual_total = self._count_total_instances(service_id)
        pending = self._pending_creates.get(service_id, 0)
        effective_total = actual_total + pending

        if max_inst > 0 and effective_total >= max_inst:
            return  # at capacity

        created = 0

        def _has_budget():
            if max_inst <= 0:
                return True
            return (effective_total + created) < max_inst

        # 1. Fill warm to min_instances first (priority)
        #    min_instances counts warm + running (provisioned capacity),
        #    not just warm idle — like AWS Provisioned Concurrency.
        #    Include pending creates to avoid double-creating.
        alive_count = self._count_alive_instances(service_id)
        warm_deficit = max(0, min_inst - alive_count - pending)

        for _ in range(warm_deficit):
            if not _has_budget():
                break
            if node.available.can_fit(mem_req):
                lm.prepare_instance_for_service(node, service_id, target_state="warm")
                created += 1

        # 2. Fill pool_targets with remaining budget
        for state in self._pool_states:
            target = self._pool_targets.get(service_id, {}).get(state, 0)
            if target <= 0:
                continue
            at_state = [
                i for i in lm.get_instances_for_node(node_id)
                if i.service_id == service_id and i.state == state
            ]
            deficit = target - len(at_state)
            for _ in range(deficit):
                if not _has_budget():
                    break
                if node.available.can_fit(mem_req):
                    lm.prepare_instance_for_service(node, service_id, target_state=state)
                    created += 1

        self._pending_creates[service_id] = self._pending_creates.get(service_id, 0) + created

    def initial_fill(self) -> None:
        """Fill all pools to their targets on startup."""
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for svc_id in self.ctx.workload_manager.services:
                self.notify_pool_change(node.node_id, svc_id)

    # ------------------------------------------------------------------
    # Instance counting helpers
    # ------------------------------------------------------------------

    def _count_total_instances(self, service_id: str) -> int:
        """Count all instances for service across all nodes (excluding evicted)."""
        total = 0
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in self.ctx.lifecycle_manager.get_instances_for_node(node.node_id):
                if inst.service_id == service_id:
                    total += 1
        return total

    def _count_warm_instances(self, service_id: str) -> int:
        """Count warm (idle) instances for service across all nodes."""
        count = 0
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in self.ctx.lifecycle_manager.get_instances_for_node(node.node_id):
                if inst.service_id == service_id and inst.state == "warm":
                    count += 1
        return count

    def _count_alive_instances(self, service_id: str) -> int:
        """Count warm + running instances for service across all nodes.

        This represents provisioned capacity — instances that exist and
        can serve requests (warm) or are currently serving (running).
        Used for min_instances enforcement.
        """
        count = 0
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in self.ctx.lifecycle_manager.get_instances_for_node(node.node_id):
                if inst.service_id == service_id and inst.state in ("warm", "running"):
                    count += 1
        return count

    # ------------------------------------------------------------------
    # min_instances / max_instances access
    # ------------------------------------------------------------------

    def get_min_instances(self, service_id: str) -> int:
        return self._min_instances.get(service_id, 0)

    def set_min_instances(self, service_id: str, value: int) -> None:
        self._min_instances[service_id] = value
        # Trigger reactive fill on all nodes
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            self.notify_pool_change(node.node_id, service_id)

    def get_max_instances(self, service_id: str) -> int:
        return self._max_instances.get(service_id, 0)

    def set_max_instances(self, service_id: str, value: int) -> None:
        self._max_instances[service_id] = value

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
                "Ignoring pool target for '%s' -- only intermediate states %s are valid",
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
    # Idle timeout
    # ------------------------------------------------------------------

    def get_idle_timeout(self, service_id: str) -> float:
        return self._idle_timeout.get(service_id, 60.0)

    def set_idle_timeout(self, service_id: str, value: float) -> None:
        self._idle_timeout[service_id] = value
