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

    def __init__(self, ctx: SimContext, reconcile_interval: float = 5.0,
                 pool_mode: str = "per_node"):
        self.ctx = ctx
        self.logger = ctx.logger
        self.reconcile_interval = reconcile_interval
        self.pool_mode = pool_mode

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

        # Initialize from config (service config + optional autoscaling_defaults)
        for svc in ctx.workload_manager.services.values():
            svc_cfg = self._find_service_config(svc.service_id)
            defaults = svc_cfg.get("autoscaling_defaults", {}) if svc_cfg else {}

            self._min_instances[svc.service_id] = svc.min_instances
            self._max_instances[svc.service_id] = svc.max_instances
            self._idle_timeout[svc.service_id] = defaults.get("idle_timeout", 60.0)
            self._pool_targets[svc.service_id] = dict(defaults.get("pool_targets", {}))

    def _find_service_config(self, service_id: str) -> dict | None:
        """Find the raw service config dict by service_id."""
        for svc_cfg in self.ctx.config.get("services", []):
            if svc_cfg.get("service_id") == service_id:
                return svc_cfg
        return None

    def _get_pool_states(self, service_id: str) -> list[str]:
        """Get intermediate states (between null and warm) for a service."""
        sm = self.ctx.workload_manager.services[service_id].state_machine
        chain = sm.get_cold_start_path()
        return [s for s in chain if s not in ("null", "warm")]

    def _get_evictable_states(self, service_id: str) -> set[str]:
        """Get evictable states for a service."""
        sm = self.ctx.workload_manager.services[service_id].state_machine
        return sm.get_evictable_states()

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
                    if timeout < 0:
                        continue  # -1 means never evict
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
                    if i.state in self._get_evictable_states(i.service_id) and i.is_idle
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
        """Reactively replenish pool targets for *service_id*.

        Called when an instance is consumed or evicted.
        In per_node mode, fills on the specific node.
        In global mode, fills across all nodes using placement strategy.
        """
        if self.pool_mode == "global":
            self._fill_pool_global(service_id)
        else:
            self._fill_pool_per_node(node_id, service_id)

    def _fill_pool_per_node(self, node_id: str, service_id: str) -> None:
        """Fill pool targets on a specific node (per_node mode)."""
        lm = self.ctx.lifecycle_manager
        node = self.ctx.cluster_manager.get_node(node_id)
        service = self.ctx.workload_manager.services[service_id]
        mem_req = ResourceProfile(cpu=0.0, memory=service.peak_memory)

        max_inst = self._max_instances.get(service_id, 0)
        min_inst = self._min_instances.get(service_id, 0)

        now = self.ctx.env.now
        if now != self._pending_time:
            self._pending_creates.clear()
            self._pending_time = now

        actual_total = self._count_total_instances(service_id)
        pending = self._pending_creates.get(service_id, 0)
        effective_total = actual_total + pending

        if max_inst > 0 and effective_total >= max_inst:
            return

        created = 0

        def _has_budget():
            if max_inst <= 0:
                return True
            return (effective_total + created) < max_inst

        alive_count = self._count_alive_instances(service_id)
        warm_deficit = max(0, min_inst - alive_count - pending)

        for _ in range(warm_deficit):
            if not _has_budget():
                break
            if node.available.can_fit(mem_req):
                lm.prepare_instance_for_service(node, service_id, target_state="warm")
                created += 1

        pool_states = self._get_pool_states(service_id) + ["warm"]
        for state in pool_states:
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

    def _fill_pool_global(self, service_id: str) -> None:
        """Fill pool targets across all nodes (global mode)."""
        lm = self.ctx.lifecycle_manager
        placement = self.ctx.placement_strategy

        max_inst = self._max_instances.get(service_id, 0)
        min_inst = self._min_instances.get(service_id, 0)

        now = self.ctx.env.now
        if now != self._pending_time:
            self._pending_creates.clear()
            self._pending_time = now

        actual_total = self._count_total_instances(service_id)
        pending = self._pending_creates.get(service_id, 0)
        effective_total = actual_total + pending

        if max_inst > 0 and effective_total >= max_inst:
            return

        created = 0
        nodes = self.ctx.cluster_manager.get_enabled_nodes()

        def _has_budget():
            if max_inst <= 0:
                return True
            return (effective_total + created) < max_inst

        # 1. Fill warm to min_instances (global count)
        alive_count = self._count_alive_instances(service_id)
        warm_deficit = max(0, min_inst - alive_count - pending)

        for _ in range(warm_deficit):
            if not _has_budget():
                break
            node = placement.select_node(nodes, service_id, self.ctx)
            if node is None:
                break
            lm.prepare_instance_for_service(node, service_id, target_state="warm")
            created += 1

        # 2. Fill pool_targets (global count across all nodes)
        pool_states = self._get_pool_states(service_id) + ["warm"]
        for state in pool_states:
            target = self._pool_targets.get(service_id, {}).get(state, 0)
            if target <= 0:
                continue
            at_state_count = sum(
                1 for n in nodes
                for i in lm.get_instances_for_node(n.node_id)
                if i.service_id == service_id and i.state == state
            )
            deficit = target - at_state_count
            for _ in range(deficit):
                if not _has_budget():
                    break
                node = placement.select_node(nodes, service_id, self.ctx)
                if node is None:
                    break
                lm.prepare_instance_for_service(node, service_id, target_state=state)
                created += 1

        self._pending_creates[service_id] = self._pending_creates.get(service_id, 0) + created

    def initial_fill(self) -> None:
        """Fill all pools to their targets on startup."""
        if self.pool_mode == "global":
            for svc_id in self.ctx.workload_manager.services:
                self._fill_pool_global(svc_id)
            return
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
        old = self._min_instances.get(service_id, 0)
        self._min_instances[service_id] = value
        if value < old:
            self._evict_excess_warm(service_id)
        elif value > old:
            if self.pool_mode == "global":
                self._fill_pool_global(service_id)
            else:
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

        If target increases, triggers reactive top-up on all nodes.
        If target decreases, evicts excess pool containers at that state.
        """
        pool_states = self._get_pool_states(service_id)
        valid_states = set(pool_states) | {"warm"}
        if state not in valid_states:
            self.logger.warning(
                "Ignoring pool target for '%s' -- only %s are valid",
                state, sorted(valid_states),
            )
            return
        old = self._pool_targets.get(service_id, {}).get(state, 0)
        self._pool_targets.setdefault(service_id, {})[state] = count
        if count < old:
            self._evict_excess_pool(service_id, state)
        elif count > old:
            if self.pool_mode == "global":
                self._fill_pool_global(service_id)
            else:
                for node in self.ctx.cluster_manager.get_enabled_nodes():
                    self.notify_pool_change(node.node_id, service_id)

    def batch_set_pool_targets(self, service_id: str, targets: dict[str, int]) -> None:
        """Set multiple pool targets at once, then fill/evict once.

        Much faster than calling set_pool_target() per state.
        """
        need_fill = False
        need_evict_states = []

        for state, count in targets.items():
            pool_states = self._get_pool_states(service_id)
            valid_states = set(pool_states) | {"warm"}
            if state not in valid_states:
                continue
            old = self._pool_targets.get(service_id, {}).get(state, 0)
            self._pool_targets.setdefault(service_id, {})[state] = count
            if count > old:
                need_fill = True
            elif count < old:
                need_evict_states.append(state)

        # Evict excess (per state)
        for state in need_evict_states:
            self._evict_excess_pool(service_id, state)

        # Fill once for all increased targets
        if need_fill:
            if self.pool_mode == "global":
                self._fill_pool_global(service_id)
            else:
                for node in self.ctx.cluster_manager.get_enabled_nodes():
                    self.notify_pool_change(node.node_id, service_id)

    def get_all_pool_targets(self, service_id: str) -> dict[str, int]:
        """Get all pool targets for a service."""
        return dict(self._pool_targets.get(service_id, {}))

    # ------------------------------------------------------------------
    # Excess eviction (when targets decrease)
    # ------------------------------------------------------------------

    def _evict_excess_pool(self, service_id: str, state: str) -> None:
        """Evict excess pool containers at/heading-to *state* when target decreases.

        Matches containers currently at the state OR transitioning towards it
        (target_state == state) to prevent accumulation from in-flight promotions.
        """
        target = self._pool_targets.get(service_id, {}).get(state, 0)
        lm = self.ctx.lifecycle_manager

        def _matches(inst):
            if inst.service_id != service_id:
                return False
            # Already at target state
            if inst.state == state:
                return True
            # In-flight: being promoted towards target state
            if inst.target_state == state:
                return True
            return False

        if self.pool_mode == "global":
            candidates = [
                i for node in self.ctx.cluster_manager.get_enabled_nodes()
                for i in lm.get_instances_for_node(node.node_id)
                if _matches(i)
            ]
            excess = len(candidates) - target
            if excess > 0:
                candidates.sort(key=lambda i: i.created_at)
                for inst in candidates[:excess]:
                    lm.evict_instance(inst)
        else:
            for node in self.ctx.cluster_manager.get_enabled_nodes():
                candidates = [
                    i for i in lm.get_instances_for_node(node.node_id)
                    if _matches(i)
                ]
                excess = len(candidates) - target
                if excess <= 0:
                    continue
                candidates.sort(key=lambda i: i.created_at)
                for inst in candidates[:excess]:
                    lm.evict_instance(inst)

    def _evict_excess_warm(self, service_id: str) -> None:
        """Evict excess idle warm containers when min_instances decreases."""
        min_inst = self._min_instances.get(service_id, 0)
        lm = self.ctx.lifecycle_manager

        while True:
            alive = self._count_alive_instances(service_id)
            if alive <= min_inst:
                break
            # Find an idle warm container to evict (oldest first)
            candidate = None
            for node in self.ctx.cluster_manager.get_enabled_nodes():
                for inst in lm.get_instances_for_node(node.node_id):
                    if (inst.service_id == service_id
                            and inst.state == "warm" and inst.is_idle):
                        if candidate is None or inst.last_used_at < candidate.last_used_at:
                            candidate = inst
            if candidate is None:
                break  # no idle warm to evict
            lm.evict_instance(candidate)

    # ------------------------------------------------------------------
    # Idle timeout
    # ------------------------------------------------------------------

    def get_idle_timeout(self, service_id: str) -> float:
        return self._idle_timeout.get(service_id, 60.0)

    def set_idle_timeout(self, service_id: str, value: float) -> None:
        self._idle_timeout[service_id] = value
