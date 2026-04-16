from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class OpenWhiskPoolAutoscaler:
    """Persistent-pool autoscaler with min/max instances and per-state pool targets.

    Pool containers are persistent — they are NOT evicted by idle_timeout.
    After serving, they return to their pool_state. Only explicit API calls
    (set_pool_target decrease) can remove pool containers.

    Demand containers (created on-the-fly by request dispatch) ARE subject
    to idle_timeout eviction.

    Periodic reconcile loop:
    1. Evict idle demand containers past idle_timeout (respects min_instances)
    2. Demote idle pool containers past idle_timeout back to pool_state
    3. LRU eviction when node is over memory (prefers demand containers first)
    """

    def __init__(self, ctx: SimContext, pool_mode: str = "per_node", **kwargs):
        self.ctx = ctx
        self.logger = ctx.logger
        self.pool_mode = pool_mode

        # Per-service min/max instances (from service config)
        self._min_instances: dict[str, int] = {}
        self._max_instances: dict[str, int] = {}

        # Per-service parameters (controller/policy-managed)
        self._idle_timeout: dict[str, float] = {}
        self._pool_targets: dict[str, dict[str, int]] = {}

        # Pending instances: scheduled but not yet created by SimPy
        self._pending: dict[str, int] = {}

        # Initialize from config
        for svc in ctx.workload_manager.services.values():
            svc_cfg = self._find_service_config(svc.service_id)
            defaults = svc_cfg.get("autoscaling_defaults", {}) if svc_cfg else {}

            self._min_instances[svc.service_id] = svc.min_instances
            self._max_instances[svc.service_id] = svc.max_instances
            self._idle_timeout[svc.service_id] = defaults.get("idle_timeout", 60.0)
            self._pool_targets[svc.service_id] = dict(defaults.get("pool_targets", {}))

    def _find_service_config(self, service_id: str) -> dict | None:
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
        sm = self.ctx.workload_manager.services[service_id].state_machine
        return sm.get_evictable_states()

    # ------------------------------------------------------------------
    # Start / reconcile
    # ------------------------------------------------------------------

    def start(self) -> None:
        for svc_id in self.ctx.workload_manager.services:
            # Fill min_instances as warm pool containers
            min_inst = self._min_instances.get(svc_id, 0)
            if min_inst > 0:
                self._fill_pool_to_target_count(svc_id, "warm", min_inst, pool_state="warm")
            # Fill pool_targets from config
            for state, target in self._pool_targets.get(svc_id, {}).items():
                if target > 0:
                    self._fill_pool_to_target(svc_id, state)

    def handle_idle_timeout(self, instance) -> None:
        """Called by lifecycle manager when idle timer fires.

        Demand containers → evict. Pool containers → demote to pool_state.
        """
        lm = self.ctx.lifecycle_manager
        if not instance.is_idle or instance.evicted:
            return
        if instance.is_pool_container:
            if instance.pool_state != instance.state:
                lm.demote_to_pool_state(instance)
        else:
            lm.evict_instance(instance)

    # ------------------------------------------------------------------
    # Pool fill (only on explicit API calls)
    # ------------------------------------------------------------------

    def _fill_pool_to_target(self, service_id: str, state: str) -> None:
        """Fill pool containers for a specific state up to its target.

        Counts by pool_state field (not current state) to handle in-flight containers.
        """
        target = self._pool_targets.get(service_id, {}).get(state, 0)
        current = self._count_pool_containers(service_id, state)
        deficit = target - current
        if deficit <= 0:
            return
        self._fill_pool_to_target_count(service_id, state, deficit, pool_state=state)

    def _get_state_priority(self, service_id: str, state: str) -> int:
        """Higher priority for states closer to warm in the cold-start chain.

        demand (None) = 0, null = 1, prewarm = 2, ..., warm = highest.
        """
        sm = self.ctx.workload_manager.services[service_id].state_machine
        chain = sm.get_cold_start_path()  # ["null", "prewarm", "warm"]
        if state in chain:
            return chain.index(state)
        return 0

    def _evict_lower_priority_for_budget(self, service_id: str, target_state: str,
                                          needed: int) -> int:
        """Evict lower-priority idle containers to make room for higher-priority pool.

        Returns number of slots freed.
        """
        target_prio = self._get_state_priority(service_id, target_state)
        lm = self.ctx.lifecycle_manager
        freed = 0

        # Collect candidates: demand containers and pool containers with lower priority
        candidates = []
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in lm.get_instances_for_node(node.node_id):
                if inst.service_id != service_id or not inst.is_idle:
                    continue
                if not inst.is_pool_container:
                    # Demand container — lowest priority
                    candidates.append((0, inst))
                elif self._get_state_priority(service_id, inst.pool_state) < target_prio:
                    # Lower-priority pool container
                    candidates.append((self._get_state_priority(service_id, inst.pool_state), inst))

        # Sort: lowest priority first
        candidates.sort(key=lambda x: (x[0], x[1].created_at))

        for _, inst in candidates:
            if freed >= needed:
                break
            lm.evict_instance(inst)
            freed += 1
            # Update pool target for evicted pool containers
            if inst.is_pool_container:
                old_target = self._pool_targets.get(service_id, {}).get(inst.pool_state, 0)
                if old_target > 0:
                    current_count = self._count_pool_containers(service_id, inst.pool_state)
                    # Reduce target to match current count (don't refill evicted)
                    self._pool_targets[service_id][inst.pool_state] = min(old_target, current_count)

        return freed

    def _fill_pool_to_target_count(self, service_id: str, target_state: str,
                                    count: int, pool_state: str | None = None) -> None:
        """Create exactly `count` containers with given pool_state.

        If max_instances budget is exhausted, evicts lower-priority containers
        (demand first, then lower pool states) to make room.
        """
        lm = self.ctx.lifecycle_manager
        max_inst = self._max_instances.get(service_id, 0)

        def _total():
            return self._count_total_instances(service_id) + self._pending.get(service_id, 0)

        def _has_budget():
            if max_inst <= 0:
                return True
            return _total() < max_inst

        # Try to free slots if needed
        if max_inst > 0 and (_total() + count) > max_inst:
            needed = min(count, _total() + count - max_inst)
            self._evict_lower_priority_for_budget(service_id, target_state, needed)

        if self.pool_mode == "global":
            placement = self.ctx.placement_strategy
            nodes = self.ctx.cluster_manager.get_enabled_nodes()
            for _ in range(count):
                if not _has_budget():
                    break
                node = placement.select_node(nodes, service_id, self.ctx)
                if node is None:
                    break
                self._pending[service_id] = self._pending.get(service_id, 0) + 1
                lm.prepare_instance_for_service(node, service_id,
                                                 target_state=target_state,
                                                 pool_state=pool_state)
        else:
            nodes = self.ctx.cluster_manager.get_enabled_nodes()
            service = self.ctx.workload_manager.services[service_id]
            for _ in range(count):
                if not _has_budget():
                    break
                # Round-robin across nodes that have flavor capacity
                placed = False
                for node in nodes:
                    if node.can_fit_flavor(service.peak_cpu, service.peak_memory):
                        self._pending[service_id] = self._pending.get(service_id, 0) + 1
                        lm.prepare_instance_for_service(node, service_id,
                                                         target_state=target_state,
                                                         pool_state=pool_state)
                        placed = True
                        break
                if not placed:
                    break

    # ------------------------------------------------------------------
    # Instance counting helpers
    # ------------------------------------------------------------------

    def _count_total_instances(self, service_id: str) -> int:
        """Count all instances for service across all nodes."""
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
        """Count warm + running instances (provisioned capacity)."""
        count = 0
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in self.ctx.lifecycle_manager.get_instances_for_node(node.node_id):
                if inst.service_id == service_id and inst.state in ("warm", "running"):
                    count += 1
        return count

    def _count_pool_containers(self, service_id: str, pool_state: str) -> int:
        """Count containers with matching pool_state (regardless of current state)."""
        count = 0
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in self.ctx.lifecycle_manager.get_instances_for_node(node.node_id):
                if inst.service_id == service_id and inst.pool_state == pool_state:
                    count += 1
        return count

    def _count_demand_containers(self, service_id: str) -> int:
        """Count demand containers (pool_state is None)."""
        count = 0
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in self.ctx.lifecycle_manager.get_instances_for_node(node.node_id):
                if inst.service_id == service_id and inst.pool_state is None:
                    count += 1
        return count

    # ------------------------------------------------------------------
    # min_instances / max_instances
    # ------------------------------------------------------------------

    def get_min_instances(self, service_id: str) -> int:
        return self._min_instances.get(service_id, 0)

    def set_min_instances(self, service_id: str, value: int) -> None:
        old = self._min_instances.get(service_id, 0)
        self._min_instances[service_id] = value
        if value < old:
            self._evict_excess_warm(service_id)
        elif value > old:
            # Fill new warm pool containers up to new min
            deficit = value - self._count_alive_instances(service_id)
            if deficit > 0:
                self._fill_pool_to_target_count(service_id, "warm", deficit, pool_state="warm")

    def get_max_instances(self, service_id: str) -> int:
        return self._max_instances.get(service_id, 0)

    def set_max_instances(self, service_id: str, value: int) -> None:
        self._max_instances[service_id] = value

    # ------------------------------------------------------------------
    # Per-state pool target access
    # ------------------------------------------------------------------

    def get_pool_target(self, service_id: str, state: str) -> int:
        return self._pool_targets.get(service_id, {}).get(state, 0)

    def set_pool_target(self, service_id: str, state: str, count: int) -> None:
        """Set pool target. Increase → create pool containers. Decrease → evict excess."""
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
            self._fill_pool_to_target(service_id, state)

    def batch_set_pool_targets(self, service_id: str, targets: dict[str, int]) -> None:
        """Set multiple pool targets at once, then fill/evict."""
        need_fill_states = []
        need_evict_states = []

        for state, count in targets.items():
            pool_states = self._get_pool_states(service_id)
            valid_states = set(pool_states) | {"warm"}
            if state not in valid_states:
                continue
            old = self._pool_targets.get(service_id, {}).get(state, 0)
            self._pool_targets.setdefault(service_id, {})[state] = count
            if count > old:
                need_fill_states.append(state)
            elif count < old:
                need_evict_states.append(state)

        for state in need_evict_states:
            self._evict_excess_pool(service_id, state)
        for state in need_fill_states:
            self._fill_pool_to_target(service_id, state)

    def get_all_pool_targets(self, service_id: str) -> dict[str, int]:
        return dict(self._pool_targets.get(service_id, {}))

    # ------------------------------------------------------------------
    # Excess eviction (when targets decrease)
    # ------------------------------------------------------------------

    def _evict_excess_pool(self, service_id: str, state: str) -> None:
        """Evict excess pool containers when target decreases.

        Matches by pool_state field. Running containers get demoted to
        demand (pool_state=None) so they'll be evicted after idle_timeout.
        """
        target = self._pool_targets.get(service_id, {}).get(state, 0)
        lm = self.ctx.lifecycle_manager

        candidates = []
        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in lm.get_instances_for_node(node.node_id):
                if inst.service_id == service_id and inst.pool_state == state:
                    candidates.append(inst)

        excess = len(candidates) - target
        if excess <= 0:
            return

        # Evict idle first, then demote running to demand
        candidates.sort(key=lambda i: (i.active_requests > 0, i.created_at))
        for inst in candidates[:excess]:
            if inst.is_idle:
                lm.evict_instance(inst)
            else:
                # Running: demote to demand, will be evicted after idle_timeout
                inst.pool_state = None

    def _evict_excess_warm(self, service_id: str) -> None:
        """Evict excess idle warm containers when min_instances decreases."""
        min_inst = self._min_instances.get(service_id, 0)
        lm = self.ctx.lifecycle_manager

        while True:
            alive = self._count_alive_instances(service_id)
            if alive <= min_inst:
                break
            candidate = None
            for node in self.ctx.cluster_manager.get_enabled_nodes():
                for inst in lm.get_instances_for_node(node.node_id):
                    if (inst.service_id == service_id
                            and inst.state == "warm" and inst.is_idle):
                        if candidate is None or inst.last_used_at < candidate.last_used_at:
                            candidate = inst
            if candidate is None:
                break
            lm.evict_instance(candidate)

    # ------------------------------------------------------------------
    # Idle timeout
    # ------------------------------------------------------------------

    def get_idle_timeout(self, service_id: str) -> float:
        return self._idle_timeout.get(service_id, 60.0)

    def set_idle_timeout(self, service_id: str, value: float) -> None:
        self._idle_timeout[service_id] = value
