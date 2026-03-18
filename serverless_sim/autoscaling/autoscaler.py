from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.lifecycle.container_instance import ContainerInstance

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class OpenWhiskPoolAutoscaler:
    """Memory-bounded pool autoscaler with LRU eviction and idle timeout.

    Periodic reconcile loop:
    1. Evict idle containers past idle_timeout
    2. Evict LRU idle containers when node is over memory
    3. Top-up prewarm containers to target count
    """

    def __init__(self, ctx: SimContext, reconcile_interval: float = 5.0):
        self.ctx = ctx
        self.logger = ctx.logger
        self.reconcile_interval = reconcile_interval

        # Per-service parameters (can be adjusted by controller/RL)
        self._idle_timeout: dict[str, float] = {}
        self._prewarm_count: dict[str, int] = {}

        # Initialize from config
        for svc in ctx.workload_manager.services.values():
            self._idle_timeout[svc.service_id] = svc.idle_timeout
            self._prewarm_count[svc.service_id] = svc.prewarm_count

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
                if inst.state in ("warm", "prewarm") and inst.is_idle:
                    timeout = self._idle_timeout.get(inst.service_id, 60.0)
                    if (now - inst.last_used_at) >= timeout:
                        lm.evict_instance(inst)

            # 2. LRU eviction when node memory is overcommitted
            # (re-fetch after evictions above)
            instances = lm.get_instances_for_node(node.node_id)
            while node.available.memory < 0:
                # Find LRU idle stable instance to evict
                candidates = [
                    i for i in instances
                    if i.state in ("warm", "prewarm") and i.is_idle
                ]
                if not candidates:
                    break
                candidates.sort(key=lambda i: i.last_used_at)
                lm.evict_instance(candidates[0])
                instances = lm.get_instances_for_node(node.node_id)

            # 3. Prewarm top-up
            for svc_id, target in self._prewarm_count.items():
                if target <= 0:
                    continue
                # Count alive prewarm+ instances for this service on this node
                alive = [
                    i for i in lm.get_instances_for_node(node.node_id)
                    if i.service_id == svc_id and i.state not in ("null", "evicted")
                ]
                deficit = target - len(alive)
                for _ in range(deficit):
                    service = self.ctx.workload_manager.services[svc_id]
                    mem_req = ResourceProfile(cpu=0.0, memory=service.memory)
                    if node.available.can_fit(mem_req):
                        lm.prepare_instance_for_service(node, svc_id)

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_idle_timeout(self, service_id: str) -> float:
        return self._idle_timeout.get(service_id, 60.0)

    def set_idle_timeout(self, service_id: str, value: float) -> None:
        self._idle_timeout[service_id] = value

    def get_prewarm_count(self, service_id: str) -> int:
        return self._prewarm_count.get(service_id, 0)

    def set_prewarm_count(self, service_id: str, count: int) -> None:
        self._prewarm_count[service_id] = count
