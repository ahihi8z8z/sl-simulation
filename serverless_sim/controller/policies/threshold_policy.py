from __future__ import annotations

from typing import TYPE_CHECKING

from serverless_sim.controller.policies.base_policy import BaseControlPolicy

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class ThresholdPolicy(BaseControlPolicy):
    """Rule-based policy: adjust prewarm/idle_timeout based on CPU thresholds.

    Rules:
    - CPU utilization > cpu_high → increase prewarm_count (up to max)
    - CPU utilization < cpu_low → decrease prewarm_count (down to min)
    - CPU utilization > cpu_high → decrease idle_timeout (make eviction faster)
    - CPU utilization < cpu_low → increase idle_timeout (keep containers longer)
    """

    def __init__(
        self,
        cpu_high: float = 0.8,
        cpu_low: float = 0.3,
        prewarm_min: int = 0,
        prewarm_max: int = 10,
        prewarm_step: int = 1,
        idle_timeout_min: float = 5.0,
        idle_timeout_max: float = 120.0,
        idle_timeout_step: float = 5.0,
    ):
        self.cpu_high = cpu_high
        self.cpu_low = cpu_low
        self.prewarm_min = prewarm_min
        self.prewarm_max = prewarm_max
        self.prewarm_step = prewarm_step
        self.idle_timeout_min = idle_timeout_min
        self.idle_timeout_max = idle_timeout_max
        self.idle_timeout_step = idle_timeout_step

    def decide(self, snapshot: dict, ctx: SimContext) -> list[dict]:
        actions = []
        cpu_util = snapshot.get("cluster.cpu_utilization", 0.0)

        if ctx.autoscaling_manager is None:
            return actions

        for svc_id in ctx.workload_manager.services:
            current_prewarm = ctx.autoscaling_manager.get_prewarm_count(svc_id)
            current_idle = ctx.autoscaling_manager.get_idle_timeout(svc_id)

            if cpu_util > self.cpu_high:
                # High load: increase prewarm, decrease idle timeout
                new_prewarm = min(current_prewarm + self.prewarm_step, self.prewarm_max)
                new_idle = max(current_idle - self.idle_timeout_step, self.idle_timeout_min)
            elif cpu_util < self.cpu_low:
                # Low load: decrease prewarm, increase idle timeout
                new_prewarm = max(current_prewarm - self.prewarm_step, self.prewarm_min)
                new_idle = min(current_idle + self.idle_timeout_step, self.idle_timeout_max)
            else:
                continue

            if new_prewarm != current_prewarm:
                actions.append({
                    "action": "set_prewarm_count",
                    "service_id": svc_id,
                    "value": new_prewarm,
                })
            if new_idle != current_idle:
                actions.append({
                    "action": "set_idle_timeout",
                    "service_id": svc_id,
                    "value": new_idle,
                })

        return actions
