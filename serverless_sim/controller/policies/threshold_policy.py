from __future__ import annotations

from typing import TYPE_CHECKING

from serverless_sim.controller.policies.base_policy import BaseControlPolicy

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class ThresholdPolicy(BaseControlPolicy):
    """Rule-based policy: adjust pool_target/idle_timeout based on CPU thresholds.

    Rules:
    - CPU utilization > cpu_high -> increase pool_target for first pool state (up to max)
    - CPU utilization < cpu_low -> decrease pool_target for first pool state (down to min)
    - CPU utilization > cpu_high -> decrease idle_timeout (make eviction faster)
    - CPU utilization < cpu_low -> increase idle_timeout (keep containers longer)
    """

    def __init__(
        self,
        cpu_high: float = 0.8,
        cpu_low: float = 0.3,
        pool_target_min: int = 0,
        pool_target_max: int = 10,
        pool_target_step: int = 1,
        idle_timeout_min: float = 5.0,
        idle_timeout_max: float = 120.0,
        idle_timeout_step: float = 5.0,
        # Backward-compat aliases
        prewarm_min: int | None = None,
        prewarm_max: int | None = None,
        prewarm_step: int | None = None,
    ):
        self.cpu_high = cpu_high
        self.cpu_low = cpu_low
        self.pool_target_min = prewarm_min if prewarm_min is not None else pool_target_min
        self.pool_target_max = prewarm_max if prewarm_max is not None else pool_target_max
        self.pool_target_step = prewarm_step if prewarm_step is not None else pool_target_step
        self.idle_timeout_min = idle_timeout_min
        self.idle_timeout_max = idle_timeout_max
        self.idle_timeout_step = idle_timeout_step

    def decide(self, snapshot: dict, ctx: SimContext) -> list[dict]:
        actions = []
        cpu_util = snapshot.get("cluster.cpu_utilization", 0.0)

        if ctx.autoscaling_manager is None:
            return actions

        # Determine the first pool state for pool_target adjustments
        pool_states = ctx.autoscaling_manager._pool_states
        first_state = pool_states[0] if pool_states else "prewarm"

        for svc_id in ctx.workload_manager.services:
            current_pool_target = ctx.autoscaling_manager.get_pool_target(svc_id, first_state)
            current_idle = ctx.autoscaling_manager.get_idle_timeout(svc_id)

            if cpu_util > self.cpu_high:
                # High load: increase pool_target, decrease idle timeout
                new_pool_target = min(current_pool_target + self.pool_target_step, self.pool_target_max)
                new_idle = max(current_idle - self.idle_timeout_step, self.idle_timeout_min)
            elif cpu_util < self.cpu_low:
                # Low load: decrease pool_target, increase idle timeout
                new_pool_target = max(current_pool_target - self.pool_target_step, self.pool_target_min)
                new_idle = min(current_idle + self.idle_timeout_step, self.idle_timeout_max)
            else:
                continue

            if new_pool_target != current_pool_target:
                actions.append({
                    "action": "set_pool_target",
                    "service_id": svc_id,
                    "state": first_state,
                    "value": new_pool_target,
                })
            if new_idle != current_idle:
                actions.append({
                    "action": "set_idle_timeout",
                    "service_id": svc_id,
                    "value": new_idle,
                })

        return actions
