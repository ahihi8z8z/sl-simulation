from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from serverless_sim.monitoring.monitor_api import MonitorAPI

if TYPE_CHECKING:
    from serverless_sim.controller.policies.base_policy import BaseControlPolicy
    from serverless_sim.core.simulation.sim_context import SimContext


class BaseController:
    """Periodic control loop: read monitor → call policy → apply actions."""

    def __init__(
        self,
        ctx: SimContext,
        policy: BaseControlPolicy,
        interval: float = 5.0,
    ):
        self.ctx = ctx
        self.policy = policy
        self.interval = interval
        self.logger = ctx.logger
        self._monitor_api = MonitorAPI(ctx.monitor_manager)
        self.step_count = 0

    def start(self) -> None:
        self.ctx.env.process(self._loop())

    def _loop(self):
        while True:
            yield self.ctx.env.timeout(self.interval)
            self._step()

    def _step(self) -> None:
        """One control step: collect → decide → apply."""
        self.step_count += 1

        # Collect latest metrics
        snapshot = self._monitor_api.get_snapshot()

        # Policy decides actions
        actions = self.policy.decide(snapshot, self.ctx)

        # Apply actions
        for action in actions:
            self._apply_action(action)

        if actions:
            self.logger.debug(
                "t=%.3f | CONTROLLER step=%d | %d actions applied",
                self.ctx.env.now,
                self.step_count,
                len(actions),
            )

    def _apply_action(self, action: dict) -> None:
        """Apply a single action to the autoscaler."""
        if self.ctx.autoscaling_manager is None:
            return

        act = action["action"]
        svc_id = action["service_id"]
        value = action["value"]

        if act == "set_prewarm_count":
            self.ctx.autoscaling_manager.set_prewarm_count(svc_id, value)
        elif act == "set_idle_timeout":
            self.ctx.autoscaling_manager.set_idle_timeout(svc_id, value)
        elif act == "set_pool_target":
            state = action["state"]
            self.ctx.autoscaling_manager.set_pool_target(svc_id, state, value)
