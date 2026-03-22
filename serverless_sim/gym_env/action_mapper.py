from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.autoscaling.autoscaling_api import AutoscalingAPI


class ActionMapper:
    """Maps discrete action index to autoscaling API calls.

    Default action space for each service:
        0 = no-op
        1 = increase pool_target (first pool state) by 1
        2 = decrease pool_target (first pool state) by 1
        3 = increase idle_timeout by 5s
        4 = decrease idle_timeout by 5s

    For N services, total_actions = N * 5
        action // 5 -> service index
        action % 5 -> local action
    """

    ACTIONS_PER_SERVICE = 5

    def __init__(
        self,
        service_ids: list[str],
        pool_target_min: int = 0,
        pool_target_max: int = 10,
        idle_timeout_min: float = 5.0,
        idle_timeout_max: float = 120.0,
        idle_timeout_step: float = 5.0,
        # Backward-compat aliases
        prewarm_min: int | None = None,
        prewarm_max: int | None = None,
    ):
        self.service_ids = service_ids
        self.n_actions = len(service_ids) * self.ACTIONS_PER_SERVICE
        self.pool_target_min = prewarm_min if prewarm_min is not None else pool_target_min
        self.pool_target_max = prewarm_max if prewarm_max is not None else pool_target_max
        self.idle_timeout_min = idle_timeout_min
        self.idle_timeout_max = idle_timeout_max
        self.idle_timeout_step = idle_timeout_step
        # First pool state determined at apply time from autoscaler
        self._first_pool_state: str | None = None

    def _get_first_pool_state(self, api: AutoscalingAPI) -> str:
        """Get the first pool state from the autoscaler."""
        if self._first_pool_state is None:
            pool_states = api._autoscaler._pool_states
            self._first_pool_state = pool_states[0] if pool_states else "prewarm"
        return self._first_pool_state

    def apply(self, action: int, api: AutoscalingAPI) -> None:
        """Apply a discrete action through the autoscaling API."""
        svc_idx = action // self.ACTIONS_PER_SERVICE
        local_action = action % self.ACTIONS_PER_SERVICE

        if svc_idx >= len(self.service_ids):
            return  # invalid action, no-op

        svc_id = self.service_ids[svc_idx]

        if local_action == 0:
            pass  # no-op
        elif local_action == 1:
            state = self._get_first_pool_state(api)
            cur = api.get_pool_target(svc_id, state)
            api.set_pool_target(svc_id, state, min(cur + 1, self.pool_target_max))
        elif local_action == 2:
            state = self._get_first_pool_state(api)
            cur = api.get_pool_target(svc_id, state)
            api.set_pool_target(svc_id, state, max(cur - 1, self.pool_target_min))
        elif local_action == 3:
            cur = api.get_idle_timeout(svc_id)
            api.set_idle_timeout(svc_id, min(cur + self.idle_timeout_step, self.idle_timeout_max))
        elif local_action == 4:
            cur = api.get_idle_timeout(svc_id)
            api.set_idle_timeout(svc_id, max(cur - self.idle_timeout_step, self.idle_timeout_min))
