from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.autoscaling.autoscaling_api import AutoscalingAPI


class ActionMapper:
    """Maps discrete action index to autoscaling API calls.

    Default action space for each service:
        0 = no-op
        1 = increase prewarm_count by 1
        2 = decrease prewarm_count by 1
        3 = increase idle_timeout by 5s
        4 = decrease idle_timeout by 5s

    For N services, total actions = 5^N (joint action space),
    or we flatten: action_index → (service_idx, local_action).
    For simplicity we use per-service sequential layout:
        total_actions = N * 5
        action // 5 → service index
        action % 5 → local action
    """

    ACTIONS_PER_SERVICE = 5

    def __init__(
        self,
        service_ids: list[str],
        prewarm_min: int = 0,
        prewarm_max: int = 10,
        idle_timeout_min: float = 5.0,
        idle_timeout_max: float = 120.0,
        idle_timeout_step: float = 5.0,
    ):
        self.service_ids = service_ids
        self.n_actions = len(service_ids) * self.ACTIONS_PER_SERVICE
        self.prewarm_min = prewarm_min
        self.prewarm_max = prewarm_max
        self.idle_timeout_min = idle_timeout_min
        self.idle_timeout_max = idle_timeout_max
        self.idle_timeout_step = idle_timeout_step

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
            cur = api.get_prewarm_count(svc_id)
            api.set_prewarm_count(svc_id, min(cur + 1, self.prewarm_max))
        elif local_action == 2:
            cur = api.get_prewarm_count(svc_id)
            api.set_prewarm_count(svc_id, max(cur - 1, self.prewarm_min))
        elif local_action == 3:
            cur = api.get_idle_timeout(svc_id)
            api.set_idle_timeout(svc_id, min(cur + self.idle_timeout_step, self.idle_timeout_max))
        elif local_action == 4:
            cur = api.get_idle_timeout(svc_id)
            api.set_idle_timeout(svc_id, max(cur - self.idle_timeout_step, self.idle_timeout_min))
