"""MultiDiscrete action mapper: set pool_target per state + idle_timeout in minutes.

Action space: gymnasium.spaces.MultiDiscrete([pool_max+1, pool_max+1, ..., idle_max_min+1])
  - One dimension per pool state per service: value = target container count
  - One dimension per service for idle_timeout: value = timeout in minutes

Example with 1 service, pool states [prewarm, code_loaded], idle_max=10min:
  action = [3, 1, 5]
  → pool_target(prewarm) = 3
  → pool_target(code_loaded) = 1
  → idle_timeout = 5 * 60 = 300s
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from serverless_sim.autoscaling.autoscaling_api import AutoscalingAPI


class MultiActionMapper:
    """Maps MultiDiscrete action to autoscaling API calls.

    Parameters
    ----------
    service_ids : list[str]
        Service IDs.
    pool_states : dict[str, list[str]]
        Per-service pool states, e.g. {"svc-a": ["prewarm", "code_loaded"]}.
    pool_target_max : int
        Max containers per pool state (default 10).
    idle_timeout_max_minutes : int
        Max idle timeout in minutes (default 10).
    """

    def __init__(
        self,
        service_ids: list[str],
        pool_states: dict[str, list[str]],
        pool_target_max: int = 10,
        idle_timeout_max_minutes: int = 10,
        delta_max: int = 0,
    ):
        self.service_ids = service_ids
        self.pool_states = pool_states
        self.pool_target_max = pool_target_max
        self.idle_timeout_max_minutes = idle_timeout_max_minutes
        self.delta_max = delta_max  # 0 = absolute mode, >0 = delta mode

        # Build action dimensions: [pool_state_1, pool_state_2, ..., idle_timeout] per service
        self.dimensions = []  # list of (max_value+1,)
        self._action_map = []  # list of (service_id, type, state_or_none)

        for svc_id in service_ids:
            states = pool_states.get(svc_id, ["prewarm"])
            for state in states:
                self.dimensions.append(pool_target_max + 1)
                self._action_map.append((svc_id, "pool_target", state))
            # idle_timeout in minutes (0 to max)
            self.dimensions.append(idle_timeout_max_minutes + 1)
            self._action_map.append((svc_id, "idle_timeout", None))

        self.n_dimensions = len(self.dimensions)

    @property
    def flat_n_actions(self) -> int:
        """Total number of actions when flattened to single Discrete."""
        n = 1
        for d in self.dimensions:
            n *= d
        return n

    def unflatten(self, flat_action: int) -> np.ndarray:
        """Convert flat action index to MultiDiscrete array."""
        action = np.zeros(self.n_dimensions, dtype=int)
        for i in reversed(range(self.n_dimensions)):
            action[i] = flat_action % self.dimensions[i]
            flat_action //= self.dimensions[i]
        return action

    def apply(self, action, api: AutoscalingAPI, continuous: bool = False) -> None:
        """Apply action through the autoscaling API.

        Accepts MultiDiscrete array, flat int, or continuous Box array.
        If continuous=True, pool_targets are rounded and idle_timeout is in seconds.
        If delta_max > 0, pool_target actions are deltas added to current values.
        """
        if isinstance(action, (int, np.integer)):
            action = self.unflatten(int(action))
        else:
            action = np.asarray(action, dtype=np.float32).flatten()

        # Batch: collect all pool_target changes, apply once
        pool_updates = {}  # {svc_id: {state: value}}
        for i, (svc_id, action_type, state) in enumerate(self._action_map):
            raw = float(action[i]) if i < len(action) else 0

            if action_type == "pool_target":
                if self.delta_max > 0:
                    # Delta mode: action is delta, clamp to [-delta_max, +delta_max]
                    delta = int(round(np.clip(raw, -self.delta_max, self.delta_max)))
                    current = api.get_pool_target(svc_id, state)
                    value = max(0, min(current + delta, self.pool_target_max))
                else:
                    value = max(0, min(int(round(raw)), self.pool_target_max))
                pool_updates.setdefault(svc_id, {})[state] = value
            elif action_type == "idle_timeout":
                if continuous:
                    api.set_idle_timeout(svc_id, max(0.0, raw))
                else:
                    api.set_idle_timeout(svc_id, int(raw) * 60.0)

        # Apply pool targets in batch (single fill per service)
        for svc_id, targets in pool_updates.items():
            api.batch_set_pool_targets(svc_id, targets)
