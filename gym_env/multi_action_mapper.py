"""Action mapper for MultiDiscrete / flat Discrete / continuous-softmax action spaces.

Discrete layout (softmax=False): per service, one dim per pool state + one dim
for idle_timeout (minutes). Used with MultiDiscrete or flattened Discrete.
  action = [pool_prewarm, pool_warm, ..., idle_timeout_min] per service

Softmax layout (softmax=True, continuous only): per service,
  [total, logit_1, ..., logit_N, idle_timeout]
where `total` ∈ [-1, 1] is scaled to [0, pool_target_max], logits feed a
softmax to get proportions (split total across pool states), and idle_timeout
is scaled to seconds. Pool counts are integerized via largest-remainder so
sum(pool_counts) == round(total) exactly — enforcing the shared memory budget
by construction rather than hoping the policy learns it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from serverless_sim.autoscaling.autoscaling_api import AutoscalingAPI


class MultiActionMapper:
    """Maps action array to autoscaling API calls.

    Parameters
    ----------
    service_ids : list[str]
        Service IDs.
    pool_states : dict[str, list[str]]
        Per-service pool states, e.g. {"svc-a": ["prewarm", "code_loaded"]}.
    pool_target_max : int
        Max total containers per service (default 10). In discrete mode this
        caps each pool state individually; in softmax mode it caps the sum.
    idle_timeout_max_minutes : int
        Max idle timeout in minutes (default 10).
    delta_max : int
        If > 0 and softmax=False, pool_target actions are deltas clamped to
        [-delta_max, +delta_max] (ignored in softmax mode).
    softmax : bool
        If True, use continuous softmax layout (see module docstring).
    """

    def __init__(
        self,
        service_ids: list[str],
        pool_states: dict[str, list[str]],
        pool_target_max: int = 10,
        idle_timeout_max_minutes: int = 10,
        delta_max: int = 0,
        softmax: bool = False,
    ):
        self.service_ids = service_ids
        self.pool_states = pool_states
        self.pool_target_max = pool_target_max
        self.idle_timeout_max_minutes = idle_timeout_max_minutes
        self.delta_max = delta_max
        self.softmax = softmax

        if softmax:
            # Layout per service: [total, (logits if N>1), idle_timeout].
            # N=1 skips logits — softmax of 1 element is always [1.0], so a
            # lone logit is a dead action dim that just wastes exploration.
            self._softmax_layout: list[tuple[str, list[str]]] = []
            n_dims = 0
            for svc_id in service_ids:
                states = pool_states.get(svc_id, ["prewarm"])
                self._softmax_layout.append((svc_id, states))
                n = len(states)
                n_dims += 1 + (n if n > 1 else 0) + 1
            self.n_dims = n_dims
        else:
            # Discrete layout: [pool_state_1, ..., idle_timeout] per service
            self.dimensions: list[int] = []
            self._action_map: list[tuple[str, str, str | None]] = []
            for svc_id in service_ids:
                states = pool_states.get(svc_id, ["prewarm"])
                for state in states:
                    self.dimensions.append(pool_target_max + 1)
                    self._action_map.append((svc_id, "pool_target", state))
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

    def apply(self, action, api: AutoscalingAPI) -> None:
        """Apply action through the autoscaling API."""
        if self.softmax:
            self._apply_softmax(action, api)
            return

        if isinstance(action, (int, np.integer)):
            action = self.unflatten(int(action))
        else:
            action = np.asarray(action).flatten()

        pool_updates: dict[str, dict[str, int]] = {}
        for i, (svc_id, action_type, state) in enumerate(self._action_map):
            raw = float(action[i]) if i < len(action) else 0

            if action_type == "pool_target":
                if self.delta_max > 0:
                    delta = int(round(np.clip(raw, -self.delta_max, self.delta_max)))
                    current = api.get_pool_target(svc_id, state)
                    value = max(0, min(current + delta, self.pool_target_max))
                else:
                    value = max(0, min(int(round(raw)), self.pool_target_max))
                pool_updates.setdefault(svc_id, {})[state] = value
            elif action_type == "idle_timeout":
                api.set_idle_timeout(svc_id, int(raw) * 60.0)

        for svc_id, targets in pool_updates.items():
            api.batch_set_pool_targets(svc_id, targets)

    def _apply_softmax(self, action, api: AutoscalingAPI) -> None:
        """Softmax allocation: action is Box(-1,1)^n_dims, laid out per service as
        [total, (logits if N>1), idle_timeout]."""
        raw = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)
        idx = 0
        for svc_id, states in self._softmax_layout:
            n = len(states)
            total_raw = float(raw[idx])
            idx += 1
            if n > 1:
                props = _softmax(raw[idx : idx + n])
                idx += n
            else:
                props = np.array([1.0])
            timeout_raw = float(raw[idx])
            idx += 1

            total = int(round((total_raw + 1.0) * 0.5 * self.pool_target_max))
            counts = _largest_remainder(total, props)
            targets = {state: int(counts[i]) for i, state in enumerate(states)}
            api.batch_set_pool_targets(svc_id, targets)

            timeout_sec = (timeout_raw + 1.0) * 0.5 * self.idle_timeout_max_minutes * 60.0
            api.set_idle_timeout(svc_id, max(0.0, timeout_sec))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def _largest_remainder(total: int, props: np.ndarray) -> np.ndarray:
    """Integerize `total * props` so counts sum to exactly `total`, by giving
    +1 to the entries with largest fractional part."""
    n = len(props)
    if total <= 0:
        return np.zeros(n, dtype=int)
    raw = total * props.astype(np.float64)
    floors = np.floor(raw).astype(int)
    remainder = total - int(floors.sum())
    if remainder > 0:
        fracs = raw - floors
        order = np.argsort(-fracs)
        for i in order[:remainder]:
            floors[i] += 1
    return floors
