from __future__ import annotations


class RewardCalculator:
    """Computes reward as weighted penalty sum.

    Penalties (all non-positive contributions):
    - drop_penalty * number_of_new_drops
    - timeout_penalty * number_of_new_timeouts
    - cold_start_penalty * number_of_new_cold_starts
    - latency_penalty * mean_latency (if available)
    - resource_penalty * cpu_utilization (to penalize waste)

    Reward per step:
    - throughput_reward * number_of_new_completions
    """

    def __init__(
        self,
        drop_penalty: float = -1.0,
        timeout_penalty: float = -1.0,
        cold_start_penalty: float = -0.1,
        latency_penalty: float = -0.5,
        resource_penalty: float = -0.1,
        throughput_reward: float = 0.1,
    ):
        self.drop_penalty = drop_penalty
        self.timeout_penalty = timeout_penalty
        self.cold_start_penalty = cold_start_penalty
        self.latency_penalty = latency_penalty
        self.resource_penalty = resource_penalty
        self.throughput_reward = throughput_reward

        # Previous values for computing deltas
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_timed_out = 0.0
        self._prev_cold_starts = 0.0

    def reset(self) -> None:
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_timed_out = 0.0
        self._prev_cold_starts = 0.0

    def compute(self, snapshot: dict) -> float:
        """Compute reward from current metric snapshot."""
        completed = snapshot.get("request.completed", 0.0)
        dropped = snapshot.get("request.dropped", 0.0)
        timed_out = snapshot.get("request.timed_out", 0.0)
        cold_starts = snapshot.get("request.cold_starts", 0.0)
        cpu_util = snapshot.get("cluster.cpu_utilization", 0.0)
        latency = snapshot.get("request.latency_mean", 0.0)

        # Deltas since last step
        d_completed = completed - self._prev_completed
        d_dropped = dropped - self._prev_dropped
        d_timed_out = timed_out - self._prev_timed_out
        d_cold_starts = cold_starts - self._prev_cold_starts

        # Update previous values
        self._prev_completed = completed
        self._prev_dropped = dropped
        self._prev_timed_out = timed_out
        self._prev_cold_starts = cold_starts

        reward = (
            self.throughput_reward * d_completed
            + self.drop_penalty * d_dropped
            + self.timeout_penalty * d_timed_out
            + self.cold_start_penalty * d_cold_starts
            + self.latency_penalty * latency
            + self.resource_penalty * cpu_util
        )

        return float(reward)
