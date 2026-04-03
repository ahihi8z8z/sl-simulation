from __future__ import annotations


class RewardCalculator:
    """Computes reward from ratios and resource-per-request metrics.

    Reward = -w_drop * drop_ratio
           - w_cold * cold_start_ratio
           - w_mem  * mem_per_request
           - w_cpu  * cpu_per_request

    All weights are positive. Penalties: lower resource usage = higher reward.
    """

    def __init__(
        self,
        drop_penalty: float = 1.0,
        cold_start_penalty: float = 1.0,
        mem_per_request_penalty: float = 0.001,
        cpu_per_request_penalty: float = 0.01,
        # Backward-compat: ignored
        memory_efficiency_reward: float = 0.0,
        cpu_efficiency_reward: float = 0.0,
        latency_penalty: float = 0.0,
        resource_penalty: float = 0.0,
        throughput_reward: float = 0.0,
    ):
        self.drop_penalty = abs(drop_penalty)
        self.cold_start_penalty = abs(cold_start_penalty)
        self.mem_per_request_penalty = abs(mem_per_request_penalty)
        self.cpu_per_request_penalty = abs(cpu_per_request_penalty)

        # Previous cumulative values for computing deltas
        self._prev_total = 0.0
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_cold_starts = 0.0
        self._prev_latency_sum = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_total_cpu_sec = 0.0

    def reset(self) -> None:
        self._prev_total = 0.0
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_cold_starts = 0.0
        self._prev_latency_sum = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_total_cpu_sec = 0.0

    def compute(self, snapshot: dict) -> float:
        """Compute reward from current metric snapshot."""
        total = snapshot.get("request.total", 0.0)
        completed = snapshot.get("request.completed", 0.0)
        dropped = snapshot.get("request.dropped", 0.0)
        cold_starts = snapshot.get("request.cold_starts", 0.0)
        latency_mean = snapshot.get("request.latency_mean", 0.0)
        total_mem_sec = snapshot.get("lifecycle.total_memory_seconds", 0.0)
        total_cpu_sec = snapshot.get("lifecycle.total_cpu_seconds", 0.0)

        # Deltas
        d_total = total - self._prev_total
        d_completed = completed - self._prev_completed
        d_dropped = dropped - self._prev_dropped
        d_cold = cold_starts - self._prev_cold_starts

        # Per-step latency
        latency_sum_now = latency_mean * completed
        d_latency_sum = latency_sum_now - self._prev_latency_sum
        step_latency_mean = (d_latency_sum / d_completed) if d_completed > 0 else 0.0

        d_total_mem = total_mem_sec - self._prev_total_mem_sec
        d_total_cpu = total_cpu_sec - self._prev_total_cpu_sec

        self._prev_total = total
        self._prev_completed = completed
        self._prev_dropped = dropped
        self._prev_cold_starts = cold_starts
        self._prev_latency_sum = latency_sum_now
        self._prev_total_mem_sec = total_mem_sec
        self._prev_total_cpu_sec = total_cpu_sec

        # Ratios
        drop_ratio = d_dropped / max(d_total, 1.0)
        cold_ratio = d_cold / max(d_total, 1.0)

        # Resource per request
        mem_per_req = (d_total_mem / d_completed) if d_completed > 0 else 0.0
        cpu_per_req = (d_total_cpu / d_completed) if d_completed > 0 else 0.0

        reward = (
            - self.drop_penalty * drop_ratio
            - self.cold_start_penalty * cold_ratio
            - self.mem_per_request_penalty * mem_per_req
            - self.cpu_per_request_penalty * cpu_per_req
        )

        self.last_components = {
            "drop_ratio": drop_ratio,
            "cold_start_ratio": cold_ratio,
            "latency_mean": step_latency_mean,
            "mem_per_request": mem_per_req,
            "cpu_per_request": cpu_per_req,
            "d_total": d_total,
            "d_completed": d_completed,
            "d_cold": d_cold,
            "d_dropped": d_dropped,
        }

        return float(reward)
