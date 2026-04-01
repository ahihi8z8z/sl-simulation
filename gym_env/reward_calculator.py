from __future__ import annotations


class RewardCalculator:
    """Computes reward from ratios and efficiency metrics.

    Reward = -w_drop * drop_ratio
           - w_cold * cold_start_ratio
           + w_mem  * memory_efficiency
           + w_cpu  * cpu_efficiency

    All weights are positive. All components in [0, 1].
    Reward range: [-(w_drop + w_cold), (w_mem + w_cpu)].
    """

    def __init__(
        self,
        drop_penalty: float = 1.0,
        cold_start_penalty: float = 0.5,
        memory_efficiency_reward: float = 0.3,
        cpu_efficiency_reward: float = 0.2,
        # Backward-compat: ignored if set
        latency_penalty: float = 0.0,
        resource_penalty: float = 0.0,
        throughput_reward: float = 0.0,
    ):
        self.drop_penalty = abs(drop_penalty)
        self.cold_start_penalty = abs(cold_start_penalty)
        self.memory_efficiency_reward = abs(memory_efficiency_reward)
        self.cpu_efficiency_reward = abs(cpu_efficiency_reward)

        # Previous cumulative values for computing deltas
        self._prev_total = 0.0
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_cold_starts = 0.0
        self._prev_latency_sum = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_running_mem_sec = 0.0
        self._prev_total_cpu_sec = 0.0
        self._prev_running_cpu_sec = 0.0

    def reset(self) -> None:
        self._prev_total = 0.0
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_cold_starts = 0.0
        self._prev_latency_sum = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_running_mem_sec = 0.0
        self._prev_total_cpu_sec = 0.0
        self._prev_running_cpu_sec = 0.0

    def compute(self, snapshot: dict) -> float:
        """Compute reward from current metric snapshot."""
        total = snapshot.get("request.total", 0.0)
        completed = snapshot.get("request.completed", 0.0)
        dropped = snapshot.get("request.dropped", 0.0)
        cold_starts = snapshot.get("request.cold_starts", 0.0)
        latency_mean = snapshot.get("request.latency_mean", 0.0)
        total_mem_sec = snapshot.get("lifecycle.total_memory_seconds", 0.0)
        running_mem_sec = snapshot.get("lifecycle.running_memory_seconds", 0.0)
        total_cpu_sec = snapshot.get("lifecycle.total_cpu_seconds", 0.0)
        running_cpu_sec = snapshot.get("lifecycle.running_cpu_seconds", 0.0)

        # Deltas
        d_total = total - self._prev_total
        d_completed = completed - self._prev_completed
        d_dropped = dropped - self._prev_dropped
        d_cold = cold_starts - self._prev_cold_starts

        # Per-step latency: derive from cumulative mean and completed count
        latency_sum_now = latency_mean * completed
        d_latency_sum = latency_sum_now - self._prev_latency_sum
        step_latency_mean = (d_latency_sum / d_completed) if d_completed > 0 else 0.0
        d_total_mem = total_mem_sec - self._prev_total_mem_sec
        d_running_mem = running_mem_sec - self._prev_running_mem_sec
        d_total_cpu = total_cpu_sec - self._prev_total_cpu_sec
        d_running_cpu = running_cpu_sec - self._prev_running_cpu_sec

        self._prev_total = total
        self._prev_completed = completed
        self._prev_dropped = dropped
        self._prev_cold_starts = cold_starts
        self._prev_latency_sum = latency_sum_now
        self._prev_total_mem_sec = total_mem_sec
        self._prev_running_mem_sec = running_mem_sec
        self._prev_total_cpu_sec = total_cpu_sec
        self._prev_running_cpu_sec = running_cpu_sec

        # Ratios (0 = good, 1 = bad)
        drop_ratio = d_dropped / max(d_total, 1.0)
        cold_ratio = d_cold / max(d_total, 1.0)

        # Efficiency (0 = waste, 1 = perfect)
        mem_eff = (d_running_mem / d_total_mem) if d_total_mem > 0 else 0.0
        cpu_eff = (d_running_cpu / d_total_cpu) if d_total_cpu > 0 else 0.0

        reward = (
            - self.drop_penalty * drop_ratio
            - self.cold_start_penalty * cold_ratio
            + self.memory_efficiency_reward * mem_eff
            + self.cpu_efficiency_reward * cpu_eff
        )

        self.last_components = {
            "drop_ratio": drop_ratio,
            "cold_start_ratio": cold_ratio,
            "memory_efficiency": mem_eff,
            "cpu_efficiency": cpu_eff,
            "latency_mean": step_latency_mean,
            "d_total": d_total,
            "d_completed": d_completed,
            "d_cold": d_cold,
            "d_dropped": d_dropped,
        }

        return float(reward)
