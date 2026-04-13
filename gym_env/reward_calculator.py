from __future__ import annotations


class RewardCalculator:
    """Computes reward from ratios and resource utilization metrics.

    Reward = -w_drop * drop_ratio
           - w_cold * cold_start_ratio
           - w_mem  * mem_utilization
           - w_cpu  * cpu_utilization

    All weights positive. All components in [0, 1].
    mem/cpu_utilization = avg resource used / total cluster capacity.
    """

    def __init__(
        self,
        drop_penalty: float = 1.0,
        cold_start_penalty: float = 1.0,
        mem_utilization_penalty: float = 0.5,
        cpu_utilization_penalty: float = 0.5,
        step_duration: float = 3600.0,
        cluster_memory: float = 16384.0,
        cluster_cpu: float = 16.0,
        # Backward-compat: ignored
        mem_per_request_penalty: float = 0.0,
        cpu_per_request_penalty: float = 0.0,
        memory_efficiency_reward: float = 0.0,
        cpu_efficiency_reward: float = 0.0,
        latency_penalty: float = 0.0,
        cold_start_normalize: float = 0.0,
        drop_normalize: float = 0.0,
        resource_penalty: float = 0.0,
        throughput_reward: float = 0.0,
    ):
        self.drop_penalty = abs(drop_penalty)
        self.cold_start_penalty = abs(cold_start_penalty)
        self.mem_utilization_penalty = abs(mem_utilization_penalty)
        self.cpu_utilization_penalty = abs(cpu_utilization_penalty)
        self.latency_penalty = abs(latency_penalty)
        self.cold_start_normalize = cold_start_normalize  # 0 = ratio mode, >0 = fixed denominator
        self.drop_normalize = drop_normalize
        self.step_duration = step_duration
        self.cluster_memory = cluster_memory
        self.cluster_cpu = cluster_cpu

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

        # Ratios (0 = good, 1 = bad)
        cold_denom = self.cold_start_normalize if self.cold_start_normalize > 0 else max(d_total, 1.0)
        drop_denom = self.drop_normalize if self.drop_normalize > 0 else max(d_total, 1.0)
        drop_ratio = d_dropped / drop_denom
        cold_ratio = d_cold / cold_denom

        # Resource utilization (0 = idle, 1 = full cluster)
        max_mem_sec = self.step_duration * self.cluster_memory
        max_cpu_sec = self.step_duration * self.cluster_cpu
        mem_util = (d_total_mem / max_mem_sec) if max_mem_sec > 0 else 0.0
        cpu_util = (d_total_cpu / max_cpu_sec) if max_cpu_sec > 0 else 0.0

        # Resource per request (for logging)
        mem_per_req = (d_total_mem / d_completed) if d_completed > 0 else 0.0
        cpu_per_req = (d_total_cpu / d_completed) if d_completed > 0 else 0.0

        reward = (
            - self.drop_penalty * drop_ratio
            - self.cold_start_penalty * cold_ratio
            - self.mem_utilization_penalty * mem_util
            - self.cpu_utilization_penalty * cpu_util
            - self.latency_penalty * (step_latency_mean / 2.5)
        )

        self.last_components = {
            "drop_ratio": drop_ratio,
            "cold_start_ratio": cold_ratio,
            "mem_utilization": mem_util,
            "cpu_utilization": cpu_util,
            "latency_mean": step_latency_mean,
            "mem_per_request": mem_per_req,
            "cpu_per_request": cpu_per_req,
            "d_total": d_total,
            "d_completed": d_completed,
            "d_cold": d_cold,
            "d_dropped": d_dropped,
        }

        return float(reward)
