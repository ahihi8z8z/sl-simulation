from __future__ import annotations


class RewardCalculator:
    """Computes reward from request counts and time-averaged cluster utilization.

    Reward = -w_drop * drop_ratio
           - w_cold * cold_ratio
           - w_mem  * mem_utilization
           - w_cpu  * cpu_utilization
           - w_lat  * (step_latency_mean / 2.5)

    drop_ratio = d_dropped / d_total, cold_ratio = d_cold / d_total
    over the step (both 0 when no arrivals).

    utilization = d(total_resource_seconds) / (capacity * step_duration),
    i.e. the time-averaged fraction of cluster capacity allocated to
    containers over the step. This penalizes the cluster footprint as
    a whole — shrinking the pool reduces the penalty, regardless of
    whether the allocated containers are actually running requests.
    """

    def __init__(
        self,
        step_duration: float,
        drop_penalty: float = 1.0,
        cold_start_penalty: float = 1.0,
        mem_utilization_penalty: float = 0.5,
        cpu_utilization_penalty: float = 0.5,
        latency_penalty: float = 0.0,
    ):
        self.step_duration = step_duration
        self.drop_penalty = abs(drop_penalty)
        self.cold_start_penalty = abs(cold_start_penalty)
        self.mem_utilization_penalty = abs(mem_utilization_penalty)
        self.cpu_utilization_penalty = abs(cpu_utilization_penalty)
        self.latency_penalty = abs(latency_penalty)

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
        capacity_mem = snapshot.get("cluster.memory_capacity", 0.0)
        capacity_cpu = snapshot.get("cluster.cpu_capacity", 0.0)

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

        # Time-averaged utilization over the step. Max possible allocated
        # resource-seconds is capacity * step_duration.
        max_mem_sec = capacity_mem * self.step_duration
        max_cpu_sec = capacity_cpu * self.step_duration
        mem_util = (d_total_mem / max_mem_sec) if max_mem_sec > 0 else 0.0
        cpu_util = (d_total_cpu / max_cpu_sec) if max_cpu_sec > 0 else 0.0

        drop_ratio = (d_dropped / d_total) if d_total > 0 else 0.0
        cold_ratio = (d_cold / d_total) if d_total > 0 else 0.0

        reward = (
            - self.drop_penalty * drop_ratio
            - self.cold_start_penalty * cold_ratio
            - self.mem_utilization_penalty * mem_util
            - self.cpu_utilization_penalty * cpu_util
            - self.latency_penalty * (step_latency_mean / 2.5)
        )

        self.last_components = {
            "mem_utilization": mem_util,
            "cpu_utilization": cpu_util,
            "latency_mean": step_latency_mean,
            "d_total": d_total,
            "d_completed": d_completed,
            "d_cold": d_cold,
            "d_dropped": d_dropped,
            "drop_ratio": drop_ratio,
            "cold_ratio": cold_ratio,
        }

        return float(reward)
