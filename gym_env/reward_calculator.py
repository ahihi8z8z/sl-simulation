from __future__ import annotations


class RewardCalculator:
    """Computes reward from raw counts and resource efficiency metrics.

    Reward = -w_drop * d_dropped
           - w_cold * d_cold
           - w_mem  * (1 - mem_efficiency)
           - w_cpu  * (1 - cpu_efficiency)
           - w_lat  * (step_latency_mean / 2.5)

    efficiency = d(running_resource_seconds) / d(total_resource_seconds),
    i.e. fraction of allocated resource actually in the running state.
    When no resource is allocated (d_total == 0), efficiency defaults to 1
    (no inefficiency penalty).
    """

    def __init__(
        self,
        drop_penalty: float = 1.0,
        cold_start_penalty: float = 1.0,
        mem_utilization_penalty: float = 0.5,
        cpu_utilization_penalty: float = 0.5,
        latency_penalty: float = 0.0,
    ):
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
        self._prev_running_mem_sec = 0.0
        self._prev_running_cpu_sec = 0.0

    def reset(self) -> None:
        self._prev_total = 0.0
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_cold_starts = 0.0
        self._prev_latency_sum = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_total_cpu_sec = 0.0
        self._prev_running_mem_sec = 0.0
        self._prev_running_cpu_sec = 0.0

    def compute(self, snapshot: dict) -> float:
        """Compute reward from current metric snapshot."""
        total = snapshot.get("request.total", 0.0)
        completed = snapshot.get("request.completed", 0.0)
        dropped = snapshot.get("request.dropped", 0.0)
        cold_starts = snapshot.get("request.cold_starts", 0.0)
        latency_mean = snapshot.get("request.latency_mean", 0.0)
        total_mem_sec = snapshot.get("lifecycle.total_memory_seconds", 0.0)
        total_cpu_sec = snapshot.get("lifecycle.total_cpu_seconds", 0.0)
        running_mem_sec = snapshot.get("lifecycle.running_memory_seconds", 0.0)
        running_cpu_sec = snapshot.get("lifecycle.running_cpu_seconds", 0.0)

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
        d_running_mem = running_mem_sec - self._prev_running_mem_sec
        d_running_cpu = running_cpu_sec - self._prev_running_cpu_sec

        self._prev_total = total
        self._prev_completed = completed
        self._prev_dropped = dropped
        self._prev_cold_starts = cold_starts
        self._prev_latency_sum = latency_sum_now
        self._prev_total_mem_sec = total_mem_sec
        self._prev_total_cpu_sec = total_cpu_sec
        self._prev_running_mem_sec = running_mem_sec
        self._prev_running_cpu_sec = running_cpu_sec

        # Efficiency = running / total allocated. Default 1.0 when nothing is
        # allocated, so idle steps with no containers incur no penalty.
        mem_efficiency = (d_running_mem / d_total_mem) if d_total_mem > 0 else 1.0
        cpu_efficiency = (d_running_cpu / d_total_cpu) if d_total_cpu > 0 else 1.0

        # Resource per request (for logging)
        mem_per_req = (d_total_mem / d_completed) if d_completed > 0 else 0.0
        cpu_per_req = (d_total_cpu / d_completed) if d_completed > 0 else 0.0

        reward = (
            - self.drop_penalty * d_dropped
            - self.cold_start_penalty * d_cold
            - self.mem_utilization_penalty * (1.0 - mem_efficiency)
            - self.cpu_utilization_penalty * (1.0 - cpu_efficiency)
            - self.latency_penalty * (step_latency_mean / 2.5)
        )

        self.last_components = {
            "mem_efficiency": mem_efficiency,
            "cpu_efficiency": cpu_efficiency,
            "latency_mean": step_latency_mean,
            "mem_per_request": mem_per_req,
            "cpu_per_request": cpu_per_req,
            "d_total": d_total,
            "d_completed": d_completed,
            "d_cold": d_cold,
            "d_dropped": d_dropped,
        }

        return float(reward)
