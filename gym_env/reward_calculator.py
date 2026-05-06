from __future__ import annotations


class RewardCalculator:
    """Per-service reward, averaged across services.

    For each service s, compute the same penalty terms as before but using
    the service's own request counters and lifecycle resource-seconds:

        drop_ratio_s = d(request.s.dropped) / d(request.s.total)
        cold_ratio_s = d(request.s.cold_starts) / d(request.s.total)
        mem_util_s   = d(lifecycle.s.total_memory_seconds) / (cluster.memory_capacity * step)
        cpu_util_s   = d(lifecycle.s.total_cpu_seconds) / (cluster.cpu_capacity * step)
        latency_s    = d(latency_sum_s) / d(request.s.completed)
                       where latency_sum_s = latency_mean_s * completed_s

        r_s = -w_drop * drop_ratio_s
              -w_cold * cold_ratio_s
              -w_mem  * mem_util_s
              -w_cpu  * cpu_util_s
              -w_lat  * (latency_s / 2.5)

    Final reward = (1/N) * sum(r_s) over all discovered services.

    Service IDs are discovered lazily from snapshot keys matching
    ``request.<svc>.total``.
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

        # Per-service prev counters: {svc_id: {key: prev_value}}
        self._prev: dict[str, dict[str, float]] = {}

    def reset(self) -> None:
        self._prev.clear()

    @staticmethod
    def _discover_services(snapshot: dict) -> list[str]:
        prefix = "request."
        suffix = ".total"
        services = []
        for key in snapshot.keys():
            if key.startswith(prefix) and key.endswith(suffix):
                svc = key[len(prefix):-len(suffix)]
                if svc:  # skip the cluster-wide "request.total"
                    services.append(svc)
        return sorted(services)

    def compute(self, snapshot: dict) -> float:
        services = self._discover_services(snapshot)
        if not services:
            self.last_components = {}
            return 0.0

        capacity_mem = float(snapshot.get("cluster.memory_capacity", 0.0))
        capacity_cpu = float(snapshot.get("cluster.cpu_capacity", 0.0))
        max_mem_sec = capacity_mem * self.step_duration
        max_cpu_sec = capacity_cpu * self.step_duration

        per_service: dict[str, dict[str, float]] = {}
        rewards: list[float] = []

        for svc in services:
            prev = self._prev.setdefault(svc, {})

            total = float(snapshot.get(f"request.{svc}.total", 0.0))
            completed = float(snapshot.get(f"request.{svc}.completed", 0.0))
            dropped = float(snapshot.get(f"request.{svc}.dropped", 0.0))
            cold_starts = float(snapshot.get(f"request.{svc}.cold_starts", 0.0))
            latency_mean = float(snapshot.get(f"request.{svc}.latency_mean", 0.0))
            total_mem_sec = float(snapshot.get(f"lifecycle.{svc}.total_memory_seconds", 0.0))
            total_cpu_sec = float(snapshot.get(f"lifecycle.{svc}.total_cpu_seconds", 0.0))

            d_total = total - prev.get("total", 0.0)
            d_completed = completed - prev.get("completed", 0.0)
            d_dropped = dropped - prev.get("dropped", 0.0)
            d_cold = cold_starts - prev.get("cold_starts", 0.0)

            latency_sum_now = latency_mean * completed
            d_latency_sum = latency_sum_now - prev.get("latency_sum", 0.0)
            step_latency_mean = (d_latency_sum / d_completed) if d_completed > 0 else 0.0

            d_mem = total_mem_sec - prev.get("total_mem_sec", 0.0)
            d_cpu = total_cpu_sec - prev.get("total_cpu_sec", 0.0)

            prev["total"] = total
            prev["completed"] = completed
            prev["dropped"] = dropped
            prev["cold_starts"] = cold_starts
            prev["latency_sum"] = latency_sum_now
            prev["total_mem_sec"] = total_mem_sec
            prev["total_cpu_sec"] = total_cpu_sec

            mem_util = (d_mem / max_mem_sec) if max_mem_sec > 0 else 0.0
            cpu_util = (d_cpu / max_cpu_sec) if max_cpu_sec > 0 else 0.0
            drop_ratio = (d_dropped / d_total) if d_total > 0 else 0.0
            cold_ratio = (d_cold / d_total) if d_total > 0 else 0.0

            r_s = (
                - self.drop_penalty * drop_ratio
                - self.cold_start_penalty * cold_ratio
                - self.mem_utilization_penalty * mem_util
                - self.cpu_utilization_penalty * cpu_util
                - self.latency_penalty * (step_latency_mean / 2.5)
            )
            rewards.append(r_s)

            per_service[svc] = {
                "drop_ratio": drop_ratio,
                "cold_ratio": cold_ratio,
                "mem_utilization": mem_util,
                "cpu_utilization": cpu_util,
                "latency_mean": step_latency_mean,
                "d_total": d_total,
                "d_completed": d_completed,
                "d_dropped": d_dropped,
                "d_cold": d_cold,
                "reward": r_s,
            }

        reward = sum(rewards) / len(rewards)

        agg_keys = ("drop_ratio", "cold_ratio", "mem_utilization",
                    "cpu_utilization", "latency_mean",
                    "d_total", "d_completed", "d_dropped", "d_cold")
        agg = {
            k: sum(c[k] for c in per_service.values()) / len(per_service)
            for k in agg_keys
        }
        agg["services"] = per_service
        self.last_components = agg

        return float(reward)
