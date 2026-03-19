from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class BaseCollector:
    """Interface for metric collectors."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        """Collect metrics at the given time. Returns {metric_name: value}."""
        raise NotImplementedError


class RequestCollector(BaseCollector):
    """Collects request-level metrics from the request store counters."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        store = ctx.request_table
        c = store.counters

        metrics = {
            "request.total": c.total,
            "request.completed": c.completed,
            "request.dropped": c.dropped,
            "request.timed_out": c.timed_out,
            "request.cold_starts": c.cold_starts,
            "request.truncated": c.truncated,
            "request.in_flight": store.active_count,
        }

        # store.latencies is kept sorted via bisect.insort — no sort needed
        latencies = store.latencies
        if latencies:
            n = len(latencies)
            metrics["request.latency_mean"] = store._latency_sum / n
            metrics["request.latency_p50"] = latencies[n // 2]
            metrics["request.latency_p95"] = latencies[int(n * 0.95)]
            metrics["request.latency_p99"] = latencies[int(n * 0.99)]

        return metrics


class ClusterCollector(BaseCollector):
    """Collects cluster-level resource metrics."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        metrics = {}
        if ctx.cluster_manager is None:
            return metrics

        nodes = ctx.cluster_manager.get_enabled_nodes()
        total_cpu = sum(n.capacity.cpu for n in nodes)
        used_cpu = sum(n.allocated.cpu for n in nodes)
        total_mem = sum(n.capacity.memory for n in nodes)
        used_mem = sum(n.allocated.memory for n in nodes)

        metrics["cluster.nodes_enabled"] = len(nodes)
        metrics["cluster.cpu_total"] = total_cpu
        metrics["cluster.cpu_used"] = used_cpu
        metrics["cluster.cpu_utilization"] = used_cpu / total_cpu if total_cpu > 0 else 0.0
        metrics["cluster.memory_total"] = total_mem
        metrics["cluster.memory_used"] = used_mem
        metrics["cluster.memory_utilization"] = used_mem / total_mem if total_mem > 0 else 0.0

        return metrics


class LifecycleCollector(BaseCollector):
    """Collects lifecycle instance metrics per state and per service."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        if ctx.lifecycle_manager is None:
            return {}

        total_instances = 0
        warm_instances = 0
        running_instances = 0
        prewarm_instances = 0
        evicted_instances = 0
        per_service: dict[str, dict[str, int]] = {}

        for node in ctx.cluster_manager.get_enabled_nodes():
            instances = ctx.lifecycle_manager.get_instances_for_node(node.node_id)
            for inst in instances:
                if inst.state == "evicted":
                    evicted_instances += 1
                    continue
                total_instances += 1
                if inst.state == "warm":
                    warm_instances += 1
                elif inst.state == "running":
                    running_instances += 1
                elif inst.state == "prewarm":
                    prewarm_instances += 1

                # Per-service counts
                svc_counts = per_service.setdefault(inst.service_id, {"total": 0, "running": 0})
                svc_counts["total"] += 1
                if inst.state == "running":
                    svc_counts["running"] += 1

        metrics = {
            "lifecycle.instances_total": total_instances,
            "lifecycle.instances_warm": warm_instances,
            "lifecycle.instances_running": running_instances,
            "lifecycle.instances_prewarm": prewarm_instances,
            "lifecycle.instances_evicted": evicted_instances,
        }

        for svc_id, counts in per_service.items():
            metrics[f"lifecycle.{svc_id}.instances_total"] = counts["total"]
            metrics[f"lifecycle.{svc_id}.instances_running"] = counts["running"]

        return metrics


class AutoscalingCollector(BaseCollector):
    """Collects autoscaling parameter metrics."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        if ctx.autoscaling_manager is None:
            return {}

        metrics = {}
        for svc_id in ctx.workload_manager.services:
            metrics[f"autoscaling.{svc_id}.idle_timeout"] = ctx.autoscaling_manager.get_idle_timeout(svc_id)
            # Per-state pool targets
            targets = ctx.autoscaling_manager.get_all_pool_targets(svc_id)
            for state, count in targets.items():
                metrics[f"autoscaling.{svc_id}.pool_target.{state}"] = count

        return metrics
