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
            "request.cold_starts": c.cold_starts,
            "request.truncated": c.truncated,
            "request.in_flight": store.active_count,
        }

        if c.completed > 0:
            metrics["request.latency_mean"] = store.latency_mean

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

        # Resource used only by running containers
        running_cpu = 0.0
        running_mem = 0.0
        if ctx.lifecycle_manager is not None:
            for node in nodes:
                for inst in ctx.lifecycle_manager.get_instances_for_node(node.node_id):
                    if inst.state == "running":
                        running_cpu += inst.allocated_cpu
                        running_mem += inst.allocated_memory

        metrics["cluster.nodes_enabled"] = len(nodes)
        metrics["cluster.cpu_total"] = total_cpu
        metrics["cluster.cpu_used"] = used_cpu
        metrics["cluster.cpu_used_running"] = running_cpu
        metrics["cluster.cpu_utilization"] = used_cpu / total_cpu if total_cpu > 0 else 0.0
        metrics["cluster.memory_total"] = total_mem
        metrics["cluster.memory_used"] = used_mem
        metrics["cluster.memory_used_running"] = running_mem
        metrics["cluster.memory_utilization"] = used_mem / total_mem if total_mem > 0 else 0.0

        return metrics


class LifecycleCollector(BaseCollector):
    """Collects lifecycle instance metrics per state and per service.

    Dynamically counts instances in every state (including extended
    states like ``code_loaded``), not just the default three.
    """

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        if ctx.lifecycle_manager is None:
            return {}

        total_instances = 0
        per_state: dict[str, int] = {}
        per_service: dict[str, dict[str, int]] = {}

        for node in ctx.cluster_manager.get_enabled_nodes():
            instances = ctx.lifecycle_manager.get_instances_for_node(node.node_id)
            for inst in instances:
                total_instances += 1
                per_state[inst.state] = per_state.get(inst.state, 0) + 1

                # Per-service counts
                svc_counts = per_service.setdefault(inst.service_id, {"total": 0, "running": 0})
                svc_counts["total"] += 1
                if inst.state == "running":
                    svc_counts["running"] += 1

        metrics: dict[str, float] = {
            "lifecycle.instances_total": total_instances,
            "lifecycle.instances_evicted": ctx.lifecycle_manager._evicted_count,
        }

        # Per-state counts (all states, including extended)
        for state, count in per_state.items():
            metrics[f"lifecycle.instances_{state}"] = count

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
            metrics[f"autoscaling.{svc_id}.min_instances"] = ctx.autoscaling_manager.get_min_instances(svc_id)
            metrics[f"autoscaling.{svc_id}.max_instances"] = ctx.autoscaling_manager.get_max_instances(svc_id)
            metrics[f"autoscaling.{svc_id}.current_instances"] = ctx.autoscaling_manager._count_total_instances(svc_id)
            metrics[f"autoscaling.{svc_id}.alive_instances"] = ctx.autoscaling_manager._count_alive_instances(svc_id)
            metrics[f"autoscaling.{svc_id}.warm_instances"] = ctx.autoscaling_manager._count_warm_instances(svc_id)
            # Per-state pool targets
            targets = ctx.autoscaling_manager.get_all_pool_targets(svc_id)
            for state, count in targets.items():
                metrics[f"autoscaling.{svc_id}.pool_target.{state}"] = count

        return metrics
