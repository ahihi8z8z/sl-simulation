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

        lm = ctx.lifecycle_manager
        lm.flush_resource_seconds()
        metrics: dict[str, float] = {
            "lifecycle.instances_total": total_instances,
            "lifecycle.instances_evicted": lm._evicted_count,
            "lifecycle.total_cpu_seconds": lm.total_cpu_seconds,
            "lifecycle.total_memory_seconds": lm.total_memory_seconds,
            "lifecycle.running_cpu_seconds": lm.running_cpu_seconds,
            "lifecycle.running_memory_seconds": lm.running_memory_seconds,
        }

        # Per-state counts — always emit base states with default 0
        for base_state in ("prewarm", "warm", "running"):
            metrics[f"lifecycle.instances_{base_state}"] = per_state.get(base_state, 0)
        for state, count in per_state.items():
            metrics[f"lifecycle.instances_{state}"] = count

        for svc_id, counts in per_service.items():
            metrics[f"lifecycle.{svc_id}.instances_total"] = counts["total"]
            metrics[f"lifecycle.{svc_id}.instances_running"] = counts["running"]

        return metrics


class InterArrivalCollector(BaseCollector):
    """Collects per-service inter-arrival time and last cold-start flag.

    Metrics exposed:
        request.<svc>.inter_arrival_time  – seconds since previous invocation
        request.<svc>.last_cold_start     – 1.0 if last completed request was cold, else 0.0
    """

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        store = ctx.request_table
        metrics: dict[str, float] = {}

        for svc_id in ctx.workload_manager.services:
            iat = store._inter_arrival.get(svc_id)
            if iat is not None:
                metrics[f"request.{svc_id}.inter_arrival_time"] = iat

            cold = store._last_cold_start.get(svc_id)
            if cold is not None:
                metrics[f"request.{svc_id}.last_cold_start"] = 1.0 if cold else 0.0

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
            # Remaining capacity (max_instances - current)
            max_inst = ctx.autoscaling_manager.get_max_instances(svc_id)
            current = ctx.autoscaling_manager._count_total_instances(svc_id)
            metrics[f"autoscaling.{svc_id}.remaining_capacity"] = max(0, max_inst - current)

            # Per-state pool targets and pool container counts
            targets = ctx.autoscaling_manager.get_all_pool_targets(svc_id)
            for state, count in targets.items():
                metrics[f"autoscaling.{svc_id}.pool_target.{state}"] = count
                metrics[f"autoscaling.{svc_id}.pool_containers.{state}"] = (
                    ctx.autoscaling_manager._count_pool_containers(svc_id, state)
                )

            # Demand container count
            metrics[f"autoscaling.{svc_id}.demand_containers"] = (
                ctx.autoscaling_manager._count_demand_containers(svc_id)
            )

        return metrics
