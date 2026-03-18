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
    """Collects request-level metrics from the request table."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        table = ctx.request_table
        total = len(table)
        completed = sum(1 for inv in table.values() if inv.status == "completed")
        dropped = sum(1 for inv in table.values() if inv.dropped)
        timed_out = sum(1 for inv in table.values() if inv.timed_out)
        in_flight = sum(
            1 for inv in table.values()
            if inv.status in ("queued", "arrived") or
            (inv.execution_start_time is not None and inv.execution_end_time is None)
        )

        return {
            "request.total": total,
            "request.completed": completed,
            "request.dropped": dropped,
            "request.timed_out": timed_out,
            "request.in_flight": in_flight,
        }


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
    """Collects lifecycle instance metrics (basic version)."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        if ctx.lifecycle_manager is None:
            return {}

        total_instances = 0
        warm_instances = 0
        running_instances = 0

        for node in ctx.cluster_manager.get_enabled_nodes():
            instances = ctx.lifecycle_manager.get_instances_for_node(node.node_id)
            for inst in instances:
                if inst.state == "evicted":
                    continue
                total_instances += 1
                if inst.state == "warm":
                    warm_instances += 1
                elif inst.state == "running":
                    running_instances += 1

        return {
            "lifecycle.instances_total": total_instances,
            "lifecycle.instances_warm": warm_instances,
            "lifecycle.instances_running": running_instances,
        }


class AutoscalingCollector(BaseCollector):
    """Placeholder for autoscaling metrics (Step 11)."""

    def collect(self, env_time: float, ctx: SimContext) -> dict[str, float]:
        return {}
