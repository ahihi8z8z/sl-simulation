from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class SummaryWriter:
    """Writes summary.json at end of simulation."""

    def __init__(self, ctx: SimContext):
        self.ctx = ctx

    def write(self, wall_clock_seconds: float | None = None) -> str:
        """Write summary.json and return the file path."""
        run_dir = self.ctx.run_dir
        path = os.path.join(run_dir, "summary.json")

        store = self.ctx.request_table
        c = store.counters
        config = self.ctx.config
        duration = config["simulation"]["duration"]
        sim_end_time = self.ctx.env.now

        # --- Simulation info ---
        summary: dict = {
            "simulation": {
                "duration_config": duration,
                "sim_end_time": round(sim_end_time, 3),
                "wall_clock_seconds": round(wall_clock_seconds, 2) if wall_clock_seconds else None,
                "seed": config["simulation"]["seed"],
                "num_services": len(config["services"]),
                "num_nodes": sum(n.get("count", 1) for n in config["cluster"]["nodes"]),
            },
            "requests": {
                "total": c.total,
                "completed": c.completed,
                "dropped": c.dropped,
                "truncated": c.truncated,
                "cold_starts": c.cold_starts,
            },
        }

        # --- Latency ---
        if c.completed > 0:
            summary["latency"] = {
                "mean": round(store.latency_mean, 6),
            }

        # --- Rates ---
        if c.total > 0:
            summary["rates"] = {
                "throughput_req_per_s": round(c.completed / duration, 4),
                "drop_rate_pct": round(c.dropped / c.total * 100, 2),
                "cold_start_rate_pct": round(c.cold_starts / max(c.completed, 1) * 100, 2),
            }

        # --- Resource metrics (from accumulated resource-seconds) ---
        if self.ctx.lifecycle_manager:
            eff = self._compute_effective_ratio()
            summary["effective_resource_ratio"] = eff

            # Avg utilization from resource-seconds
            nodes = self.ctx.cluster_manager.get_enabled_nodes()
            cluster_cpu = sum(n.capacity.cpu for n in nodes)
            cluster_mem = sum(n.capacity.memory for n in nodes)

            flavor_cpu = sum(n.flavor_cpu_used for n in nodes)
            flavor_mem = sum(n.flavor_memory_used for n in nodes)

            # Power model params
            cluster_cfg = self.ctx.config.get("cluster", {})
            power_base = cluster_cfg.get("power_base", 90.0)
            power_max = cluster_cfg.get("power_max", 150.0)

            # Average power from resource-seconds
            node_cpu_cap = nodes[0].capacity.cpu if nodes else 1.0
            avg_cpu_used = eff["total_cpu_seconds"] / sim_end_time if sim_end_time > 0 else 0.0
            # Active servers: approximate from total instances over time
            lm = self.ctx.lifecycle_manager
            active_servers_now = sum(
                1 for n in nodes
                if len(lm.get_instances_for_node(n.node_id)) > 0
            ) if lm else 0
            avg_power = active_servers_now * power_base + (avg_cpu_used / node_cpu_cap) * (power_max - power_base)

            summary["cluster_utilization"] = {
                "cpu_total": cluster_cpu,
                "memory_total": cluster_mem,
                "nodes_enabled": len(nodes),
                "avg_cpu_utilization": round(eff["total_cpu_seconds"] / (sim_end_time * cluster_cpu), 4) if cluster_cpu > 0 and sim_end_time > 0 else 0.0,
                "avg_memory_utilization": round(eff["total_memory_seconds"] / (sim_end_time * cluster_mem), 4) if cluster_mem > 0 and sim_end_time > 0 else 0.0,
                "flavor_cpu_used": round(flavor_cpu, 4),
                "flavor_memory_used": round(flavor_mem, 2),
                "flavor_cpu_utilization": round(flavor_cpu / cluster_cpu, 4) if cluster_cpu > 0 else 0.0,
                "flavor_memory_utilization": round(flavor_mem / cluster_mem, 4) if cluster_mem > 0 else 0.0,
                "active_servers": active_servers_now,
                "power_base": power_base,
                "power_max": power_max,
                "avg_power": round(avg_power, 2),
            }

            # Lifecycle: count actual instances from lifecycle_manager
            lm = self.ctx.lifecycle_manager
            state_counts = {}
            total_instances = 0
            for node in nodes:
                for inst in lm.get_instances_for_node(node.node_id):
                    total_instances += 1
                    state_counts[inst.state] = state_counts.get(inst.state, 0) + 1

            summary["lifecycle"] = {
                "instances_total": total_instances,
                "instances_evicted": lm._evicted_count,
                **{f"instances_{state}": count for state, count in sorted(state_counts.items())},
            }

            # Per-service autoscaling
            if self.ctx.autoscaling_manager:
                autoscaling = {}
                for svc_id in self.ctx.workload_manager.services:
                    autoscaling[svc_id] = {
                        "min_instances": self.ctx.autoscaling_manager.get_min_instances(svc_id),
                        "max_instances": self.ctx.autoscaling_manager.get_max_instances(svc_id),
                        "idle_timeout": self.ctx.autoscaling_manager.get_idle_timeout(svc_id),
                        "current_instances": self.ctx.autoscaling_manager._count_total_instances(svc_id),
                        "alive_instances": self.ctx.autoscaling_manager._count_alive_instances(svc_id),
                        "pool_targets": self.ctx.autoscaling_manager.get_all_pool_targets(svc_id),
                    }
                summary["autoscaling"] = autoscaling

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        return path

    def _compute_effective_ratio(self) -> dict:
        """Compute effective resource ratio from accumulated resource-seconds."""
        lm = self.ctx.lifecycle_manager
        if lm is None:
            return {}

        # Flush: accumulate for currently alive instances
        now = self.ctx.env.now
        total_cpu = lm.total_cpu_seconds
        total_mem = lm.total_memory_seconds
        running_cpu = lm.running_cpu_seconds
        running_mem = lm.running_memory_seconds

        for node in self.ctx.cluster_manager.get_enabled_nodes():
            for inst in lm.get_instances_for_node(node.node_id):
                time_in_state = now - inst.state_entered_at
                if time_in_state > 0:
                    total_cpu += time_in_state * inst.allocated_cpu
                    total_mem += time_in_state * inst.allocated_memory
                    if inst.state == "running":
                        running_cpu += time_in_state * inst.allocated_cpu
                        running_mem += time_in_state * inst.allocated_memory

        completed = self.ctx.request_table.counters.completed
        return {
            "cpu_effective_ratio": round(running_cpu / total_cpu, 4) if total_cpu > 0 else 0.0,
            "memory_effective_ratio": round(running_mem / total_mem, 4) if total_mem > 0 else 0.0,
            "cpu_per_request": round(total_cpu / completed, 4) if completed > 0 else 0.0,
            "memory_per_request": round(total_mem / completed, 4) if completed > 0 else 0.0,
            "total_cpu_seconds": round(total_cpu, 2),
            "running_cpu_seconds": round(running_cpu, 2),
            "total_memory_seconds": round(total_mem, 2),
            "running_memory_seconds": round(running_mem, 2),
        }
