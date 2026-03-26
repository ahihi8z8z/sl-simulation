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

        # --- Cluster resource usage (latest snapshot) ---
        if self.ctx.monitor_manager:
            from serverless_sim.monitoring.monitor_api import MonitorAPI
            api = MonitorAPI(self.ctx.monitor_manager)
            snapshot = api.get_snapshot()

            summary["cluster_resources"] = {
                "cpu_total": snapshot.get("cluster.cpu_total", 0),
                "cpu_used": snapshot.get("cluster.cpu_used", 0),
                "cpu_utilization": round(snapshot.get("cluster.cpu_utilization", 0), 4),
                "memory_total": snapshot.get("cluster.memory_total", 0),
                "memory_used": snapshot.get("cluster.memory_used", 0),
                "memory_utilization": round(snapshot.get("cluster.memory_utilization", 0), 4),
                "nodes_enabled": int(snapshot.get("cluster.nodes_enabled", 0)),
            }

            summary["lifecycle"] = {
                "instances_total": int(snapshot.get("lifecycle.instances_total", 0)),
                "instances_warm": int(snapshot.get("lifecycle.instances_warm", 0)),
                "instances_running": int(snapshot.get("lifecycle.instances_running", 0)),
                "instances_prewarm": int(snapshot.get("lifecycle.instances_prewarm", 0)),
                "instances_evicted": int(snapshot.get("lifecycle.instances_evicted", 0)),
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
