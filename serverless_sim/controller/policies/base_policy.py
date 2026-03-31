from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class BaseControlPolicy:
    """Interface for pluggable control policies."""

    def decide(self, snapshot: dict, ctx: SimContext) -> list[dict]:
        """Given a metric snapshot, return a list of actions.

        Each action is a dict like:
            {"action": "set_pool_target", "service_id": "svc-a", "state": "prewarm", "value": 3}
            {"action": "set_idle_timeout", "service_id": "svc-a", "value": 15.0}
            {"action": "set_min_instances", "service_id": "svc-a", "value": 2}
            {"action": "set_max_instances", "service_id": "svc-a", "value": 10}
        """
        return []  # no-op by default
