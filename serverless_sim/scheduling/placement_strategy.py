"""Pluggable node selection strategies for container placement.

Used by the autoscaler in global pool mode to decide which node
should host a new container.

Config: ``autoscaling.placement_strategy`` (default: ``"least_loaded"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.cluster.node import Node
    from serverless_sim.core.simulation.sim_context import SimContext


class BasePlacementStrategy:
    """Interface for container placement strategies."""

    def select_node(self, nodes: list[Node], service_id: str, ctx: SimContext) -> Node | None:
        """Select a node for placing a new container.

        Returns None if no node has sufficient resources.
        """
        raise NotImplementedError


class LeastLoadedPlacement(BasePlacementStrategy):
    """Pick the node with the most available memory that can fit the service."""

    def select_node(self, nodes: list[Node], service_id: str, ctx: SimContext) -> Node | None:
        from serverless_sim.cluster.resource_profile import ResourceProfile

        service = ctx.workload_manager.services[service_id]
        mem_req = ResourceProfile(cpu=0.0, memory=service.peak_memory)

        candidates = [n for n in nodes if n.available.can_fit(mem_req)]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.available.memory)


PLACEMENT_REGISTRY = {
    "least_loaded": LeastLoadedPlacement,
}


def create_placement_strategy(name: str = "least_loaded") -> BasePlacementStrategy:
    """Factory function to create a placement strategy by name."""
    cls = PLACEMENT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown placement strategy '{name}'. Choose from: {list(PLACEMENT_REGISTRY.keys())}")
    return cls()
