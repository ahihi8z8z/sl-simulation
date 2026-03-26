from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import simpy

from serverless_sim.cluster.compute_class import ComputeClass
from serverless_sim.cluster.node import Node
from serverless_sim.cluster.resource_profile import ResourceProfile
from serverless_sim.cluster.serving_model import FixedRateModel, PrecomputedServingModel

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class ClusterManager:
    """Creates and manages worker nodes from config."""

    def __init__(
        self,
        env: simpy.Environment,
        config: dict,
        logger: logging.Logger | None = None,
    ):
        self.env = env
        self.logger = logger or logging.getLogger(__name__)
        self.nodes: dict[str, Node] = {}

        self._build_from_config(config)

    def _build_from_config(self, config: dict) -> None:
        """Create nodes from the cluster section of config."""
        cluster_cfg = config["cluster"]
        # Optional top-level compute class defaults
        default_processing_factor = cluster_cfg.get("processing_factor", 1.0)

        # Serving model selection from config
        serving_model_type = cluster_cfg.get("serving_model", "fixed_rate")

        for node_cfg in cluster_cfg["nodes"]:
            count = node_cfg.get("count", 1)
            base_id = node_cfg.get("node_id", "node")
            for ci in range(count):
                node_id = f"{base_id}-{ci}" if count > 1 else base_id
                self._create_node(node_id, node_cfg, default_processing_factor, serving_model_type)

    def _create_node(self, node_id: str, node_cfg: dict,
                     default_processing_factor: float, serving_model_type: str) -> None:
        """Create a single node from config."""
        if node_id in self.nodes:
            return
        capacity = ResourceProfile(
            cpu=node_cfg["cpu_capacity"],
            memory=node_cfg["memory_capacity"],
        )
        compute_class = ComputeClass(
            class_id=node_cfg.get("compute_class", "default"),
            processing_factor=node_cfg.get(
                "processing_factor", default_processing_factor
            ),
            max_queue_depth=node_cfg.get("max_queue_depth", 0),
        )
        if serving_model_type == "precomputed":
            serving_model = PrecomputedServingModel()
        else:
            serving_model = FixedRateModel(
                processing_factor=compute_class.processing_factor
            )
        node = Node(
            env=self.env,
            node_id=node_id,
            capacity=capacity,
            compute_class=compute_class,
            serving_model=serving_model,
            logger=self.logger,
        )
        self.nodes[node_id] = node
        self.logger.info(
            "Created node %s: cpu=%.1f, memory=%.0f",
            node_id,
            capacity.cpu,
            capacity.memory,
        )

    def get_enabled_nodes(self) -> list[Node]:
        """Return all enabled nodes."""
        return [n for n in self.nodes.values() if n.enabled]

    def get_node(self, node_id: str) -> Node:
        """Return a node by ID. Raises KeyError if not found."""
        return self.nodes[node_id]

    def set_context(self, ctx: SimContext) -> None:
        """Set the SimContext on all nodes so they can access lifecycle manager."""
        for node in self.nodes.values():
            node._ctx = ctx

    def start_all(self) -> None:
        """Start the pull loop on all nodes."""
        for node in self.nodes.values():
            node.start_pull_loop()
