from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless_sim.cluster.node import Node


class BaseServingModel:
    """Base class for computing service time of an invocation."""

    def estimate_service_time(self, job_size: float, node: Node) -> float:
        raise NotImplementedError


class FixedRateModel(BaseServingModel):
    """service_time = job_size * processing_factor"""

    def __init__(self, processing_factor: float = 1.0):
        self.processing_factor = processing_factor

    def estimate_service_time(self, job_size: float, node: Node) -> float:
        return job_size * self.processing_factor
