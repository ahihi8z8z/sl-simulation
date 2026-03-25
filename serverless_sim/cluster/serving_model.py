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

    def estimate_service_time(self, job_size: float, node: Node, **kwargs) -> float:
        return job_size * self.processing_factor


class PrecomputedServingModel(BaseServingModel):
    """Use pre-computed service_time from Invocation (set by trace generator).

    Falls back to job_size if service_time is not set.
    """

    def estimate_service_time(self, job_size: float, node: Node, **kwargs) -> float:
        service_time = kwargs.get("service_time")
        if service_time is not None:
            return service_time
        return job_size
