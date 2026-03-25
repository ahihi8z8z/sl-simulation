from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Invocation:
    """Represents a single request."""

    request_id: str = ""
    service_id: str = ""
    arrival_time: float = 0.0
    job_size: float = 0.0

    dispatch_time: Optional[float] = None
    queue_enter_time: Optional[float] = None
    execution_start_time: Optional[float] = None
    execution_end_time: Optional[float] = None
    completion_time: Optional[float] = None

    assigned_node_id: Optional[str] = None
    assigned_instance_id: Optional[str] = None

    # Pre-computed service time (set by trace generator, used by PrecomputedServingModel)
    service_time: Optional[float] = None

    cold_start: bool = False
    dropped: bool = False
    drop_reason: Optional[str] = None  # "no_capacity", "no_nodes"
    status: str = "created"
