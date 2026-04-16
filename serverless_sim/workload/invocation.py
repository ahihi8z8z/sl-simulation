from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Invocation:
    """Represents a single request."""

    request_id: str = ""
    service_id: str = ""
    arrival_time: float = 0.0

    # Execution duration — set by service_time provider before dispatch
    service_time: Optional[float] = None

    dispatch_time: Optional[float] = None
    queue_enter_time: Optional[float] = None
    execution_start_time: Optional[float] = None
    execution_end_time: Optional[float] = None
    completion_time: Optional[float] = None

    assigned_node_id: Optional[str] = None
    assigned_instance_id: Optional[str] = None

    cold_start: bool = False
    dropped: bool = False
    drop_reason: Optional[str] = None  # "no_capacity", "no_nodes"
    status: str = "created"
