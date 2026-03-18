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
    timeout: float = 30.0

    dispatch_time: Optional[float] = None
    queue_enter_time: Optional[float] = None
    execution_start_time: Optional[float] = None
    execution_end_time: Optional[float] = None
    completion_time: Optional[float] = None

    assigned_node_id: Optional[str] = None
    assigned_instance_id: Optional[str] = None

    cold_start: bool = False
    dropped: bool = False
    timed_out: bool = False
    drop_reason: Optional[str] = None  # "queue_full", "timeout", "no_capacity"
    status: str = "created"
