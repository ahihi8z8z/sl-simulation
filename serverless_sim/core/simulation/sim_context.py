from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import simpy

from serverless_sim.workload.request_store import RequestStore


@dataclass
class SimContext:
    """Central runtime dependency container."""

    env: simpy.Environment
    config: dict
    rng: Any  # numpy RandomGenerator
    logger: Any
    run_dir: str

    # Module references — set after construction by SimulationBuilder
    workload_manager: Any = None
    dispatcher: Any = None
    cluster_manager: Any = None
    lifecycle_manager: Any = None
    autoscaling_manager: Any = None
    monitor_manager: Any = None
    controller: Optional[Any] = None
    export_manager: Optional[Any] = None
    placement_strategy: Optional[Any] = None
    service_time_providers: dict[str, Any] = field(default_factory=dict)

    # Central request store (replaces plain dict for memory efficiency)
    request_table: RequestStore = field(default_factory=RequestStore)

    # Global request counter — shared across all per-service generators so
    # request IDs stay unique.
    _request_counter: int = 0

    def next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter}"
