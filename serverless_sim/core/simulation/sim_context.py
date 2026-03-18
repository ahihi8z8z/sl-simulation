from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import simpy


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

    # Central request table
    request_table: dict = field(default_factory=dict)
