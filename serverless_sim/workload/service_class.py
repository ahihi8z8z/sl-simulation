from dataclasses import dataclass


@dataclass
class ServiceClass:
    """Represents a function/service type."""

    service_id: str = ""
    display_name: str = ""
    arrival_mode: str = "poisson"
    arrival_rate: float = 1.0
    job_size: float = 1.0
    timeout: float = 30.0
    max_concurrency: int = 1
    resource_hint_cpu: float = 0.1
    resource_hint_memory: float = 128.0
    per_request_cpu: float = 0.05
