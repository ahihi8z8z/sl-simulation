from dataclasses import dataclass


@dataclass
class ComputeClass:
    """Represents a family of nodes."""

    class_id: str = ""
    node_count: int = 1
    serving_model_type: str = "fixed_rate"
    processing_factor: float = 1.0
    total_cpu: float = 4.0
    total_memory: float = 8192.0
    baseline_cpu: float = 0.5
    baseline_memory: float = 512.0
