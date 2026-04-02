from dataclasses import dataclass


@dataclass
class ComputeClass:
    """Represents a family of nodes sharing the same hardware profile."""

    class_id: str = ""
    serving_model_type: str = "fixed_rate"
    processing_factor: float = 1.0
    max_queue_depth: int = 0  # 0 = unlimited
    reserved_cpu: float = 0.0  # CPU reserved for system overhead
    reserved_memory: float = 0.0  # Memory (MB) reserved for system overhead
