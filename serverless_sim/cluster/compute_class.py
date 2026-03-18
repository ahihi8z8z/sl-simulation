from dataclasses import dataclass


@dataclass
class ComputeClass:
    """Represents a family of nodes sharing the same hardware profile."""

    class_id: str = ""
    serving_model_type: str = "fixed_rate"
    processing_factor: float = 1.0
