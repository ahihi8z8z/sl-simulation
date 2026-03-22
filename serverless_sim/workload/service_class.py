from dataclasses import dataclass, field


@dataclass
class ServiceClass:
    """Represents a function/service type."""

    service_id: str = ""
    display_name: str = ""
    arrival_mode: str = "poisson"
    arrival_rate: float = 1.0
    job_size: float = 1.0
    max_concurrency: int = 1
    memory: float = 256.0
    cpu: float = 1.0
    min_instances: int = 0
    max_instances: int = 0  # 0 = unlimited

    @classmethod
    def from_config(cls, cfg: dict) -> "ServiceClass":
        """Build a ServiceClass from a service config dict."""
        return cls(
            service_id=cfg["service_id"],
            display_name=cfg.get("display_name", cfg["service_id"]),
            arrival_mode=cfg.get("arrival_mode", "poisson"),
            arrival_rate=cfg["arrival_rate"],
            job_size=cfg["job_size"],
            max_concurrency=cfg["max_concurrency"],
            memory=cfg["memory"],
            cpu=cfg["cpu"],
            min_instances=cfg.get("min_instances", 0),
            max_instances=cfg.get("max_instances", 0),
        )
