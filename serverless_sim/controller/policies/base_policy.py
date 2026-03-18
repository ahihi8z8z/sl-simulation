class BaseControlPolicy:
    """Interface for pluggable control policies."""

    def decide(self, monitor_snapshot) -> list:
        raise NotImplementedError
