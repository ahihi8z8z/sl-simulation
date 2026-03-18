class AutoscalingAPI:
    """Public API for controller/RL to adjust autoscaling parameters."""

    def get_idle_timeout(self):
        raise NotImplementedError

    def set_idle_timeout(self, value: float):
        raise NotImplementedError

    def get_prewarm_count(self):
        raise NotImplementedError

    def set_prewarm_count(self, count: int):
        raise NotImplementedError
