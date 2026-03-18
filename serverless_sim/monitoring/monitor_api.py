class MonitorAPI:
    """Read-only API for querying metrics."""

    def get_latest(self, metric_name: str = None):
        raise NotImplementedError

    def query_range(self, metric_name: str, start: float, end: float):
        raise NotImplementedError

    def get_snapshot(self, metric_names=None):
        raise NotImplementedError
