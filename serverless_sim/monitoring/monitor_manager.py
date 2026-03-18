class MonitorManager:
    """Owns collectors and metric store, runs periodic collection."""

    def start(self):
        raise NotImplementedError

    def collect_once(self):
        raise NotImplementedError
