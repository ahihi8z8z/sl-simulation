class WorkloadManager:
    """Manages services and workload generation."""

    def start(self):
        raise NotImplementedError

    def register_service(self, service):
        raise NotImplementedError
