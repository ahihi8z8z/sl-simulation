class LifecycleManager:
    """Manages container instance lifecycle."""

    def find_reusable_instance(self, node, service_id):
        raise NotImplementedError

    def prepare_instance_for_service(self, node, service_id):
        raise NotImplementedError

    def start_execution(self, instance, invocation):
        raise NotImplementedError

    def finish_execution(self, instance, invocation):
        raise NotImplementedError
