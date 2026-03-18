class BaseCollector:
    def collect(self, env_time, sim_context) -> dict:
        raise NotImplementedError


class RequestCollector(BaseCollector):
    pass


class ClusterCollector(BaseCollector):
    pass


class LifecycleCollector(BaseCollector):
    pass


class AutoscalingCollector(BaseCollector):
    pass
