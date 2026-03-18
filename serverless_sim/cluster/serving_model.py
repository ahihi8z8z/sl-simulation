class BaseServingModel:
    def estimate_service_time(self, invocation, instance, node) -> float:
        raise NotImplementedError


class FixedRateModel(BaseServingModel):
    """service_time = job_size * processing_factor"""
    pass
