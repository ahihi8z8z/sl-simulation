class BaseGenerator:
    """Interface for workload generators."""

    def attach(self, sim_context):
        raise NotImplementedError

    def start_for_service(self, service):
        raise NotImplementedError


class PoissonFixedSizeGenerator(BaseGenerator):
    """Exponential inter-arrival with fixed request size."""
    pass
