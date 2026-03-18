class SimulationEngine:
    """Top-level runtime object for standalone simulation."""

    def setup(self):
        raise NotImplementedError

    def run(self, until: float):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def get_snapshot(self):
        raise NotImplementedError
