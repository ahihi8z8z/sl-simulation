class SimulationBuilder:
    """Constructs the entire simulator from config."""

    def build(self, config: dict, run_dir: str, logger):
        raise NotImplementedError
