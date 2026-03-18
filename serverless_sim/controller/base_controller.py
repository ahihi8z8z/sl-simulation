class BaseController:
    """Periodic control loop: read monitor -> call policy -> apply actions."""

    def run(self):
        """SimPy process."""
        raise NotImplementedError
