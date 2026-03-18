class ClusterManager:
    """Creates and manages worker nodes."""

    def get_enabled_nodes(self):
        raise NotImplementedError

    def get_node(self, node_id: str):
        raise NotImplementedError
