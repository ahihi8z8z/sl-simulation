import json
import os


REQUIRED_TOP_KEYS = {"simulation", "services", "cluster"}

REQUIRED_SIMULATION_KEYS = {"duration", "seed"}

REQUIRED_SERVICE_KEYS = {"service_id", "lifecycle"}

REQUIRED_CLUSTER_KEYS = {"nodes"}

REQUIRED_NODE_KEYS = {"node_id", "cpu_capacity", "memory_capacity"}


def load_config(path: str) -> dict:
    """Load and validate a JSON config file.

    Parameters
    ----------
    path : str
        Path to a JSON configuration file.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If required keys are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    _validate(config, path)
    return config


def load_config_from_dict(config: dict) -> dict:
    """Validate and return a config dictionary (no file I/O)."""
    _validate(config, "<dict>")
    return config


def _validate(config: dict, path: str) -> None:
    """Validate that required keys exist in the config."""
    missing_top = REQUIRED_TOP_KEYS - set(config.keys())
    if missing_top:
        raise ValueError(f"Config '{path}' missing top-level keys: {sorted(missing_top)}")

    # simulation section
    sim = config["simulation"]
    missing_sim = REQUIRED_SIMULATION_KEYS - set(sim.keys())
    if missing_sim:
        raise ValueError(f"Config '{path}' simulation section missing keys: {sorted(missing_sim)}")

    # services section
    services = config["services"]
    if not isinstance(services, list) or len(services) == 0:
        raise ValueError(f"Config '{path}' services must be a non-empty list")
    for i, svc in enumerate(services):
        missing_svc = REQUIRED_SERVICE_KEYS - set(svc.keys())
        if missing_svc:
            raise ValueError(f"Config '{path}' services[{i}] missing keys: {sorted(missing_svc)}")
        # Validate min_instances <= max_instances when max_instances > 0
        min_inst = svc.get("min_instances", 0)
        max_inst = svc.get("max_instances", 0)
        if max_inst > 0 and min_inst > max_inst:
            raise ValueError(
                f"Config '{path}' services[{i}] min_instances ({min_inst}) "
                f"must be <= max_instances ({max_inst})"
            )

    # cluster section
    cluster = config["cluster"]
    missing_cluster = REQUIRED_CLUSTER_KEYS - set(cluster.keys())
    if missing_cluster:
        raise ValueError(f"Config '{path}' cluster section missing keys: {sorted(missing_cluster)}")

    nodes = cluster["nodes"]
    if not isinstance(nodes, list) or len(nodes) == 0:
        raise ValueError(f"Config '{path}' cluster.nodes must be a non-empty list")
    for i, node in enumerate(nodes):
        missing_node = REQUIRED_NODE_KEYS - set(node.keys())
        if missing_node:
            raise ValueError(f"Config '{path}' cluster.nodes[{i}] missing keys: {sorted(missing_node)}")
