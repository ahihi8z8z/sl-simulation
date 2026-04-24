"""Config merge utilities for experiments and sweeps.

Supports nested key notation for overrides:
    "services[0].min_instances" → config["services"][0]["min_instances"]
    "controller.enabled"        → config["controller"]["enabled"]

Usage:
    from tools.config_merge import load_merged_config, expand_sweep

    # Single experiment
    config = load_merged_config("experimental/base_config.json",
                                 {"services[0].min_instances": 2})

    # Sweep
    configs = expand_sweep("experimental/base_config.json",
                            overrides={"services[0].min_instances": 2},
                            sweep={"services[0].autoscaling_defaults.idle_timeout": [10, 30, 60]})
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path

# Pattern: key, key[0], key[1] etc.
_SEGMENT_RE = re.compile(r"^([^\[]+)(?:\[(\d+)\])?$")


def _parse_path(key: str) -> list[str | int]:
    """Parse 'a.b[0].c' into ['a', 'b', 0, 'c']."""
    parts: list[str | int] = []
    for segment in key.split("."):
        m = _SEGMENT_RE.match(segment)
        if not m:
            raise ValueError(f"Invalid key segment: {segment!r}")
        parts.append(m.group(1))
        if m.group(2) is not None:
            parts.append(int(m.group(2)))
    return parts


def set_nested(config: dict, key: str, value) -> None:
    """Set a value in a nested dict using dot/bracket notation."""
    parts = _parse_path(key)
    obj = config
    for part in parts[:-1]:
        if isinstance(part, int):
            obj = obj[part]
        else:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
    last = parts[-1]
    if isinstance(last, int):
        obj[last] = value
    else:
        obj[last] = value


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Apply overrides to a config dict (modifies in place)."""
    for key, value in overrides.items():
        set_nested(config, key, value)
    return config


def load_merged_config(base_path: str, overrides: dict | None = None) -> dict:
    """Load base config JSON and apply overrides."""
    with open(base_path) as f:
        config = json.load(f)
    if overrides:
        apply_overrides(config, overrides)
    return config


def expand_sweep(
    base_path: str,
    overrides: dict | None = None,
    sweep: dict[str, list] | None = None,
) -> list[tuple[dict, dict[str, object]]]:
    """Expand sweep parameters into list of (config, sweep_values) tuples.

    Returns a list of (merged_config, sweep_point) where sweep_point
    is a dict like {"services[0].idle_timeout": 60} for labeling.
    """
    if not sweep:
        config = load_merged_config(base_path, overrides)
        return [(config, {})]

    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]

    results = []
    for combo in itertools.product(*value_lists):
        sweep_point = dict(zip(keys, combo))
        config = load_merged_config(base_path, overrides)
        apply_overrides(config, sweep_point)
        results.append((config, sweep_point))

    return results


def load_experiments(experiments_path: str) -> tuple[str, dict]:
    """Load experiments.json and return (base_path, full_data).

    Resolves file references for base, rl_defaults, gym_defaults
    relative to experiments.json location.
    """
    exp_path = Path(experiments_path)
    with open(exp_path) as f:
        data = json.load(f)

    exp_dir = exp_path.parent

    # Resolve base config path
    data["_base_path"] = str(exp_dir / data["base"])

    # Load rl_defaults from file if it's a string path
    rl_def = data.get("rl_defaults")
    if isinstance(rl_def, str):
        with open(exp_dir / rl_def) as f:
            data["rl_templates"] = json.load(f)
    elif isinstance(rl_def, dict):
        data["rl_templates"] = rl_def

    # Load gym_defaults from file if it's a string path
    gym_def = data.get("gym_defaults")
    if isinstance(gym_def, str):
        with open(exp_dir / gym_def) as f:
            data["gym_defaults"] = json.load(f)

    # Load run_defaults from file if it's a string path
    run_def = data.get("run_defaults")
    if isinstance(run_def, str):
        with open(exp_dir / run_def) as f:
            data["run_defaults"] = json.load(f)

    return data["_base_path"], data


def build_rl_config(experiment: dict, data: dict, output_base: str = "logs") -> dict | None:
    """Build RL config for an experiment by merging template + overrides.

    Returns None if experiment has no rl_template.
    Auto-fills tensorboard_log, output_dir, model_name from experiment name.
    """
    template_name = experiment.get("rl_template")
    if template_name is None:
        return None

    templates = data.get("rl_templates", {})
    if template_name not in templates:
        raise ValueError(f"Unknown rl_template: {template_name}")

    config = copy.deepcopy(templates[template_name])

    # Apply experiment-specific rl overrides
    rl_overrides = experiment.get("rl", {})
    config.update(rl_overrides)

    # Auto-fill paths from experiment name
    name = experiment["name"]
    config.setdefault("tensorboard_log", f"{output_base}/{name}/tensorboard/")
    config.setdefault("output_dir", f"{output_base}/{name}/models")
    config.setdefault("model_name", f"{config['algorithm']}_{name}")

    return config


def build_gym_config(experiment: dict, data: dict) -> dict | None:
    """Build gym config for an experiment by merging defaults + overrides.

    Returns None if experiment has no rl_template (baseline, no training).
    """
    if experiment.get("rl_template") is None:
        return None

    defaults = data.get("gym_defaults", {})
    config = copy.deepcopy(defaults)

    # Apply experiment-specific gym overrides with DEEP merge — so partial
    # nested overrides (e.g. reward.mem_penalty only) don't wipe the other
    # keys defined in defaults (drop_penalty, cold_penalty, ...).
    gym_overrides = experiment.get("gym", {})
    for key, value in gym_overrides.items():
        if (isinstance(value, dict) and isinstance(config.get(key), dict)):
            config[key] = {**config[key], **value}
        else:
            config[key] = value

    return config


def build_infer_config(
    experiment: dict,
    data: dict,
    seed: int | None = None,
    output_base: str = "logs",
) -> dict | None:
    """Build an inference rl_config for one experiment + one seed.

    Returns None for non-RL experiments (baselines run via simulate, not infer).

    Fields derived from:
      - rl_template        → algorithm, env, device, deterministic
      - experiment name    → model_path (default: logs/<name>/models/best/best_model)
      - rl overrides       → frame_stack (mirrored so VecFrameStack matches training)

    If seed is provided, it is inserted as rl_config["seed"].
    """
    template_name = experiment.get("rl_template")
    if template_name is None:
        return None

    templates = data.get("rl_templates", {})
    if template_name not in templates:
        raise ValueError(f"Unknown rl_template: {template_name}")
    template = templates[template_name]

    config = {
        "deterministic": template.get("deterministic", True),
        "device": template.get("device", "auto"),
    }

    name = experiment["name"]
    config["algorithm"] = template["algorithm"]
    config["env"] = template["env"]
    config["model_path"] = f"{output_base}/{name}/models/best/best_model"

    rl_overrides = experiment.get("rl", {})
    frame_stack = rl_overrides.get("frame_stack", template.get("frame_stack", 1))
    if frame_stack > 1:
        config["frame_stack"] = frame_stack

    # Apply experiment.infer overrides (e.g. custom model_path)
    infer_overrides = experiment.get("infer", {})
    config.update(infer_overrides)

    if seed is not None:
        config["seed"] = seed

    return config
