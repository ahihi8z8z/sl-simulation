"""Unit tests for Step 1: Config loader, logging, run directory."""

import json
import logging
import os
import tempfile

import pytest

from serverless_sim.core.config.loader import load_config
from serverless_sim.core.logging.logger_factory import create_logger


VALID_CONFIG = {
    "simulation": {"duration": 10.0, "seed": 42},
    "services": [
        {
            "service_id": "svc-a",
            "arrival_rate": 1.0,
            "job_size": 0.1,
            "timeout": 5.0,
            "memory": 256,
            "cpu": 1.0,
            "max_concurrency": 2,
        }
    ],
    "cluster": {
        "nodes": [{"node_id": "node-0", "cpu_capacity": 4.0, "memory_capacity": 4096}]
    },
}


def _write_config(config: dict, tmpdir: str) -> str:
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


# ------------------------------------------------------------------ #
# Config loader tests
# ------------------------------------------------------------------ #

class TestLoadConfig:
    def test_load_valid_config(self):
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = _write_config(VALID_CONFIG, tmpdir)
        cfg = load_config(path)
        assert cfg["simulation"]["duration"] == 10.0
        assert len(cfg["services"]) == 1
        assert cfg["cluster"]["nodes"][0]["node_id"] == "node-0"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.json")

    def test_invalid_json(self):
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = os.path.join(tmpdir, "bad.json")
        with open(path, "w") as f:
            f.write("NOT JSON")
        with pytest.raises(json.JSONDecodeError):
            load_config(path)

    def test_missing_top_level_key(self):
        cfg = {k: v for k, v in VALID_CONFIG.items() if k != "cluster"}
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = _write_config(cfg, tmpdir)
        with pytest.raises(ValueError, match="missing top-level"):
            load_config(path)

    def test_missing_simulation_key(self):
        cfg = {**VALID_CONFIG, "simulation": {"seed": 42}}  # missing duration
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = _write_config(cfg, tmpdir)
        with pytest.raises(ValueError, match="simulation section"):
            load_config(path)

    def test_empty_services(self):
        cfg = {**VALID_CONFIG, "services": []}
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = _write_config(cfg, tmpdir)
        with pytest.raises(ValueError, match="non-empty list"):
            load_config(path)

    def test_missing_service_key(self):
        svc = {k: v for k, v in VALID_CONFIG["services"][0].items() if k != "cpu"}
        cfg = {**VALID_CONFIG, "services": [svc]}
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = _write_config(cfg, tmpdir)
        with pytest.raises(ValueError, match="services\\[0\\]"):
            load_config(path)

    def test_empty_nodes(self):
        cfg = {**VALID_CONFIG, "cluster": {"nodes": []}}
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = _write_config(cfg, tmpdir)
        with pytest.raises(ValueError, match="non-empty list"):
            load_config(path)

    def test_missing_node_key(self):
        cfg = {**VALID_CONFIG, "cluster": {"nodes": [{"node_id": "n-0"}]}}
        tmpdir = tempfile.mkdtemp(prefix="test_cfg_")
        path = _write_config(cfg, tmpdir)
        with pytest.raises(ValueError, match="cluster.nodes\\[0\\]"):
            load_config(path)


# ------------------------------------------------------------------ #
# Logger factory tests
# ------------------------------------------------------------------ #

class TestCreateLogger:
    def test_console_mode(self):
        tmpdir = tempfile.mkdtemp(prefix="test_log_")
        logger = create_logger("test_console", tmpdir, mode="console", level="DEBUG")
        assert logger.level == logging.DEBUG
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" not in handler_types

    def test_file_mode(self):
        tmpdir = tempfile.mkdtemp(prefix="test_log_")
        logger = create_logger("test_file", tmpdir, mode="file", level="INFO")
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "FileHandler" in handler_types
        assert "StreamHandler" not in handler_types
        assert os.path.exists(os.path.join(tmpdir, "simulation.log"))

    def test_both_mode(self):
        tmpdir = tempfile.mkdtemp(prefix="test_log_")
        logger = create_logger("test_both", tmpdir, mode="both", level="WARNING")
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 2

    def test_file_mode_writes(self):
        tmpdir = tempfile.mkdtemp(prefix="test_log_")
        logger = create_logger("test_write", tmpdir, mode="file", level="INFO")
        logger.info("test message")
        for h in logger.handlers:
            h.flush()
        log_path = os.path.join(tmpdir, "simulation.log")
        with open(log_path) as f:
            content = f.read()
        assert "test message" in content

    def test_creates_run_dir_if_missing(self):
        tmpdir = tempfile.mkdtemp(prefix="test_log_")
        nested = os.path.join(tmpdir, "sub", "dir")
        logger = create_logger("test_nested", nested, mode="file")
        assert os.path.isdir(nested)
        assert os.path.exists(os.path.join(nested, "simulation.log"))
