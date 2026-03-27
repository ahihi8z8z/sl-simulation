"""Unit tests for Step 17: PPO inference with trained model."""

import json
import os
import tempfile

import pytest

from rl_agent.train import run_training
from rl_agent.infer import run_inference


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
SIM_CONFIG = os.path.join(CONFIGS_DIR, "simulation", "sample_minimal.json")
GYM_CONFIG = os.path.join(CONFIGS_DIR, "gym", "sample_gym_discrete.json")


def _train_model(tmpdir: str) -> str:
    """Train a minimal model and return its path."""
    rl_config = {
        "n_envs": 1,
        "total_timesteps": 128,
        "use_subproc": False,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "model_name": "test_model",
    }
    rl_path = os.path.join(tmpdir, "rl_train.json")
    with open(rl_path, "w") as f:
        json.dump(rl_config, f)
    return run_training(SIM_CONFIG, GYM_CONFIG, rl_path, run_dir=tmpdir)


class TestRunInference:
    def test_inference_runs(self):
        tmpdir = tempfile.mkdtemp(prefix="test_infer_")
        model_path = _train_model(tmpdir)

        infer_config = {
            "model_path": model_path,
            "n_episodes": 1,
            "seed": 100,
        }
        infer_path = os.path.join(tmpdir, "rl_infer.json")
        with open(infer_path, "w") as f:
            json.dump(infer_config, f)

        summary = run_inference(SIM_CONFIG, GYM_CONFIG, infer_path, run_dir=tmpdir)

        assert summary["n_episodes"] == 1
        assert "mean_reward" in summary
        assert isinstance(summary["mean_reward"], float)
        assert summary["total_steps"] > 0

    def test_inference_multiple_episodes(self):
        tmpdir = tempfile.mkdtemp(prefix="test_infer_")
        model_path = _train_model(tmpdir)

        infer_config = {
            "model_path": model_path,
            "n_episodes": 2,
            "seed": 42,
        }
        infer_path = os.path.join(tmpdir, "rl_infer.json")
        with open(infer_path, "w") as f:
            json.dump(infer_config, f)

        summary = run_inference(SIM_CONFIG, GYM_CONFIG, infer_path, run_dir=tmpdir)

        assert summary["n_episodes"] == 2
        assert len(summary["rewards"]) == 2

    def test_inference_missing_model(self):
        tmpdir = tempfile.mkdtemp(prefix="test_infer_")
        infer_config = {
            "model_path": "/nonexistent/model",
            "n_episodes": 1,
        }
        infer_path = os.path.join(tmpdir, "rl_infer.json")
        with open(infer_path, "w") as f:
            json.dump(infer_config, f)

        with pytest.raises(FileNotFoundError):
            run_inference(SIM_CONFIG, GYM_CONFIG, infer_path, run_dir=tmpdir)
