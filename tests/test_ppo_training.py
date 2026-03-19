"""Unit tests for Step 16: PPO training with stable-baselines3."""

import json
import os
import tempfile

import pytest

from serverless_sim.rl_agent.train import make_env, run_training


# Paths to sample configs
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
SIM_CONFIG = os.path.join(CONFIGS_DIR, "simulation", "sample_minimal.json")
GYM_CONFIG = os.path.join(CONFIGS_DIR, "gym", "sample_gym_discrete.json")


def _write_rl_config(tmpdir: str, overrides: dict | None = None) -> str:
    rl_config = {
        "n_envs": 1,
        "total_timesteps": 128,
        "use_subproc": False,
        "learning_rate": 3e-4,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "gamma": 0.99,
        "model_name": "test_ppo",
    }
    if overrides:
        rl_config.update(overrides)
    path = os.path.join(tmpdir, "rl_config.json")
    with open(path, "w") as f:
        json.dump(rl_config, f)
    return path


class TestMakeEnv:
    def test_creates_callable(self):
        fn = make_env(SIM_CONFIG, GYM_CONFIG, seed=0)
        assert callable(fn)

    def test_callable_creates_env(self):
        fn = make_env(SIM_CONFIG, GYM_CONFIG, seed=0)
        env = fn()
        obs, info = env.reset()
        assert obs is not None
        env.close()


class TestRunTraining:
    def test_train_saves_model(self):
        tmpdir = tempfile.mkdtemp(prefix="test_train_")
        rl_path = _write_rl_config(tmpdir)

        model_path = run_training(SIM_CONFIG, GYM_CONFIG, rl_path, run_dir=tmpdir)

        assert os.path.exists(model_path + ".zip")

    def test_train_with_2_envs(self):
        tmpdir = tempfile.mkdtemp(prefix="test_train_")
        rl_path = _write_rl_config(tmpdir, {"n_envs": 2})

        model_path = run_training(SIM_CONFIG, GYM_CONFIG, rl_path, run_dir=tmpdir)

        assert os.path.exists(model_path + ".zip")

    def test_model_name_custom(self):
        tmpdir = tempfile.mkdtemp(prefix="test_train_")
        rl_path = _write_rl_config(tmpdir, {"model_name": "my_model"})

        model_path = run_training(SIM_CONFIG, GYM_CONFIG, rl_path, run_dir=tmpdir)

        assert model_path.endswith("my_model")
        assert os.path.exists(model_path + ".zip")
