"""Tests for the Vahidinia et al. 2023 paper Gymnasium environment."""

import numpy as np
from gym_env.vahidinia_env import VahidiniaEnv


SIM_CONFIG = "configs/simulation/sample_minimal.json"


class TestVahidiniaEnv:
    def test_spaces(self):
        env = VahidiniaEnv(SIM_CONFIG)
        n_svc = len(env._service_ids)
        assert env.observation_space.shape == (n_svc * 2,)
        assert env.action_space.shape == (n_svc,)
        assert env.action_space.low[0] == env.idle_timeout_min
        assert env.action_space.high[0] == env.idle_timeout_max
        env.close()

    def test_reset(self):
        env = VahidiniaEnv(SIM_CONFIG)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert "snapshot" in info
        env.close()

    def test_step_returns(self):
        env = VahidiniaEnv(SIM_CONFIG)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["step"] == 1
        env.close()

    def test_observation_contains_inter_arrival(self):
        """After a few steps, inter-arrival time should be positive."""
        env = VahidiniaEnv(SIM_CONFIG)
        env.reset()
        for _ in range(5):
            obs, _, _, _, _ = env.step(env.action_space.sample())
        # obs[0] = inter_arrival_time for first service
        assert obs[0] >= 0.0
        env.close()

    def test_reward_is_negative(self):
        """Paper reward -(Cold/N) - P should always be <= 0."""
        env = VahidiniaEnv(SIM_CONFIG)
        env.reset()
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert reward <= 0.0
            if terminated or truncated:
                break
        env.close()

    def test_continuous_action_clipped(self):
        """Actions outside bounds should be clipped without error."""
        env = VahidiniaEnv(SIM_CONFIG)
        env.reset()
        # Way below min
        obs, reward, _, _, _ = env.step(np.array([0.0]))
        assert isinstance(reward, float)
        # Way above max
        obs, reward, _, _, _ = env.step(np.array([99999.0]))
        assert isinstance(reward, float)
        env.close()

    def test_truncation_at_max_steps(self):
        env = VahidiniaEnv(SIM_CONFIG, gym_config_path=None)
        env.max_steps = 5
        env.reset()
        truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert truncated
        env.close()

    def test_custom_config(self):
        """Gym config overrides should be respected."""
        import json, tempfile, os
        cfg = {
            "max_steps": 10,
            "idle_timeout_min": 60.0,
            "idle_timeout_max": 300.0,
            "memory_penalty_weight": 0.05,
        }
        tmp = tempfile.mktemp(suffix=".json")
        with open(tmp, "w") as f:
            json.dump(cfg, f)
        try:
            env = VahidiniaEnv(SIM_CONFIG, gym_config_path=tmp)
            assert env.max_steps == 10
            assert env.idle_timeout_min == 60.0
            assert env.idle_timeout_max == 300.0
            assert env.memory_penalty_weight == 0.05
            assert env.action_space.low[0] == 60.0
            assert env.action_space.high[0] == 300.0
            env.close()
        finally:
            os.unlink(tmp)
