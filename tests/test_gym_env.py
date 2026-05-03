"""Unit tests for the unified ServerlessEnv gym wrapper."""

import numpy as np
from gym_env.serverless_env import ServerlessEnv


SIM_CONFIG = "configs/simulation/sample_minimal.json"
GYM_CONFIG = "configs/gym/sample_gym_discrete.json"


class TestServerlessEnv:
    def test_reset(self):
        env = ServerlessEnv(SIM_CONFIG, GYM_CONFIG)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.observation_space.shape[0],)
        assert "snapshot" in info
        env.close()

    def test_step(self):
        env = ServerlessEnv(SIM_CONFIG, GYM_CONFIG)
        obs, info = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_multiple_steps(self):
        env = ServerlessEnv(SIM_CONFIG, GYM_CONFIG)
        obs, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert info["step"] >= 1
        env.close()

    def test_observation_shape_consistent(self):
        env = ServerlessEnv(SIM_CONFIG, GYM_CONFIG)
        obs, _ = env.reset()
        expected_shape = env.observation_space.shape
        assert obs.shape == expected_shape
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert obs.shape == expected_shape
        env.close()

    def test_truncation_at_max_steps(self):
        env = ServerlessEnv(SIM_CONFIG, GYM_CONFIG)
        env.reset()
        truncated = False
        for i in range(env.max_steps + 5):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if truncated or terminated:
                break
        assert truncated
        env.close()
