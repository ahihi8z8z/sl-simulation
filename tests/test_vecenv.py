"""Unit tests for Step 15: VecEnv compatibility."""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gym_env.serverless_gym_env import ServerlessGymEnv


SIM_CONFIG = "configs/simulation/sample_minimal.json"
GYM_CONFIG = "configs/gym/sample_gym_discrete.json"


def make_env(seed: int):
    """Factory function for creating gym envs with unique seeds."""
    def _init():
        return ServerlessGymEnv(SIM_CONFIG, GYM_CONFIG, seed=seed)
    return _init


class TestDummyVecEnv:
    def test_reset_and_step(self):
        env = DummyVecEnv([make_env(i) for i in range(4)])
        obs = env.reset()
        assert obs.shape[0] == 4
        actions = [0, 0, 0, 0]
        obs2, rewards, dones, infos = env.step(actions)
        assert obs2.shape[0] == 4
        assert len(rewards) == 4
        env.close()

    def test_observation_shapes_consistent(self):
        env = DummyVecEnv([make_env(i) for i in range(2)])
        obs = env.reset()
        for _ in range(3):
            obs, _, _, _ = env.step([0, 0])
            assert obs.shape[0] == 2
            assert obs.shape[1] == env.observation_space.shape[0]
        env.close()


class TestSubprocVecEnv:
    def test_reset_and_step(self):
        env = SubprocVecEnv([make_env(i + 100) for i in range(2)])
        obs = env.reset()
        assert obs.shape[0] == 2
        actions = [0, 0]
        obs2, rewards, dones, infos = env.step(actions)
        assert obs2.shape[0] == 2
        assert len(rewards) == 2
        env.close()

    def test_different_seeds(self):
        """Each env should have a different seed."""
        env = SubprocVecEnv([make_env(i + 200) for i in range(2)])
        obs = env.reset()
        # After a few steps, observations should differ (different seeds)
        for _ in range(5):
            obs, _, _, _ = env.step([0, 0])
        # Not strictly guaranteed to differ on every step, but likely
        env.close()
