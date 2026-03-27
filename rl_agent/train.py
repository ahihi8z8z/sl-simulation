"""PPO training entry point."""

from __future__ import annotations

import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_env.serverless_gym_env import ServerlessGymEnv


def make_env(sim_config_path: str, gym_config_path: str, seed: int):
    def _init():
        return ServerlessGymEnv(sim_config_path, gym_config_path, seed=seed)
    return _init


def run_training(
    sim_config_path: str,
    gym_config_path: str,
    rl_config_path: str,
    run_dir: str = "logs",
) -> str:
    """Train a PPO model and save it.

    Returns the path to the saved model.
    """
    with open(rl_config_path, "r") as f:
        rl_config = json.load(f)

    n_envs = rl_config.get("n_envs", 4)
    total_timesteps = rl_config.get("total_timesteps", 10000)
    use_subproc = rl_config.get("use_subproc", False)
    learning_rate = rl_config.get("learning_rate", 3e-4)
    n_steps = rl_config.get("n_steps", 128)
    batch_size = rl_config.get("batch_size", 64)
    n_epochs = rl_config.get("n_epochs", 10)
    gamma = rl_config.get("gamma", 0.99)
    model_name = rl_config.get("model_name", "ppo_serverless")

    # Create vectorized environments
    env_fns = [make_env(sim_config_path, gym_config_path, seed=i) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
    )

    # Train
    model.learn(total_timesteps=total_timesteps)

    # Save
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, model_name)
    model.save(model_path)

    vec_env.close()

    print(f"Model saved to {model_path}")
    return model_path
