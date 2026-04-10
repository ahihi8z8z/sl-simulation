"""RL inference entry point — supports PPO (discrete) and A2C (continuous)."""

from __future__ import annotations

import json
import os

import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ALGORITHMS = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "maskable_ppo": MaskablePPO, "recurrent_ppo": RecurrentPPO}


def run_inference(
    sim_config_path: str,
    gym_config_path: str,
    rl_config_path: str,
    run_dir: str = "logs",
) -> dict:
    """Load a trained model and run inference episodes.

    rl_config JSON fields:
        algorithm   : "ppo" (default) or "a2c"
        env         : "discrete" (default) or "vahidinia"
        model_path  : path to saved model (without .zip)
        n_episodes  : number of episodes (default 1)
        seed        : random seed (default 42)

    Returns summary statistics.
    """
    with open(rl_config_path, "r") as f:
        rl_config = json.load(f)

    algo_name = rl_config.get("algorithm", "ppo").lower()
    env_type = rl_config.get("env", "discrete")
    model_path = rl_config.get("model_path", "")
    n_episodes = rl_config.get("n_episodes", 1)
    seed = rl_config.get("seed", 42)
    deterministic = rl_config.get("deterministic", True)
    device = rl_config.get("device", "auto")

    if not model_path or not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Select env class
    if env_type == "vahidinia":
        from gym_env.vahidinia_env import VahidiniaEnv
        env_class = VahidiniaEnv
    elif env_type == "multi_discrete":
        from gym_env.multi_discrete_env import MultiDiscreteEnv
        env_class = MultiDiscreteEnv
    else:
        from gym_env.serverless_gym_env import ServerlessGymEnv
        env_class = ServerlessGymEnv

    # Load model
    algo_cls = ALGORITHMS.get(algo_name)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Choose from: {list(ALGORITHMS.keys())}")
    model = algo_cls.load(model_path, device=device)

    # Check for VecNormalize stats
    vec_norm_path = model_path + "_vecnormalize.pkl"
    has_vec_norm = os.path.exists(vec_norm_path)

    # Run episodes
    all_rewards = []
    all_steps = []

    for ep in range(n_episodes):
        env = env_class(sim_config_path, gym_config_path, seed=seed + ep)

        if has_vec_norm:
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(vec_norm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        episode_reward = 0.0
        step_count = 0

        use_masks = algo_name == "maskable_ppo" and hasattr(env, "action_masks")

        while True:
            if use_masks:
                action, _ = model.predict(obs, deterministic=deterministic,
                                          action_masks=env.action_masks())
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            if has_vec_norm:
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                step_count += 1
                if done[0]:
                    break
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                if terminated or truncated:
                    break

        all_rewards.append(episode_reward)
        all_steps.append(step_count)

        print(f"Episode {ep + 1}/{n_episodes}: reward={episode_reward:.2f}, steps={step_count}")
        if has_vec_norm:
            vec_env.close()
        else:
            env.close()

    summary = {
        "n_episodes": n_episodes,
        "mean_reward": sum(all_rewards) / len(all_rewards),
        "total_steps": sum(all_steps),
        "rewards": all_rewards,
    }
    print(f"\nMean reward: {summary['mean_reward']:.2f}")
    return summary
