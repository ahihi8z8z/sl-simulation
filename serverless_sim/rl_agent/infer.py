"""PPO inference entry point."""

from __future__ import annotations

import json
import os

from stable_baselines3 import PPO

from serverless_sim.gym_env.serverless_gym_env import ServerlessGymEnv


def run_inference(
    sim_config_path: str,
    gym_config_path: str,
    rl_config_path: str,
    run_dir: str = "logs",
) -> dict:
    """Load a trained PPO model and run inference episodes.

    Returns summary statistics.
    """
    with open(rl_config_path, "r") as f:
        rl_config = json.load(f)

    model_path = rl_config.get("model_path", "")
    n_episodes = rl_config.get("n_episodes", 1)
    seed = rl_config.get("seed", 42)

    if not model_path or not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Load model
    model = PPO.load(model_path)

    # Run episodes
    all_rewards = []
    all_steps = []

    for ep in range(n_episodes):
        env = ServerlessGymEnv(sim_config_path, gym_config_path, seed=seed + ep)
        obs, _ = env.reset()
        episode_reward = 0.0
        step_count = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            step_count += 1
            if terminated or truncated:
                break

        all_rewards.append(episode_reward)
        all_steps.append(step_count)

        print(f"Episode {ep + 1}/{n_episodes}: reward={episode_reward:.2f}, steps={step_count}")
        env.close()

    summary = {
        "n_episodes": n_episodes,
        "mean_reward": sum(all_rewards) / len(all_rewards),
        "total_steps": sum(all_steps),
        "rewards": all_rewards,
    }
    print(f"\nMean reward: {summary['mean_reward']:.2f}")
    return summary
