"""RL training entry point — supports PPO (discrete) and A2C (continuous)."""

from __future__ import annotations

import json
import os

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class RewardComponentLogger(BaseCallback):
    """Logs reward components from VahidiniaEnv info to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        components = [i.get("reward_components") for i in infos if "reward_components" in i]
        if components:
            for key in components[0]:
                mean_val = np.mean([c[key] for c in components])
                self.logger.record(f"reward/{key}", mean_val)
        return True


ALGORITHMS = {"ppo": PPO, "a2c": A2C}


def _make_env(env_class, sim_config_path: str, gym_config_path: str, seed: int):
    def _init():
        env = env_class(sim_config_path, gym_config_path, seed=seed)
        return Monitor(env)
    return _init


def make_env(sim_config_path: str, gym_config_path: str, seed: int):
    """Backward-compatible: creates ServerlessGymEnv."""
    from gym_env.serverless_gym_env import ServerlessGymEnv
    return _make_env(ServerlessGymEnv, sim_config_path, gym_config_path, seed)


def run_training(
    sim_config_path: str,
    gym_config_path: str,
    rl_config_path: str,
    run_dir: str = "logs",
) -> str:
    """Train an RL model and save it.

    Returns the path to the saved model.
    """
    with open(rl_config_path, "r") as f:
        rl_config = json.load(f)

    algo_name = rl_config.get("algorithm", "ppo").lower()
    env_type = rl_config.get("env", "discrete")
    n_envs = rl_config.get("n_envs", 4)
    total_timesteps = rl_config.get("total_timesteps", 10000)
    use_subproc = rl_config.get("use_subproc", False)
    model_name = rl_config.get("model_name", "rl_serverless")
    policy_kwargs = rl_config.get("policy_kwargs", None)

    # Select env class
    if env_type == "vahidinia":
        from gym_env.vahidinia_env import VahidiniaEnv
        env_class = VahidiniaEnv
    else:
        from gym_env.serverless_gym_env import ServerlessGymEnv
        env_class = ServerlessGymEnv

    # Select algorithm
    algo_cls = ALGORITHMS.get(algo_name)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Choose from: {list(ALGORITHMS.keys())}")

    # Create vectorized environments
    env_fns = [_make_env(env_class, sim_config_path, gym_config_path, seed=i) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Common SB3 params
    model_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=rl_config.get("learning_rate", 3e-4),
        n_steps=rl_config.get("n_steps", 128),
        gamma=rl_config.get("gamma", 0.99),
        gae_lambda=rl_config.get("gae_lambda", 0.95),
        ent_coef=rl_config.get("ent_coef", 0.0),
        vf_coef=rl_config.get("vf_coef", 0.5),
        normalize_advantage=rl_config.get("normalize_advantages", True),
        device=rl_config.get("device", "auto"),
        verbose=1,
    )

    # Tensorboard logging
    tb_log = rl_config.get("tensorboard_log", None)
    if tb_log:
        model_kwargs["tensorboard_log"] = tb_log

    # Policy kwargs (net_arch, activation_fn, etc.)
    if policy_kwargs:
        model_kwargs["policy_kwargs"] = policy_kwargs

    # PPO-specific params
    if algo_name == "ppo":
        model_kwargs["batch_size"] = rl_config.get("batch_size", 64)
        model_kwargs["n_epochs"] = rl_config.get("n_epochs", 10)
        model_kwargs["clip_range"] = rl_config.get("clip_range", 0.2)

    model = algo_cls(**model_kwargs)

    # Train
    log_interval = rl_config.get("log_interval", 1)
    print(f"Training {algo_name.upper()} with {env_type} env, {n_envs} envs, {total_timesteps} steps")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=RewardComponentLogger(),
    )

    # Save
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, model_name)
    model.save(model_path)

    vec_env.close()

    print(f"Model saved to {model_path}")
    return model_path
