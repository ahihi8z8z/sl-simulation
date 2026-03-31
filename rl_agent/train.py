"""RL training entry point — supports PPO (discrete) and A2C (continuous)."""

from __future__ import annotations

import json
import os

import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


class RewardComponentLogger(BaseCallback):
    """Logs reward components from VahidiniaEnv info to TensorBoard.

    Accumulates values across all steps in a rollout and logs
    the mean when SB3 calls _on_rollout_end.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._buffer: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            rc = info.get("reward_components")
            if rc:
                for key, val in rc.items():
                    self._buffer.setdefault(key, []).append(val)
        return True

    def _on_rollout_end(self) -> None:
        for key, vals in self._buffer.items():
            self.logger.record(f"reward/{key}", np.mean(vals))
        self._buffer.clear()


ALGORITHMS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


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
    elif env_type == "multi_discrete":
        from gym_env.multi_discrete_env import MultiDiscreteEnv
        env_class = MultiDiscreteEnv
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

    # Optional VecNormalize wrapper
    normalize_obs = rl_config.get("normalize_obs", False)
    normalize_reward = rl_config.get("normalize_reward", False)
    if normalize_obs or normalize_reward:
        vec_env = VecNormalize(vec_env, norm_obs=normalize_obs, norm_reward=normalize_reward)

    # Common SB3 params
    model_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=rl_config.get("learning_rate", 3e-4),
        gamma=rl_config.get("gamma", 0.99),
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

    # On-policy params (PPO, A2C)
    if algo_name in ("ppo", "a2c"):
        model_kwargs["n_steps"] = rl_config.get("n_steps", 128)
        model_kwargs["gae_lambda"] = rl_config.get("gae_lambda", 0.95)
        model_kwargs["ent_coef"] = rl_config.get("ent_coef", 0.0)
        model_kwargs["vf_coef"] = rl_config.get("vf_coef", 0.5)
        model_kwargs["normalize_advantage"] = rl_config.get("normalize_advantages", True)

    # PPO-specific params
    if algo_name == "ppo":
        model_kwargs["batch_size"] = rl_config.get("batch_size", 64)
        model_kwargs["n_epochs"] = rl_config.get("n_epochs", 10)
        model_kwargs["clip_range"] = rl_config.get("clip_range", 0.2)

    # DQN-specific params
    if algo_name == "dqn":
        model_kwargs["buffer_size"] = rl_config.get("buffer_size", 100000)
        model_kwargs["batch_size"] = rl_config.get("batch_size", 32)
        model_kwargs["learning_starts"] = rl_config.get("learning_starts", 1000)
        model_kwargs["train_freq"] = rl_config.get("train_freq", 4)
        model_kwargs["target_update_interval"] = rl_config.get("target_update_interval", 1000)
        model_kwargs["exploration_fraction"] = rl_config.get("exploration_fraction", 0.1)
        model_kwargs["exploration_final_eps"] = rl_config.get("exploration_final_eps", 0.05)

    # Resume from existing model or create new
    resume_path = rl_config.get("resume_model_path", None)
    if resume_path and os.path.exists(resume_path + ".zip"):
        # Load VecNormalize stats if they exist
        vec_norm_path = resume_path + "_vecnormalize.pkl"
        if isinstance(vec_env, VecNormalize) and os.path.exists(vec_norm_path):
            vec_env = VecNormalize.load(vec_norm_path, vec_env.venv)
            print(f"VecNormalize restored from {vec_norm_path}")
        model = algo_cls.load(resume_path, env=vec_env, **{
            k: v for k, v in model_kwargs.items() if k not in ("policy", "env")
        })
        print(f"Resumed from {resume_path}")
    else:
        model = algo_cls(**model_kwargs)

    # Generate versioned run name: model_name_v1, model_name_v2, ...
    save_dir = rl_config.get("output_dir", run_dir)
    os.makedirs(save_dir, exist_ok=True)

    version = 1
    while os.path.exists(os.path.join(save_dir, f"{model_name}_v{version}.zip")):
        version += 1
    versioned_name = f"{model_name}_v{version}"

    # Train
    log_interval = rl_config.get("log_interval", 1)
    print(f"Training {algo_name.upper()} with {env_type} env, {n_envs} envs, {total_timesteps} steps")
    print(f"Run: {versioned_name}")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=RewardComponentLogger(),
        tb_log_name=versioned_name,
    )

    # Save model with versioned name
    model_path = os.path.join(save_dir, versioned_name)
    model.save(model_path)

    # Save VecNormalize stats if used
    if isinstance(vec_env, VecNormalize):
        vec_norm_path = model_path + "_vecnormalize.pkl"
        vec_env.save(vec_norm_path)
        print(f"VecNormalize saved to {vec_norm_path}")

    vec_env.close()

    print(f"Model saved to {model_path}")
    return model_path
