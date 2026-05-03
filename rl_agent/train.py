"""RL training entry point — supports PPO (discrete) and A2C (continuous)."""

from __future__ import annotations

import json
import os

from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack, unwrap_vec_normalize,
)
from sb3_contrib import MaskablePPO, RecurrentPPO


def _parse_lr(v):
    """Parse learning rate: plain number or "lin_<start>" for linear decay to 0."""
    if isinstance(v, str) and v.startswith("lin_"):
        return LinearSchedule(float(v[4:]), 0.0, 1.0)
    return float(v)


class CheckpointWithNormalize(BaseCallback):
    """Save model + VecNormalize stats periodically."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "checkpoint", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            # Save VecNormalize if present (may be wrapped by VecFrameStack etc.)
            vec_norm = unwrap_vec_normalize(self.model.get_env())
            if vec_norm is not None:
                vec_norm.save(path + "_vecnormalize.pkl")
            if self.verbose:
                print(f"Checkpoint saved: {path}")
        return True


class _SaveVecNormOnBest(BaseCallback):
    """Callback triggered when EvalCallback finds a new best model."""

    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_norm = unwrap_vec_normalize(self.model.get_env())
        if vec_norm is not None:
            vec_norm.save(os.path.join(self.save_path, "best_model_vecnormalize.pkl"))
        return True


class RewardComponentLogger(BaseCallback):
    """Log reward components via logger.record_mean() — accumulated across
    all steps between SB3 dumps, then reset. No manual buffer/flush needed."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            rc = info.get("reward_components")
            if rc and rc.get("d_total", 0) > 0:
                for key, val in rc.items():
                    self.logger.record_mean(f"reward/{key}", float(val))
        return True


ALGORITHMS = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "sac": SAC, "maskable_ppo": MaskablePPO, "recurrent_ppo": RecurrentPPO}


def _make_env(env_class, sim_config_path: str, gym_config_path: str, seed: int):
    def _init():
        env = env_class(sim_config_path, gym_config_path, seed=seed)
        return Monitor(env)
    return _init


def make_env(sim_config_path: str, gym_config_path: str, seed: int):
    from gym_env.serverless_env import ServerlessEnv
    return _make_env(ServerlessEnv, sim_config_path, gym_config_path, seed)


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
    env_type = rl_config.get("env", "serverless")
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
        from gym_env.serverless_env import ServerlessEnv
        env_class = ServerlessEnv

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

    # Optional frame stacking (stack last N obs — gives agent temporal history).
    # Must wrap BEFORE VecNormalize so off-policy algos (SAC/DQN) can call
    # VecNormalize.get_original_obs() and get the stacked shape back for the
    # replay buffer (buffer is allocated with stacked observation_space).
    frame_stack = rl_config.get("frame_stack", 1)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    # Optional VecNormalize wrapper
    normalize_obs = rl_config.get("normalize_obs", False)
    normalize_reward = rl_config.get("normalize_reward", False)
    if normalize_obs or normalize_reward:
        vec_env = VecNormalize(vec_env, norm_obs=normalize_obs, norm_reward=normalize_reward)

    # Common SB3 params
    policy_name = "MlpLstmPolicy" if algo_name == "recurrent_ppo" else "MlpPolicy"
    model_kwargs = dict(
        policy=policy_name,
        env=vec_env,
        learning_rate=_parse_lr(rl_config.get("learning_rate", 3e-4)),
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

    # On-policy params (PPO, A2C, MaskablePPO)
    if algo_name in ("ppo", "a2c", "maskable_ppo", "recurrent_ppo"):
        model_kwargs["n_steps"] = rl_config.get("n_steps", 128)
        model_kwargs["gae_lambda"] = rl_config.get("gae_lambda", 0.95)
        model_kwargs["ent_coef"] = rl_config.get("ent_coef", 0.0)
        model_kwargs["vf_coef"] = rl_config.get("vf_coef", 0.5)
        model_kwargs["normalize_advantage"] = rl_config.get("normalize_advantages", True)

    # PPO-specific params (PPO and MaskablePPO)
    if algo_name in ("ppo", "maskable_ppo", "recurrent_ppo"):
        model_kwargs["batch_size"] = rl_config.get("batch_size", 64)
        model_kwargs["n_epochs"] = rl_config.get("n_epochs", 10)
        model_kwargs["clip_range"] = rl_config.get("clip_range", 0.2)

    # RecurrentPPO: LSTM/policy flags
    if algo_name == "recurrent_ppo":
        pk = (model_kwargs.get("policy_kwargs") or {}).copy()
        pk["lstm_hidden_size"]   = rl_config.get("lstm_hidden_size", 64)
        pk["enable_critic_lstm"] = rl_config.get("enable_critic_lstm", True)
        pk["ortho_init"]         = rl_config.get("ortho_init", False)
        model_kwargs["policy_kwargs"] = pk

    # DQN-specific params
    if algo_name == "dqn":
        model_kwargs["buffer_size"] = rl_config.get("buffer_size", 100000)
        model_kwargs["batch_size"] = rl_config.get("batch_size", 32)
        model_kwargs["learning_starts"] = rl_config.get("learning_starts", 1000)
        model_kwargs["train_freq"] = rl_config.get("train_freq", 4)
        model_kwargs["target_update_interval"] = rl_config.get("target_update_interval", 1000)
        model_kwargs["exploration_fraction"] = rl_config.get("exploration_fraction", 0.1)
        model_kwargs["exploration_final_eps"] = rl_config.get("exploration_final_eps", 0.05)

    # SAC-specific params
    if algo_name == "sac":
        model_kwargs["buffer_size"] = rl_config.get("buffer_size", 100000)
        model_kwargs["batch_size"] = rl_config.get("batch_size", 256)
        model_kwargs["learning_starts"] = rl_config.get("learning_starts", 1000)
        model_kwargs["train_freq"] = rl_config.get("train_freq", 1)
        model_kwargs["gradient_steps"] = rl_config.get("gradient_steps", 1)
        model_kwargs["ent_coef"] = rl_config.get("ent_coef", "auto")

    # Resume from existing model or create new
    resume_path = rl_config.get("resume_model_path", None)
    if resume_path and os.path.exists(resume_path + ".zip"):
        # Load VecNormalize stats if they exist
        vec_norm_path = resume_path + "_vecnormalize.pkl"
        if isinstance(vec_env, VecNormalize) and os.path.exists(vec_norm_path):
            vec_env = VecNormalize.load(vec_norm_path, vec_env.venv)
            print(f"VecNormalize restored from {vec_norm_path}")
        # SB3 requires exact match of stored policy_kwargs, so drop it on load —
        # the architecture is already baked into the saved model.
        model = algo_cls.load(resume_path, env=vec_env, **{
            k: v for k, v in model_kwargs.items()
            if k not in ("policy", "env", "policy_kwargs")
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
    save_freq = rl_config.get("save_freq", 0)

    callbacks = [RewardComponentLogger()]
    if save_freq > 0:
        callbacks.append(CheckpointWithNormalize(
            save_freq=save_freq,
            save_path=save_dir,
            name_prefix=versioned_name,
        ))

    # Early stopping based on eval reward
    eval_freq = rl_config.get("eval_freq", 0)
    early_stop_patience = rl_config.get("early_stop_patience", 0)
    if eval_freq > 0:
        eval_env = DummyVecEnv([_make_env(env_class, sim_config_path, gym_config_path, seed=9999)])
        if frame_stack > 1:
            eval_env = VecFrameStack(eval_env, n_stack=frame_stack)
        if normalize_obs or normalize_reward:
            eval_env = VecNormalize(eval_env, norm_obs=normalize_obs, norm_reward=normalize_reward)

        eval_kwargs = dict(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=rl_config.get("n_eval_episodes", 1),
            best_model_save_path=os.path.join(save_dir, "best"),
            deterministic=True,
            verbose=1,
        )
        if early_stop_patience > 0:
            eval_kwargs["callback_after_eval"] = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=early_stop_patience,
                verbose=1,
            )
        eval_kwargs["callback_on_new_best"] = _SaveVecNormOnBest(
            save_path=os.path.join(save_dir, "best"),
        )
        callbacks.append(EvalCallback(**eval_kwargs))

    print(f"Training {algo_name.upper()} with {env_type} env, {n_envs} envs, {total_timesteps} steps")
    print(f"Run: {versioned_name}")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        progress_bar=True,
        callback=callbacks,
        tb_log_name=versioned_name,
    )

    # Save final model with versioned name
    model_path = os.path.join(save_dir, versioned_name)
    model.save(model_path)
    vec_norm = unwrap_vec_normalize(vec_env)
    if vec_norm is not None:
        vec_norm.save(model_path + "_vecnormalize.pkl")
    print(f"Final model saved to {model_path}")

    # Copy best model to versioned path
    import shutil
    best_model = os.path.join(save_dir, "best", "best_model.zip")
    best_vecnorm = os.path.join(save_dir, "best", "best_model_vecnormalize.pkl")
    if os.path.exists(best_model):
        best_dest = os.path.join(save_dir, versioned_name + "_best")
        shutil.copy2(best_model, best_dest + ".zip")
        if os.path.exists(best_vecnorm):
            shutil.copy2(best_vecnorm, best_dest + "_vecnormalize.pkl")
        print(f"Best model copied to {best_dest}")

    vec_env.close()
    return model_path
