"""Gymnasium environment for Vahidinia et al. 2023 paper reproduction.

Two-layer approach to mitigate cold start in serverless computing:
  Layer 1 (this env): Actor-Critic learns idle-container window (continuous action)
  Layer 2 (external):  LSTM predicts concurrent invocations for pre-warm scaling

State space : [inter_arrival_time, last_cold_start] per service
Action space: continuous idle-container window value in [min, max] seconds
Reward      : -(cold_starts / total_invocations) + weight * memory_efficiency
"""

from __future__ import annotations

import json
import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from serverless_sim.core.config.loader import load_config
from serverless_sim.core.simulation.sim_builder import SimulationBuilder
from serverless_sim.core.simulation.sim_engine import SimulationEngine
from serverless_sim.autoscaling.autoscaling_api import AutoscalingAPI
from serverless_sim.monitoring.monitor_api import MonitorAPI


class VahidiniaEnv(gym.Env):
    """Gymnasium env with continuous idle-window action and paper reward.

    Config (gym_config_path JSON):
        max_steps          : int   – steps per episode (default 200)
        idle_timeout_min   : float – lower bound in seconds (default 180 = 3 min)
        idle_timeout_max   : float – upper bound in seconds (default 900 = 15 min)
        memory_penalty_weight : float – weight for memory inefficiency penalty (default 1.0)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_config_path: str,
        gym_config_path: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.sim_config_path = sim_config_path
        self.sim_config = load_config(sim_config_path)

        if seed is not None:
            self.sim_config["simulation"]["seed"] = seed

        self.gym_config: dict = {}
        if gym_config_path:
            with open(gym_config_path, "r") as f:
                self.gym_config = json.load(f)

        # Step duration syncs with controller interval
        ctrl_cfg = self.sim_config.get("controller", {})
        self.step_duration = ctrl_cfg.get("interval", 5.0)
        self.max_steps = self.gym_config.get("max_steps", 200)

        # Action bounds (idle-container window in seconds)
        self.idle_timeout_min = self.gym_config.get("idle_timeout_min", 180.0)
        self.idle_timeout_max = self.gym_config.get("idle_timeout_max", 900.0)

        # Reward config
        self.memory_penalty_weight = self.gym_config.get("memory_penalty_weight", 1.0)

        # Internal state
        self._engine: SimulationEngine | None = None
        self._monitor_api: MonitorAPI | None = None
        self._autoscaling_api: AutoscalingAPI | None = None
        self._service_ids: list[str] = []
        self._current_step = 0
        self._prev_cold_starts = 0.0
        self._prev_total = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_running_mem_sec = 0.0
        self._last_reward_components = {"d_total": 0.0, "d_cold": 0.0}

        # Build once to determine spaces
        self._build()

        # Observation: [inter_arrival_time, last_cold_start] per service
        n_services = len(self._service_ids)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(n_services * 2,),
            dtype=np.float32,
        )

        # Action: one continuous value per service (idle-container window)
        self.action_space = spaces.Box(
            low=self.idle_timeout_min,
            high=self.idle_timeout_max,
            shape=(n_services,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Build / reset
    # ------------------------------------------------------------------

    def _build(self) -> None:
        logger = logging.getLogger(f"vahidinia_env_{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(
            config=self.sim_config,
            run_dir="/tmp/vahidinia_gym_run",
            logger=logger,
            export_mode_override=0,
        )

        self._engine = SimulationEngine(ctx)
        self._engine.setup()

        self._monitor_api = MonitorAPI(ctx.monitor_manager)
        if ctx.autoscaling_manager:
            self._autoscaling_api = AutoscalingAPI(ctx.autoscaling_manager)

        self._service_ids = list(ctx.workload_manager.services.keys())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.sim_config["simulation"]["seed"] = seed

        self._build()
        self._current_step = 0
        self._prev_cold_starts = 0.0
        self._prev_total = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_running_mem_sec = 0.0
        self._last_reward_components = {"d_total": 0.0, "d_cold": 0.0}

        snapshot = self._get_snapshot()
        obs = self._build_obs(snapshot)
        return obs, {"snapshot": snapshot, "step": 0}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        self._current_step += 1

        # Apply continuous action: set idle_timeout per service
        if self._autoscaling_api is not None:
            action = np.asarray(action, dtype=np.float32).flatten()
            for i, svc_id in enumerate(self._service_ids):
                value = float(np.clip(
                    action[i] if i < len(action) else action[-1],
                    self.idle_timeout_min,
                    self.idle_timeout_max,
                ))
                self._autoscaling_api.set_idle_timeout(svc_id, value)

        # Advance simulation
        ctx = self._engine.ctx
        ctx.env.run(until=ctx.env.now + self.step_duration)

        # Collect — compute reward first (updates deltas), then build obs from deltas
        snapshot = self._get_snapshot()
        reward = self._compute_reward(snapshot, action)
        obs = self._build_obs(snapshot)

        terminated = False
        truncated = self._current_step >= self.max_steps

        return obs, reward, terminated, truncated, {
            "snapshot": snapshot,
            "step": self._current_step,
            "reward_components": self._last_reward_components,
        }

    # ------------------------------------------------------------------
    # Observation: [avg_inter_arrival_time, cold_start_ratio] per service
    # ------------------------------------------------------------------

    def _build_obs(self, snapshot: dict) -> np.ndarray:
        obs = np.zeros(len(self._service_ids) * 2, dtype=np.float32)
        rc = self._last_reward_components
        for i, svc_id in enumerate(self._service_ids):
            d_total = rc.get("d_total", 0.0)
            d_cold = rc.get("d_cold", 0.0)
            # Average inter-arrival time = step_duration / n_requests
            obs[2 * i] = self.step_duration / max(d_total, 1.0)
            # Cold start ratio over the step
            obs[2 * i + 1] = d_cold / max(d_total, 1.0)
        return obs

    # ------------------------------------------------------------------
    # Reward: -cold_ratio + weight * memory_efficiency
    # ------------------------------------------------------------------

    def _compute_reward(self, snapshot: dict, action: np.ndarray) -> float:
        cold_starts = float(snapshot.get("request.cold_starts", 0.0))
        total = float(snapshot.get("request.total", 0.0))

        # Deltas since last step
        d_cold = cold_starts - self._prev_cold_starts
        d_total = total - self._prev_total
        self._prev_cold_starts = cold_starts
        self._prev_total = total

        # Cold-start ratio penalty
        cold_ratio = d_cold / max(d_total, 1.0)

        # Memory efficiency from resource-seconds (running vs total)
        total_mem_sec = float(snapshot.get("lifecycle.total_memory_seconds", 0.0))
        running_mem_sec = float(snapshot.get("lifecycle.running_memory_seconds", 0.0))

        d_total_mem = total_mem_sec - self._prev_total_mem_sec
        d_running_mem = running_mem_sec - self._prev_running_mem_sec
        self._prev_total_mem_sec = total_mem_sec
        self._prev_running_mem_sec = running_mem_sec

        # Efficiency: fraction of memory-seconds used for running
        # 0.0 = all memory wasted, 1.0 = perfectly efficient
        memory_efficiency = (d_running_mem / d_total_mem) if d_total_mem > 0 else 0.0

        reward = -cold_ratio + self.memory_penalty_weight * memory_efficiency

        self._last_reward_components = {
            "cold_start_ratio": cold_ratio,
            "memory_efficiency": memory_efficiency,
            "d_cold": d_cold,
            "d_total": d_total,
        }

        return float(reward)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_snapshot(self) -> dict:
        self._engine.ctx.monitor_manager.collect_once()
        return self._monitor_api.get_snapshot()

    def close(self):
        self._engine = None
