"""Gymnasium environment for Vahidinia et al. 2023 paper reproduction.

Two-layer approach to mitigate cold start in serverless computing:
  Layer 1 (this env): Actor-Critic learns idle-container window (continuous action)
  Layer 2 (external):  LSTM predicts concurrent invocations for pre-warm scaling

State space : [inter_arrival_time, last_cold_start] per service
Action space: continuous idle-container window value in [min, max] seconds
Reward      : -(cold_starts / total_invocations) - weight * mem_utilization
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
        self.mem_utilization_penalty = self.gym_config.get("mem_utilization_penalty", 0.5)

        # Internal state
        self._engine: SimulationEngine | None = None
        self._monitor_api: MonitorAPI | None = None
        self._autoscaling_api: AutoscalingAPI | None = None
        self._service_ids: list[str] = []
        self._current_step = 0
        self._prev_cold_starts = 0.0
        self._prev_total = 0.0
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_latency_sum = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_total_cpu_sec = 0.0
        self._last_reward_components = {"d_total": 0.0, "d_cold": 0.0}
        self._exported = False

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
        export_mode = 0 if self._exported else self.gym_config.get("export_mode", 0)
        run_dir = self.gym_config.get("run_dir", "/tmp/vahidinia_gym_run")
        ctx = builder.build(
            config=self.sim_config,
            run_dir=run_dir,
            logger=logger,
            export_mode_override=export_mode,
        )

        self._engine = SimulationEngine(ctx)
        self._engine.setup()

        self._monitor_api = MonitorAPI(ctx.monitor_manager)
        if ctx.autoscaling_manager:
            self._autoscaling_api = AutoscalingAPI(ctx.autoscaling_manager)

        self._service_ids = list(ctx.workload_manager.services.keys())

        nodes = ctx.cluster_manager.get_enabled_nodes()
        self._cluster_memory = sum(n.capacity.memory for n in nodes)
        self._cluster_cpu = sum(n.capacity.cpu for n in nodes)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.sim_config["simulation"]["seed"] = seed
        else:
            self.sim_config["simulation"]["seed"] = int(self.np_random.integers(0, 2**31))

        self._build()
        self._current_step = 0
        self._prev_cold_starts = 0.0
        self._prev_total = 0.0
        self._prev_completed = 0.0
        self._prev_dropped = 0.0
        self._prev_latency_sum = 0.0
        self._prev_total_mem_sec = 0.0
        self._prev_total_cpu_sec = 0.0
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

        # Export before returning done=True, because DummyVecEnv auto-resets
        # on done which would rebuild the engine and lose all simulation data.
        if (terminated or truncated) and not self._exported:
            if self._engine is not None:
                self._engine.shutdown()
            self._exported = True

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
    # Reward: -cold_ratio - weight * mem_utilization
    # ------------------------------------------------------------------

    def _compute_reward(self, snapshot: dict, action: np.ndarray) -> float:
        cold_starts = float(snapshot.get("request.cold_starts", 0.0))
        total = float(snapshot.get("request.total", 0.0))
        completed = float(snapshot.get("request.completed", 0.0))
        dropped = float(snapshot.get("request.dropped", 0.0))
        latency_mean = float(snapshot.get("request.latency_mean", 0.0))

        # Deltas since last step
        d_cold = cold_starts - self._prev_cold_starts
        d_total = total - self._prev_total
        d_completed = completed - self._prev_completed
        d_dropped = dropped - self._prev_dropped
        latency_sum_now = latency_mean * completed
        d_latency_sum = latency_sum_now - self._prev_latency_sum
        step_latency_mean = (d_latency_sum / d_completed) if d_completed > 0 else 0.0

        self._prev_cold_starts = cold_starts
        self._prev_total = total
        self._prev_completed = completed
        self._prev_dropped = dropped
        self._prev_latency_sum = latency_sum_now

        # Ratios
        cold_ratio = d_cold / max(d_total, 1.0)
        drop_ratio = d_dropped / max(d_total, 1.0)

        # Resource-seconds for per-request cost
        total_mem_sec = float(snapshot.get("lifecycle.total_memory_seconds", 0.0))
        total_cpu_sec = float(snapshot.get("lifecycle.total_cpu_seconds", 0.0))

        d_total_mem = total_mem_sec - self._prev_total_mem_sec
        d_total_cpu = total_cpu_sec - self._prev_total_cpu_sec
        self._prev_total_mem_sec = total_mem_sec
        self._prev_total_cpu_sec = total_cpu_sec

        # Resource utilization (0 = idle, 1 = full cluster)
        max_mem_sec = self.step_duration * self._cluster_memory
        max_cpu_sec = self.step_duration * self._cluster_cpu
        mem_util = (d_total_mem / max_mem_sec) if max_mem_sec > 0 else 0.0
        cpu_util = (d_total_cpu / max_cpu_sec) if max_cpu_sec > 0 else 0.0

        # Resource per request (for logging)
        mem_per_req = (d_total_mem / d_completed) if d_completed > 0 else 0.0
        cpu_per_req = (d_total_cpu / d_completed) if d_completed > 0 else 0.0

        reward = -cold_ratio - self.mem_utilization_penalty * mem_util

        self._last_reward_components = {
            "cold_start_ratio": cold_ratio,
            "drop_ratio": drop_ratio,
            "mem_utilization": mem_util,
            "cpu_utilization": cpu_util,
            "latency_mean": step_latency_mean,
            "mem_per_request": mem_per_req,
            "cpu_per_request": cpu_per_req,
            "d_total": d_total,
            "d_completed": d_completed,
            "d_cold": d_cold,
            "d_dropped": d_dropped,
        }

        return float(reward)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_snapshot(self) -> dict:
        self._engine.ctx.monitor_manager.collect_once()
        return self._monitor_api.get_snapshot()

    def close(self):
        if self._engine is not None and not self._exported:
            self._engine.shutdown()
        self._engine = None
