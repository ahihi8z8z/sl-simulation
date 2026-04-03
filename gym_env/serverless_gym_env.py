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
from gym_env.observation_builder import ObservationBuilder
from gym_env.action_mapper import ActionMapper
from gym_env.reward_calculator import RewardCalculator


class ServerlessGymEnv(gym.Env):
    """Gymnasium wrapper for the serverless simulator."""

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

        # Override seed if provided
        if seed is not None:
            self.sim_config["simulation"]["seed"] = seed

        # Load gym config
        self.gym_config = {}
        if gym_config_path:
            with open(gym_config_path, "r") as f:
                self.gym_config = json.load(f)

        # Extract gym parameters
        # step_duration syncs with controller interval (one control cycle per step)
        ctrl_cfg = self.sim_config.get("controller", {})
        self.step_duration = ctrl_cfg.get("interval", 5.0)
        self.max_steps = self.gym_config.get("max_steps", 100)

        # Build components (deferred to reset)
        self._engine: SimulationEngine | None = None
        self._monitor_api: MonitorAPI | None = None
        self._autoscaling_api: AutoscalingAPI | None = None
        self._obs_builder: ObservationBuilder | None = None
        self._action_mapper: ActionMapper | None = None
        self._reward_calc: RewardCalculator | None = None

        self._current_step = 0

        # Build once to determine spaces
        self._build()

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_builder.obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._action_mapper.n_actions)

    def _build(self) -> None:
        """Build or rebuild the simulation."""
        logger = logging.getLogger(f"gym_env_{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        ctx = builder.build(
            config=self.sim_config,
            run_dir="/tmp/gym_run",
            logger=logger,
            export_mode_override=0,
        )

        self._engine = SimulationEngine(ctx)
        self._engine.setup()

        self._monitor_api = MonitorAPI(ctx.monitor_manager)
        if ctx.autoscaling_manager:
            self._autoscaling_api = AutoscalingAPI(ctx.autoscaling_manager)

        # Observation builder
        obs_metrics = self.gym_config.get("observation_metrics", None)
        self._obs_builder = ObservationBuilder(metric_names=obs_metrics,
                                                  step_duration=self.step_duration)

        # Action mapper
        service_ids = list(ctx.workload_manager.services.keys())
        self._action_mapper = ActionMapper(
            service_ids=service_ids,
            prewarm_max=self.gym_config.get("prewarm_max", 10),
            idle_timeout_max=self.gym_config.get("idle_timeout_max", 120.0),
        )

        # Reward calculator — compute cluster totals for utilization
        nodes = ctx.cluster_manager.get_enabled_nodes()
        cluster_memory = sum(n.capacity.memory for n in nodes)
        cluster_cpu = sum(n.capacity.cpu for n in nodes)

        reward_cfg = self.gym_config.get("reward", {})
        self._reward_calc = RewardCalculator(
            step_duration=self.step_duration,
            cluster_memory=cluster_memory,
            cluster_cpu=cluster_cpu,
            **reward_cfg,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.sim_config["simulation"]["seed"] = seed

        self._build()
        self._current_step = 0
        self._reward_calc.reset()

        # Initial observation
        snapshot = self._get_snapshot()
        obs = self._obs_builder.build(snapshot)
        info = {"snapshot": snapshot, "step": 0}
        return obs, info

    def step(self, action: int):
        self._current_step += 1

        # Apply action
        if self._autoscaling_api:
            self._action_mapper.apply(action, self._autoscaling_api)

        # Advance simulation
        ctx = self._engine.ctx
        target_time = ctx.env.now + self.step_duration
        ctx.env.run(until=target_time)

        # Collect metrics
        snapshot = self._get_snapshot()
        obs = self._obs_builder.build(snapshot)
        reward = self._reward_calc.compute(snapshot)

        terminated = False
        truncated = self._current_step >= self.max_steps

        info = {
            "snapshot": snapshot,
            "step": self._current_step,
            "reward_components": self._reward_calc.last_components,
        }

        return obs, reward, terminated, truncated, info

    def _get_snapshot(self) -> dict:
        """Collect a fresh metric snapshot."""
        self._engine.ctx.monitor_manager.collect_once()
        return self._monitor_api.get_snapshot()

    def close(self):
        self._engine = None
