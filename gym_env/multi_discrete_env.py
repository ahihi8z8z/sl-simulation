"""Gymnasium environment with MultiDiscrete action space.

Action: set pool_target per pool state + idle_timeout (in minutes) per service.
Observation: configurable metrics from monitor snapshot.
Reward: configurable (default: cold_start + memory efficiency penalty).
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
from gym_env.observation_builder import ObservationBuilder
from gym_env.multi_action_mapper import MultiActionMapper
from gym_env.reward_calculator import RewardCalculator


class MultiDiscreteEnv(gym.Env):
    """Gymnasium env with MultiDiscrete action space.

    Config (gym_config_path JSON):
        max_steps              : int (default 200)
        pool_target_max        : int (default 10)
        idle_timeout_max_minutes : int (default 10)
        observation_metrics    : list[str] (optional)
        reward                 : dict (optional, same as RewardCalculator)
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

        ctrl_cfg = self.sim_config.get("controller", {})
        self.step_duration = ctrl_cfg.get("interval", 5.0)
        self.max_steps = self.gym_config.get("max_steps", 200)
        self.flatten_action = self.gym_config.get("flatten_action", False)
        self.continuous_action = self.gym_config.get("continuous_action", False)

        self._engine: SimulationEngine | None = None
        self._monitor_api: MonitorAPI | None = None
        self._autoscaling_api: AutoscalingAPI | None = None
        self._obs_builder: ObservationBuilder | None = None
        self._action_mapper: MultiActionMapper | None = None
        self._reward_calc: RewardCalculator | None = None
        self._current_step = 0
        self._exported = False

        # Build once to determine spaces
        self._build()

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_builder.obs_size,),
            dtype=np.float32,
        )

        # Action space
        if self.continuous_action:
            # Softmax layout: mapper.n_dims = sum_svc(1 + N_pool_states + 1).
            # Box(-1, 1) — all scaling (total, logits→softmax, idle_timeout)
            # happens in mapper._apply_softmax.
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self._action_mapper.n_dims,),
                dtype=np.float32,
            )
        elif self.flatten_action:
            self.action_space = spaces.Discrete(self._action_mapper.flat_n_actions)
        else:
            self.action_space = spaces.MultiDiscrete(self._action_mapper.dimensions)

    def _build(self) -> None:
        logger = logging.getLogger(f"multi_env_{id(self)}")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

        builder = SimulationBuilder()
        # After export, disable export to prevent auto-reset from truncating files
        export_mode = 0 if self._exported else self.gym_config.get("export_mode", 0)
        run_dir = self.gym_config.get("run_dir", "/tmp/multi_gym_run")
        ctx = builder.build(
            config=self.sim_config,
            run_dir=run_dir,
            logger=logger,
            export_mode_override=export_mode,
        )

        self._engine = SimulationEngine(ctx)
        self._engine.setup()
        ctx.workload_manager.start()

        self._monitor_api = MonitorAPI(ctx.monitor_manager)
        if ctx.autoscaling_manager:
            self._autoscaling_api = AutoscalingAPI(ctx.autoscaling_manager)

        # Observation builder
        obs_metrics = self.gym_config.get("observation_metrics", None)
        self._obs_builder = ObservationBuilder(metric_names=obs_metrics,
                                                  step_duration=self.step_duration)

        # Action mapper — use config override or discover from autoscaler
        service_ids = list(ctx.workload_manager.services.keys())
        pool_states_override = self.gym_config.get("pool_states", None)
        pool_states = {}
        for svc_id in service_ids:
            if pool_states_override:
                pool_states[svc_id] = pool_states_override
            elif ctx.autoscaling_manager:
                states = ctx.autoscaling_manager._get_pool_states(svc_id)
                pool_states[svc_id] = states if states else ["prewarm"]
            else:
                pool_states[svc_id] = ["prewarm"]

        self._action_mapper = MultiActionMapper(
            service_ids=service_ids,
            pool_states=pool_states,
            pool_target_max=self.gym_config.get("pool_target_max", 10),
            idle_timeout_max_minutes=self.gym_config.get("idle_timeout_max_minutes", 10),
            delta_max=self.gym_config.get("delta_max", 0),
            softmax=self.continuous_action,
        )

        reward_cfg = self.gym_config.get("reward", {})
        self._reward_calc = RewardCalculator(**reward_cfg)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim_config["simulation"]["seed"] = seed
        else:
            # Random seed each episode for variation in transition sampling
            self.sim_config["simulation"]["seed"] = int(self.np_random.integers(0, 2**31))

        self._build()
        self._current_step = 0
        self._reward_calc.reset()

        snapshot = self._get_snapshot()
        obs = self._obs_builder.build(snapshot)
        return obs, {"snapshot": snapshot, "step": 0}

    def step(self, action: np.ndarray):
        self._current_step += 1

        if self._autoscaling_api:
            self._action_mapper.apply(action, self._autoscaling_api)

        ctx = self._engine.ctx
        ctx.env.run(until=ctx.env.now + self.step_duration)

        snapshot = self._get_snapshot()
        obs = self._obs_builder.build(snapshot)
        reward = self._reward_calc.compute(snapshot)

        terminated = False
        truncated = self._current_step >= self.max_steps

        # Export before returning done=True, because DummyVecEnv auto-resets
        # on done which would rebuild the engine and lose all simulation data.
        if (terminated or truncated) and not self._exported:
            self._engine.shutdown()
            self._exported = True

        return obs, reward, terminated, truncated, {
            "snapshot": snapshot,
            "step": self._current_step,
            "reward_components": self._reward_calc.last_components,
        }

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for each dimension.

        For MultiDiscrete: concatenated boolean arrays, one per dimension.
        For flattened Discrete: single boolean array of size n_actions.

        Masks pool_target values that exceed available memory on the cluster.
        All idle_timeout values are always valid.
        """
        ctx = self._engine.ctx
        mapper = self._action_mapper

        # Calculate max containers cluster can hold (based on flavor capacity)
        nodes = ctx.cluster_manager.get_enabled_nodes()
        total_available_mem = sum(n.capacity.memory - n.flavor_memory_used for n in nodes)

        masks = []
        for i, (svc_id, action_type, state) in enumerate(mapper._action_map):
            dim_size = mapper.dimensions[i]

            if action_type == "idle_timeout":
                # All idle_timeout values valid
                masks.append(np.ones(dim_size, dtype=bool))
            else:
                # pool_target: mask values exceeding available memory
                service = ctx.workload_manager.services[svc_id]
                mem_per_container = service.peak_memory
                max_affordable = int(total_available_mem / mem_per_container) if mem_per_container > 0 else dim_size
                mask = np.zeros(dim_size, dtype=bool)
                mask[:min(max_affordable + 1, dim_size)] = True
                masks.append(mask)

        if self.flatten_action:
            # Flatten: cartesian product of per-dimension masks
            flat_mask = masks[0]
            for m in masks[1:]:
                flat_mask = np.outer(flat_mask, m).flatten()
            return flat_mask.astype(bool)
        else:
            return np.concatenate(masks)

    def _get_snapshot(self) -> dict:
        self._engine.ctx.monitor_manager.collect_once()
        return self._monitor_api.get_snapshot()

    def close(self):
        if self._engine is not None and not self._exported:
            self._engine.shutdown()
        self._engine = None
