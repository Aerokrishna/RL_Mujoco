"""TorqueEnv: Gymnasium env over the torque-controlled Franka, configured by a `TaskCfg`.

Per policy step:

    events.step()                       # interval events (if any)
    action_mgr.apply(action)            # -> controller.set_target(...)
    for _ in range(decimation):         # hot loop: no allocation, no dicts, no logging
        plant.control_step(controller.torque)
        obs_mgr.accumulate()            # only when a term needs per-tick averaging
    events.post_step()                  # task bookkeeping shared by terms (if any)
    obs = obs_mgr.compute("policy"); reward; termination; viewer.sync() if render

The observation returned to the caller is a copy of the manager's buffer. Vectorized
envs store terminal observations by reference before calling `reset`, so returning
the shared buffer would corrupt them. This costs one small copy per policy step.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

import mujoco_rl_bed.terms  # noqa: F401  (registers the built-in terms)
from mujoco_rl_bed.env.cfg import EnvCfg
from mujoco_rl_bed.env.context import Context
from mujoco_rl_bed.env.managers.action import ActionManager
from mujoco_rl_bed.env.managers.events import EventManager
from mujoco_rl_bed.env.managers.observation import ObservationManager
from mujoco_rl_bed.env.managers.reward import RewardManager
from mujoco_rl_bed.env.managers.termination import TerminationManager
from mujoco_rl_bed.sim.plant import FrankaPlant
from mujoco_rl_bed.sim.scene import SceneBuilder
from mujoco_rl_bed.sim.viewer import make_viewer


class TorqueEnv(gym.Env):
    """Torque-controlled Franka env.

    Attributes:
        cfg: Resolved `EnvCfg`.
        ctx: Shared `Context`.
        observation_space: Box over the flat `policy` group, or over [policy | critic] when
            `cfg.obs_mode == "asymmetric"` (float32).
        policy_obs_dim: Size of the policy (actor) part of the observation.
        action_space: Box(-1, 1) of the action term (float32).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, cfg: EnvCfg) -> None:
        """Build the scene, plant, controller and managers.

        Args:
            cfg: Environment configuration.
        """
        super().__init__()
        self.cfg = cfg
        task = cfg.task
        if cfg.decimation < 1:
            raise ValueError("decimation must be >= 1")

        model, handles = SceneBuilder(task.scene, sim_dt=cfg.sim_dt).build()
        plant = FrankaPlant(model, handles)
        plant.reset()
        controller = task.controller.build(plant)
        policy_dt = cfg.policy_dt
        self.ctx = Context(
            cfg=cfg, model=model, data=plant.data, plant=plant, handles=handles, controller=controller,
            rng=np.random.default_rng(cfg.seed), policy_dt=policy_dt,
            max_episode_steps=max(1, round(task.episode_length_s / policy_dt)),
        )
        ctx = self.ctx

        # Order matters: actions allocate ctx.action, which `last_action` sizes itself from.
        self.action_mgr = ActionManager(task.action, ctx)
        self.event_mgr = EventManager(task.events, ctx)
        self.obs_mgr = ObservationManager(task.obs, ctx)
        ctx.obs_mgr = self.obs_mgr
        self.reward_mgr = RewardManager(task.rewards, ctx)
        self.term_mgr = TerminationManager(task.terminations, ctx)
        if not self.obs_mgr.has_group("policy"):
            raise ValueError("ObsCfg.groups must define a 'policy' group")

        if cfg.obs_mode not in ("policy", "asymmetric"):
            raise ValueError(f"obs_mode must be 'policy' or 'asymmetric', got '{cfg.obs_mode}'")
        self._asym = cfg.obs_mode == "asymmetric"
        if self._asym and not self.obs_mgr.has_group("critic"):
            raise ValueError("obs_mode='asymmetric' needs a 'critic' observation group in the task")
        self.policy_obs_dim = self.obs_mgr.dim("policy")
        obs_dim = self.policy_obs_dim + (self.obs_mgr.dim("critic") if self._asym else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = self.action_mgr.space

        # Cached hot-loop handles.
        self._plant = plant
        self._torque = controller.torque
        self._decimation = int(cfg.decimation)
        self._accumulate = self.obs_mgr.accumulate if self.obs_mgr.needs_accumulation else None
        self._critic = cfg.critic_in_info and self.obs_mgr.has_group("critic") and not self._asym
        self._render = bool(cfg.render)
        self._viewer = make_viewer(model, plant.data, enabled=self._render, realtime=cfg.realtime)

        self.event_mgr.startup()

    # ------------------------------------------------------------------ properties
    @property
    def plant(self) -> FrankaPlant:
        """The plant (state accessors / torque API)."""
        return self._plant

    @property
    def controller(self) -> Any:
        """The low-level controller."""
        return self.ctx.controller

    @property
    def policy_dt(self) -> float:
        """Policy period [s]."""
        return self.ctx.policy_dt

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Reset the episode.

        Order: plant.reset(q_home) -> reset events -> mj_forward -> controller.reset
        (holds the current pose) -> action reset -> `step` events -> reward/termination/obs resets.

        Args:
            seed: If given, reseed the env RNG (`np.random.default_rng(seed)`).
            options: Unused (reserved).

        Returns:
            (policy observation copy, info). `info["critic_obs"]` holds the critic group if defined.
        """
        super().reset(seed=seed)
        ctx = self.ctx
        if seed is not None:
            ctx.rng = np.random.default_rng(seed)
        ctx.episode_step = 0
        self._plant.reset()
        self.event_mgr.reset()
        mujoco.mj_forward(ctx.model, ctx.data)
        ctx.controller.reset(self._plant)
        self.action_mgr.reset()
        self.event_mgr.post_step()
        self.reward_mgr.reset()
        self.term_mgr.reset()
        self.obs_mgr.reset()
        self._viewer.reset_clock()

        info: dict[str, Any] = {}
        if self._critic:
            info["critic_obs"] = self.obs_mgr.compute("critic").copy()
        if self._render:
            self._viewer.sync()
        return self._observe(), info

    def _observe(self) -> np.ndarray:
        """Compute the returned observation (a fresh array; manager buffers are reused).

        Returns:
            The policy group, or [policy | critic] in asymmetric mode, float32.
        """
        if self._asym:
            return np.concatenate((self.obs_mgr.compute("policy"), self.obs_mgr.compute("critic")))
        return self.obs_mgr.compute("policy").copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one policy step (`decimation` physics ticks).

        Args:
            action: Policy action in [-1, 1], shape (action_dim,).

        Returns:
            (obs, reward, terminated, truncated, info). On episode end, `info` contains
            `reward_terms` (per-term weighted sums), `termination` (term name), `is_success`.
        """
        ctx = self.ctx
        if self.event_mgr.has_interval:
            self.event_mgr.step()
        self.action_mgr.apply(action)

        plant, torque = self._plant, self._torque
        if self._accumulate is None:
            for _ in range(self._decimation):
                plant.control_step(torque)
        else:
            acc = self._accumulate
            for _ in range(self._decimation):
                plant.control_step(torque)
                acc()
        ctx.episode_step += 1
        if self.event_mgr.has_post_step:
            self.event_mgr.post_step()

        obs = self._observe()
        reward = self.reward_mgr.compute()
        terminated, truncated = self.term_mgr.compute()

        info: dict[str, Any] = {}
        if self._critic:
            info["critic_obs"] = self.obs_mgr.compute("critic").copy()
        if terminated or truncated:
            info["reward_terms"] = self.reward_mgr.episode_sums()
            info["termination"] = self.term_mgr.fired
            info["is_success"] = self.term_mgr.episode_success
            for hook in ctx.episode_info_hooks:
                info.update(hook(ctx))
        if self._render:
            self._viewer.sync()
        return obs, reward, terminated, truncated, info

    def compute_obs(self, group: str) -> np.ndarray:
        """Compute an extra observation group (e.g. `dataset`) for the current state.

        Note: advances that group's history by one slot.

        Args:
            group: Group name.

        Returns:
            Copy of the group's flat float32 buffer.
        """
        return self.obs_mgr.compute(group).copy()

    def viewer_is_running(self) -> bool:
        """Whether the viewer window is open (always True when rendering is disabled).

        Returns:
            False once the user has closed the window.
        """
        return self._viewer.is_running()

    def close(self) -> None:
        """Close the viewer (if any)."""
        self._viewer.close()
