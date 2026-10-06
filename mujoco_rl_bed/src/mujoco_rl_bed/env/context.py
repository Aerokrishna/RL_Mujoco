"""Context: the single object every term function receives.

It bundles the plant, scene handles, controller, RNG and per-episode task state, so
term signatures stay uniform (`fn(ctx, ...)`) and terms never need to know about the env.

Task state lives in `ctx.state`, a dict of named numpy buffers created on first use
through `ctx.buffer(name, shape)`. Writers (events) and readers (obs/rewards) share
the same array, so nothing is reallocated per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from mujoco_rl_bed.sim.plant import FrankaPlant
from mujoco_rl_bed.sim.scene import SceneHandles

if TYPE_CHECKING:
    from mujoco_rl_bed.control.base import Controller
    from mujoco_rl_bed.env.cfg import EnvCfg


@dataclass(eq=False)
class Context:
    """Shared runtime state passed to all terms.

    Attributes:
        cfg: Resolved `EnvCfg`.
        model: Env-owned `MjModel` (events may randomize it in place).
        data: The env's `MjData`.
        plant: Torque/state interface.
        handles: Resolved scene ids.
        controller: Low-level controller.
        rng: Env RNG (`np.random.default_rng`); reseeded on `reset(seed=...)`.
        policy_dt: Policy period [s].
        max_episode_steps: Episode length in policy steps.
        episode_step: Policy steps taken in the current episode.
        action: Current clipped action, float64, shape (action_dim,) (set by ActionManager).
        prev_action: Previous action, same shape.
        applied_action: Action actually applied after term-side processing (e.g. EMA smoothing),
            same shape; equals `action` for terms without smoothing. Observed via `last_action`.
        state: Named task buffers (targets, success flags, nominal model params, ...).
        episode_info_hooks: Callables `hook(ctx) -> dict` merged into `info` at episode end
            (task metrics such as contact-force statistics).
        obs_mgr: The env's `ObservationManager` (set by `TorqueEnv`), e.g. for
            `obs_mgr.accumulated(name)` step averages in rewards/metrics.
    """

    cfg: "EnvCfg"
    model: mujoco.MjModel
    data: mujoco.MjData
    plant: FrankaPlant
    handles: SceneHandles
    controller: "Controller"
    rng: np.random.Generator
    policy_dt: float
    max_episode_steps: int
    episode_step: int = 0
    action: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(0))
    applied_action: np.ndarray = field(default_factory=lambda: np.zeros(0))
    state: dict[str, Any] = field(default_factory=dict)
    episode_info_hooks: list[Callable[["Context"], dict]] = field(default_factory=list)
    obs_mgr: Any = None

    def buffer(self, name: str, shape: int | tuple[int, ...], dtype: Any = np.float64) -> np.ndarray:
        """Get or create a named state buffer (zero-initialized on creation).

        Args:
            name: Buffer name (e.g. "target_pos").
            shape: Buffer shape.
            dtype: Buffer dtype.

        Returns:
            The persistent array stored in `self.state[name]`.
        """
        buf = self.state.get(name)
        if buf is None:
            buf = np.zeros(shape, dtype=dtype)
            self.state[name] = buf
        return buf

    def body_id(self, name: str) -> int:
        """Resolve a body name to its id (init/reset-time helper; raises on unknown names).

        Args:
            name: Body name.

        Returns:
            Body id.
        """
        try:
            return self.handles.body_ids[name]
        except KeyError:
            raise KeyError(f"Unknown body '{name}'. Known: {sorted(self.handles.body_ids)}") from None
