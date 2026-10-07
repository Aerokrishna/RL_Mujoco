"""ActionManager and the built-in action terms.

Policy actions are float32 values in [-1, 1]. The manager clips them into
`ctx.action` (float64, NaN-safe), keeps `ctx.prev_action`, and calls the active term's
`apply`, which converts them to controller targets via `controller.set_target(...)`.
This happens once per policy step; the controller then tracks the target for
`decimation` physics ticks.

Built-in terms (registered with `@action_term`):
- `delta_ee_pose`: TCP delta (3 or 6 dims) relative to the current TCP pose.
- `absolute_ee_pose`: [-1, 1] mapped to the workspace box (+ rotation about the reset pose).
- `delta_ee_pose_with_stiffness`: `delta_ee_pose` plus stiffness dims mapped to a log range
  of Kp; Kd is set to critical damping.
- `delta_joint_pos`: joint delta for `JointImpedance`.
- `anchor_relative_pos`: FORGE-style target relative to a task anchor (e.g. the hole tip),
  clipped to within λ of the current TCP; orientation held at the reset pose.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import gymnasium as gym
import numpy as np

from mujoco_rl_bed.control.math_utils import QuatBuffers, apply_rotvec, critical_damping
from mujoco_rl_bed.env.cfg import ActionCfg

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context

ACTION_TERMS: dict[str, type["ActionTerm"]] = {}


def action_term(name: str) -> Callable[[type["ActionTerm"]], type["ActionTerm"]]:
    """Class decorator registering an action term.

    Args:
        name: Unique term name used in `ActionCfg.term`.

    Returns:
        The decorator (returns the class unchanged).
    """
    def deco(cls: type["ActionTerm"]) -> type["ActionTerm"]:
        if name in ACTION_TERMS:
            raise KeyError(f"Action term '{name}' already registered")
        ACTION_TERMS[name] = cls
        cls.name = name
        return cls
    return deco


class ActionTerm:
    """Base class: maps a clipped action vector to controller targets.

    Attributes:
        name: Registered name.
        dim: Action dimension.
    """

    name: str = ""
    dim: int = 0

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Store config and context.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        self.cfg = cfg
        self.ctx = ctx

    def apply(self, a: np.ndarray) -> None:
        """Convert the action into controller targets.

        Args:
            a: Clipped action in [-1, 1], float64, shape (dim,).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Per-episode reset hook (called after the controller reset)."""

    def write_applied(self, out: np.ndarray) -> None:
        """Write the action actually applied (after smoothing etc.) into `out`.

        Default: the clipped policy action itself.

        Args:
            out: Buffer of shape (dim,).
        """
        np.copyto(out, self.ctx.action)


@action_term("delta_ee_pose")
class DeltaEEPose(ActionTerm):
    """TCP pose delta relative to the current TCP pose (world frame).

    Layout: [dx, dy, dz] (scaled by `pos_scale` [m]) and, if `rotation`, [rx, ry, rz]
    (world-frame axis-angle scaled by `rot_scale` [rad]). The target position is clipped
    to the workspace box. Without rotation, orientation is held at the reset pose.
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Allocate buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        self.rot = cfg.rotation
        self.pose_dim = 6 if self.rot else 3
        self.dim = self.pose_dim
        self._lo = np.asarray(cfg.pos_lo, dtype=np.float64)
        self._hi = np.asarray(cfg.pos_hi, dtype=np.float64)
        self._span = self._hi - self._lo
        self._pos = np.zeros(3)
        self._quat = np.array([1.0, 0, 0, 0])
        self._rotvec = np.zeros(3)
        self._nominal_quat = np.array([1.0, 0, 0, 0])
        self._qb = QuatBuffers()

    def _pose_target(self, a: np.ndarray) -> None:
        """Compute the pose target from the first `pose_dim` action entries.

        Args:
            a: Clipped action, shape (dim,).
        """
        plant = self.ctx.plant
        np.multiply(a[:3], self.cfg.pos_scale, out=self._pos)
        np.add(self._pos, plant.ee_pos, out=self._pos)
        np.clip(self._pos, self._lo, self._hi, out=self._pos)
        if self.rot:
            np.multiply(a[3:6], self.cfg.rot_scale, out=self._rotvec)
            apply_rotvec(self._quat, plant.ee_quat(), self._rotvec, self._qb)
        else:
            np.copyto(self._quat, self._nominal_quat)

    def apply(self, a: np.ndarray) -> None:
        """Set the controller pose target.

        Args:
            a: Clipped action, shape (dim,).
        """
        self._pose_target(a)
        self.ctx.controller.set_target(pos=self._pos, quat=self._quat)

    def reset(self) -> None:
        """Remember the reset orientation (held when `rotation=False`)."""
        np.copyto(self._nominal_quat, self.ctx.plant.ee_quat())


@action_term("absolute_ee_pose")
class AbsoluteEEPose(DeltaEEPose):
    """Absolute TCP target: position = box lerp of [-1, 1]; rotation = offset from the reset pose.

    The position dims map linearly from [-1, 1] to [pos_lo, pos_hi]. The rotation dims are
    a world-frame axis-angle offset (scaled by `rot_scale` [rad]) from the reset orientation.
    """

    def _pose_target(self, a: np.ndarray) -> None:
        """Map the action to an absolute pose.

        Args:
            a: Clipped action, shape (dim,).
        """
        # pos = lo + (a + 1) / 2 * (hi - lo)
        np.add(a[:3], 1.0, out=self._pos)
        self._pos *= 0.5
        self._pos *= self._span
        self._pos += self._lo
        if self.rot:
            np.multiply(a[3:6], self.cfg.rot_scale, out=self._rotvec)
            apply_rotvec(self._quat, self._nominal_quat, self._rotvec, self._qb)
        else:
            np.copyto(self._quat, self._nominal_quat)


@action_term("delta_ee_pose_with_stiffness")
class DeltaEEPoseWithStiffness(DeltaEEPose):
    """`delta_ee_pose` plus variable stiffness.

    Extra dims (3 translational, plus 3 rotational if `rotation`) map from [-1, 1] to a log
    range: kp = exp(log lo + (a + 1) / 2 (log hi - log lo)). Kd = 2 sqrt(Kp). Without
    rotation, the rotational stiffness stays at the controller's nominal value.
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Allocate gain buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        self.n_k = 6 if self.rot else 3
        self.dim = self.pose_dim + self.n_k
        lp, hp = (math.log(v) for v in cfg.kp_pos_range)
        lr, hr = (math.log(v) for v in cfg.kp_rot_range)
        self._log_lo = np.array([lp] * 3 + [lr] * 3)[: self.n_k]
        self._log_span = np.array([hp - lp] * 3 + [hr - lr] * 3)[: self.n_k]
        self._kp = np.zeros(6)
        self._kd = np.zeros(6)
        self._kp_var = self._kp[: self.n_k]

    def apply(self, a: np.ndarray) -> None:
        """Set the pose target and gains.

        Args:
            a: Clipped action, shape (dim,).
        """
        self._pose_target(a)
        np.add(a[self.pose_dim:], 1.0, out=self._kp_var)
        self._kp_var *= 0.5
        self._kp_var *= self._log_span
        self._kp_var += self._log_lo
        np.exp(self._kp_var, out=self._kp_var)
        critical_damping(self._kp, out=self._kd)
        self.ctx.controller.set_target(pos=self._pos, quat=self._quat, kp=self._kp, kd=self._kd)

    def reset(self) -> None:
        """Start from the controller's nominal gains."""
        super().reset()
        np.copyto(self._kp, self.ctx.controller.kp_nominal)


@action_term("delta_joint_pos")
class DeltaJointPos(ActionTerm):
    """Joint position delta (for `JointImpedance`): q_d = q + joint_scale * a, clipped to joint ranges."""

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Allocate buffers and cache joint ranges.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        self.dim = ctx.plant.n
        jids = [ctx.handles.joint_ids[n] for n in ctx.handles.arm_joint_names]
        self._lo = ctx.model.jnt_range[jids, 0].copy()
        self._hi = ctx.model.jnt_range[jids, 1].copy()
        self._q = np.zeros(self.dim)

    def apply(self, a: np.ndarray) -> None:
        """Set the joint target.

        Args:
            a: Clipped action, shape (n,).
        """
        np.multiply(a, self.cfg.joint_scale, out=self._q)
        np.add(self._q, self.ctx.plant.q, out=self._q)
        np.clip(self._q, self._lo, self._hi, out=self._q)
        self.ctx.controller.set_target(q=self._q)


@action_term("anchor_relative_pos")
class AnchorRelativePos(ActionTerm):
    """FORGE action (paper Eq. 5), position only.

        p_targ = clip(anchor + a * anchor_bounds, p_ee - λ, p_ee + λ)

    The anchor is read from `ctx.state[cfg.anchor]` (written by reset events, e.g. the
    noisy hole-tip estimate). λ is read from `ctx.state["action_max_step"]` (initialized to
    `cfg.max_step`; events may randomize it). The target is also clipped to the workspace box. Orientation
    is held at the reset orientation.

    With `cfg.success_prediction`, a 4th dim a_ET is mapped to p = (a_ET + 1) / 2 and stored in
    `ctx.state["pred_success"][0]` (it does not affect the controller).

    With `cfg.ema_factor` = alpha < 1, the position part is smoothed before use:
    s_t = alpha * a_t + (1 - alpha) * s_{t-1}, starting from the action that holds the reset pose.
    The success prediction is not smoothed.
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Allocate buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        self.predict = bool(cfg.success_prediction)
        self.dim = 4 if self.predict else 3
        self._pred = ctx.buffer("pred_success", 1)
        self._bounds = np.asarray(cfg.anchor_bounds, dtype=np.float64)
        # λ lives in a state buffer so reset events can randomize it per episode (FORGE DR).
        self._lam_buf = ctx.buffer("action_max_step", 1)
        self._lam_buf[0] = float(cfg.max_step)
        self._lo = np.asarray(cfg.pos_lo, dtype=np.float64)
        self._hi = np.asarray(cfg.pos_hi, dtype=np.float64)
        self._anchor = ctx.buffer(cfg.anchor, 3)
        self._pos = np.zeros(3)
        self._tmp_lo = np.zeros(3)
        self._tmp_hi = np.zeros(3)
        self._quat = np.array([1.0, 0, 0, 0])
        self._alpha = float(cfg.ema_factor)
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"ema_factor must be in (0, 1], got {self._alpha}")
        self._smooth = np.zeros(3)   # smoothed position action s_t
        self._tmp3 = np.zeros(3)

    def apply(self, a: np.ndarray) -> None:
        """Set the controller position target.

        Args:
            a: Clipped action in [-1, 1], shape (3,) or (4,) with success prediction.
        """
        ee = self.ctx.plant.ee_pos
        if self.predict:
            self._pred[0] = 0.5 * (a[3] + 1.0)
        if self._alpha < 1.0:
            # s_t = alpha * a_t + (1 - alpha) * s_{t-1}
            np.multiply(a[:3], self._alpha, out=self._tmp3)
            self._smooth *= (1.0 - self._alpha)
            self._smooth += self._tmp3
        else:
            np.copyto(self._smooth, a[:3])
        np.multiply(self._smooth, self._bounds, out=self._pos)
        np.add(self._pos, self._anchor, out=self._pos)
        lam = self._lam_buf[0]
        np.subtract(ee, lam, out=self._tmp_lo)
        np.add(ee, lam, out=self._tmp_hi)
        np.clip(self._pos, self._tmp_lo, self._tmp_hi, out=self._pos)   # within λ of the EE
        np.clip(self._pos, self._lo, self._hi, out=self._pos)           # workspace box
        self.ctx.controller.set_target(pos=self._pos, quat=self._quat)

    def reset(self) -> None:
        """Hold the reset orientation; clear the success prediction; start smoothing at the hold action."""
        np.copyto(self._quat, self.ctx.plant.ee_quat())
        self._pred[0] = 0.0
        # Action that keeps the TCP where it is, so the first smoothed steps do not pull it away.
        np.subtract(self.ctx.plant.ee_pos, self._anchor, out=self._smooth)
        np.divide(self._smooth, self._bounds, out=self._smooth)
        np.clip(self._smooth, -1.0, 1.0, out=self._smooth)

    def write_applied(self, out: np.ndarray) -> None:
        """Applied action: smoothed position dims (+ the raw success prediction).

        Args:
            out: Buffer of shape (dim,).
        """
        out[:3] = self._smooth
        if self.predict:
            out[3] = self.ctx.action[3]


class ActionManager:
    """Owns the active action term and the action space."""

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Instantiate the configured term and allocate `ctx.action`.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        if cfg.term not in ACTION_TERMS:
            raise KeyError(f"Unknown action term '{cfg.term}'. Registered: {sorted(ACTION_TERMS)}")
        self.ctx = ctx
        self.term = ACTION_TERMS[cfg.term](cfg, ctx)
        self.dim = self.term.dim
        self.space = gym.spaces.Box(-1.0, 1.0, shape=(self.dim,), dtype=np.float32)
        ctx.action = np.zeros(self.dim)
        ctx.prev_action = np.zeros(self.dim)
        ctx.applied_action = np.zeros(self.dim)
        self._first = True

    def apply(self, action: np.ndarray) -> None:
        """Clip the policy action and forward it to the term.

        Args:
            action: Raw policy action, shape (dim,), any float dtype.
        """
        ctx = self.ctx
        np.copyto(ctx.prev_action, ctx.action)
        np.clip(action, -1.0, 1.0, out=ctx.action)
        np.nan_to_num(ctx.action, copy=False, nan=0.0)
        if self._first:  # no action history yet: do not count the first action as a "change"
            np.copyto(ctx.prev_action, ctx.action)
            self._first = False
        self.term.apply(ctx.action)
        self.term.write_applied(ctx.applied_action)

    def reset(self) -> None:
        """Zero the action history and reset the term."""
        self.ctx.action.fill(0.0)
        self.ctx.prev_action.fill(0.0)
        self.term.reset()
        self.term.write_applied(self.ctx.applied_action)
        self._first = True
