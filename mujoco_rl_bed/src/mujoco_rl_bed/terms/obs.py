"""Reusable observation terms.

Each term has the signature `fn(ctx, out) -> None` and writes float64 values into
`out` (shape (dim,)) without allocating. Poses are in the world frame, which coincides
with the robot base frame (`link0` sits at the world origin in the provided MJCF).
Quaternions are (w, x, y, z), canonicalized to w >= 0 so the policy never sees the
q / -q sign flip.

Task-specific state (e.g. `target_pos`) is read from `ctx.state` buffers written by events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mujoco_rl_bed.env.managers.observation import obs_term

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context


def _n_arm(ctx: "Context") -> int:
    """Number of arm joints (dim callable)."""
    return ctx.plant.n


def _action_dim(ctx: "Context") -> int:
    """Action dimension (dim callable)."""
    return int(ctx.action.size)


def _canonical_quat(out: np.ndarray) -> None:
    """Flip a quaternion in place so that w >= 0.

    Args:
        out: Quaternion (w, x, y, z), shape (4,).
    """
    if out[0] < 0.0:
        np.negative(out, out=out)


@obs_term("joint_pos", dim=_n_arm)
def joint_pos(ctx: "Context", out: np.ndarray) -> None:
    """Arm joint positions [rad], shape (n,)."""
    np.copyto(out, ctx.plant.q)


@obs_term("joint_pos_rel", dim=_n_arm)
def joint_pos_rel(ctx: "Context", out: np.ndarray) -> None:
    """Arm joint positions relative to q_home [rad], shape (n,)."""
    np.subtract(ctx.plant.q, ctx.handles.q_home, out=out)


@obs_term("joint_vel", dim=_n_arm)
def joint_vel(ctx: "Context", out: np.ndarray) -> None:
    """Arm joint velocities [rad/s], shape (n,)."""
    np.copyto(out, ctx.plant.qd)


@obs_term("joint_torque", dim=_n_arm)
def joint_torque(ctx: "Context", out: np.ndarray) -> None:
    """Last commanded arm torques [Nm] (before MuJoCo ctrl clamping), shape (n,)."""
    np.copyto(out, ctx.plant.ctrl_arm)


@obs_term("ee_pos", dim=3)
def ee_pos(ctx: "Context", out: np.ndarray) -> None:
    """TCP position [m], shape (3,)."""
    np.copyto(out, ctx.plant.ee_pos)


@obs_term("ee_quat", dim=4)
def ee_quat(ctx: "Context", out: np.ndarray) -> None:
    """TCP orientation quaternion (w >= 0), shape (4,)."""
    np.copyto(out, ctx.plant.ee_quat())
    _canonical_quat(out)


@obs_term("ee_pose", dim=7)
def ee_pose(ctx: "Context", out: np.ndarray) -> None:
    """TCP pose [pos (3) m, quat (4)], shape (7,)."""
    out[:3] = ctx.plant.ee_pos
    out[3:] = ctx.plant.ee_quat()
    _canonical_quat(out[3:])


@obs_term("ee_vel", dim=6)
def ee_vel(ctx: "Context", out: np.ndarray) -> None:
    """TCP twist [v (3) m/s, w (3) rad/s] in world frame, shape (6,)."""
    np.copyto(out, ctx.plant.ee_vel())


@obs_term("wrist_wrench", dim=6, needs_accumulation=True)
def wrist_wrench(ctx: "Context", out: np.ndarray) -> None:
    """Flange F/T [force (3) N, torque (3) Nm] in the ft_site frame, averaged over the policy step."""
    np.copyto(out, ctx.plant.wrist_wrench())


@obs_term("last_action", dim=_action_dim)
def last_action(ctx: "Context", out: np.ndarray) -> None:
    """Most recently applied action (after clipping and any smoothing), shape (action_dim,)."""
    np.copyto(out, ctx.applied_action)


@obs_term("ee_target_pose", dim=7)
def ee_target_pose(ctx: "Context", out: np.ndarray) -> None:
    """Current Cartesian controller setpoint [pos (3), quat (4)], shape (7,)."""
    c = ctx.controller
    out[:3] = c.pos_d
    out[3:] = c.quat_d
    _canonical_quat(out[3:])


@obs_term("target_pos", dim=3)
def target_pos(ctx: "Context", out: np.ndarray) -> None:
    """Task goal position `ctx.state['target_pos']` [m], shape (3,)."""
    np.copyto(out, ctx.buffer("target_pos", 3))


@obs_term("target_rel", dim=3)
def target_rel(ctx: "Context", out: np.ndarray) -> None:
    """Goal position minus TCP position [m], shape (3,)."""
    np.subtract(ctx.buffer("target_pos", 3), ctx.plant.ee_pos, out=out)
