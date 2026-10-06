"""Reusable event terms: `fn(ctx, **params) -> None`.

Reset events run after `plant.reset()` and before `mj_forward`/controller reset, so
they can set joint positions, object poses, targets and model parameters freely.
All randomness comes from `ctx.rng`. Model-parameter events cache the nominal values
in `ctx.state` on first use, so randomization never compounds across resets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mujoco_rl_bed.env.managers.events import event_term

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context


@event_term("reset_joints_by_offset")
def reset_joints_by_offset(ctx: "Context", offset: float = 0.1) -> None:
    """Set arm joints to q_home + U(-offset, offset), clipped to the joint ranges.

    Args:
        ctx: Context.
        offset: Half-width of the uniform offset [rad].
    """
    h = ctx.handles
    key = "_arm_jnt_range"
    rng_lim = ctx.state.get(key)
    if rng_lim is None:
        jids = [h.joint_ids[n] for n in h.arm_joint_names]
        rng_lim = ctx.state[key] = ctx.model.jnt_range[jids].copy()
    q = h.q_home + ctx.rng.uniform(-offset, offset, size=h.n_arm)
    ctx.plant.q[:] = np.clip(q, rng_lim[:, 0], rng_lim[:, 1])


@event_term("sample_target_pos")
def sample_target_pos(ctx: "Context", lo: tuple[float, float, float] = (0.35, -0.25, 0.15),
                      hi: tuple[float, float, float] = (0.65, 0.25, 0.5), marker: str = "target") -> None:
    """Sample `state['target_pos']` uniformly in a box and move the mocap marker (if present).

    Args:
        ctx: Context.
        lo: Box lower corner, world frame [m].
        hi: Box upper corner, world frame [m].
        marker: Name of a mocap body to move to the target ("" = none).
    """
    t = ctx.buffer("target_pos", 3)
    t[:] = ctx.rng.uniform(lo, hi)
    if marker and marker in ctx.handles.body_ids:
        mid = ctx.model.body_mocapid[ctx.handles.body_ids[marker]]
        if mid >= 0:
            ctx.data.mocap_pos[mid] = t


@event_term("randomize_friction")
def randomize_friction(ctx: "Context", geoms: tuple[str, ...] = (), scale_range: tuple[float, float] = (0.8, 1.2)) -> None:
    """Scale sliding friction of the given geoms (all geoms if empty) by U(scale_range).

    Args:
        ctx: Context.
        geoms: Geom names.
        scale_range: Multiplicative range on the nominal friction.
    """
    key = f"_friction_nominal::{','.join(geoms)}"
    ids = ctx.state.get(key + "_ids")
    if ids is None:
        ids = np.array([ctx.handles.geom_ids[g] for g in geoms], dtype=np.int64) if geoms \
            else np.arange(ctx.model.ngeom)
        ctx.state[key + "_ids"] = ids
        ctx.state[key] = ctx.model.geom_friction[ids, 0].copy()
    ctx.model.geom_friction[ids, 0] = ctx.state[key] * ctx.rng.uniform(*scale_range, size=ids.size)


@event_term("randomize_joint_damping")
def randomize_joint_damping(ctx: "Context", scale_range: tuple[float, float] = (0.5, 1.5)) -> None:
    """Scale arm joint damping by U(scale_range) per joint.

    Args:
        ctx: Context.
        scale_range: Multiplicative range on the nominal damping.
    """
    sl = ctx.handles.arm_dof
    nominal = ctx.state.get("_damping_nominal")
    if nominal is None:
        nominal = ctx.state["_damping_nominal"] = ctx.model.dof_damping[sl].copy()
    ctx.model.dof_damping[sl] = nominal * ctx.rng.uniform(*scale_range, size=nominal.size)


@event_term("randomize_controller_gains")
def randomize_controller_gains(ctx: "Context", scale_range: tuple[float, float] = (0.8, 1.2)) -> None:
    """Scale the controller's nominal Kp by U(scale_range) per axis (Kd follows if automatic).

    Runs before the controller reset, which then applies the new nominal gains.

    Args:
        ctx: Context.
        scale_range: Multiplicative range on the configured gains.
    """
    c = ctx.controller
    base = ctx.state.get("_kp_base")
    if base is None:
        base = ctx.state["_kp_base"] = c.kp_nominal.copy()
        ctx.state["_kd_base"] = c.kd_nominal.copy()
    s = ctx.rng.uniform(*scale_range, size=base.size)
    c.kp_nominal[:] = base * s
    if getattr(c, "auto_kd", False):
        c.kd_nominal[:] = 2.0 * np.sqrt(c.kp_nominal)
    else:
        c.kd_nominal[:] = ctx.state["_kd_base"]


@event_term("joint_torque_noise")
def joint_torque_noise(ctx: "Context", std: float = 0.5) -> None:
    """Gaussian joint torque disturbance [Nm] held for one interval (written to `qfrc_applied`).

    Use as an `interval` event. `mj_resetData` clears it on reset.

    Args:
        ctx: Context.
        std: Standard deviation [Nm].
    """
    ctx.data.qfrc_applied[ctx.handles.arm_dof] = ctx.rng.normal(0.0, std, size=ctx.handles.n_arm)
