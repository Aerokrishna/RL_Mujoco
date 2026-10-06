"""Reusable termination terms: `fn(ctx, **params) -> bool`, evaluated once per policy step.

Whether a term means termination or truncation (and success) is set in its
`TerminationTermCfg`, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mujoco_rl_bed.env.managers.termination import termination_term
from mujoco_rl_bed.terms.rewards import _target_dist

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context


@termination_term("unstable")
def unstable(ctx: "Context", max_joint_vel: float = 20.0) -> bool:
    """Simulation blow-up guard: non-finite state or excessive arm joint velocity.

    Args:
        ctx: Context.
        max_joint_vel: Velocity bound [rad/s].
    """
    qd = ctx.plant.qd
    return (not np.isfinite(ctx.data.qpos).all()) or float(np.abs(qd).max()) > max_joint_vel


@termination_term("ee_target_reached")
def ee_target_reached(ctx: "Context", threshold: float = 0.01) -> bool:
    """TCP within `threshold` [m] of `state['target_pos']`."""
    return _target_dist(ctx) < threshold


@termination_term("ee_out_of_bounds")
def ee_out_of_bounds(ctx: "Context", lo: tuple[float, float, float] = (0.0, -0.8, -0.05),
                     hi: tuple[float, float, float] = (1.0, 0.8, 1.2)) -> bool:
    """TCP outside an axis-aligned box [m] (world frame)."""
    p = ctx.plant.ee_pos
    return bool(p[0] < lo[0] or p[1] < lo[1] or p[2] < lo[2] or p[0] > hi[0] or p[1] > hi[1] or p[2] > hi[2])
