"""Reusable reward terms: `fn(ctx, **params) -> float`, evaluated once per policy step.

Signs are left to the weights: distance and penalty terms return non-negative values,
so use negative weights for them in the task config.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from mujoco_rl_bed.env.managers.reward import reward_term

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context


def _target_dist(ctx: "Context") -> float:
    """Euclidean TCP-to-target distance [m].

    Args:
        ctx: Context with `state['target_pos']`.

    Returns:
        Distance [m].
    """
    t = ctx.buffer("target_pos", 3)
    p = ctx.plant.ee_pos
    return math.sqrt((t[0] - p[0]) ** 2 + (t[1] - p[1]) ** 2 + (t[2] - p[2]) ** 2)


@reward_term("ee_target_dist")
def ee_target_dist(ctx: "Context") -> float:
    """TCP-to-target distance [m] (use a negative weight)."""
    return _target_dist(ctx)


@reward_term("ee_target_tanh")
def ee_target_tanh(ctx: "Context", std: float = 0.1) -> float:
    """Bounded closeness kernel 1 - tanh(d / std), in [0, 1].

    Args:
        ctx: Context.
        std: Distance scale [m].
    """
    return 1.0 - math.tanh(_target_dist(ctx) / std)


@reward_term("ee_target_success")
def ee_target_success(ctx: "Context", threshold: float = 0.01) -> float:
    """1 if the TCP is within `threshold` [m] of the target, else 0."""
    return 1.0 if _target_dist(ctx) < threshold else 0.0


@reward_term("action_rate_l2")
def action_rate_l2(ctx: "Context") -> float:
    """Squared change of the raw (clipped) policy action between policy steps (use a negative weight).

    Penalizes step-to-step chatter in what the policy outputs (0 on the first step of an episode).
    """
    d = ctx.action - ctx.prev_action
    return float(np.dot(d, d))


@reward_term("action_l2")
def action_l2(ctx: "Context") -> float:
    """Squared action norm (use a negative weight)."""
    return float(np.dot(ctx.action, ctx.action))


@reward_term("joint_vel_l2")
def joint_vel_l2(ctx: "Context") -> float:
    """Squared joint velocity norm [rad^2/s^2] (use a negative weight)."""
    qd = ctx.plant.qd
    return float(np.dot(qd, qd))


@reward_term("torque_l2")
def torque_l2(ctx: "Context") -> float:
    """Squared commanded torque norm [Nm^2] (use a negative weight)."""
    tau = ctx.plant.ctrl_arm
    return float(np.dot(tau, tau))


@reward_term("ee_speed")
def ee_speed(ctx: "Context") -> float:
    """TCP linear speed [m/s] (use a negative weight)."""
    v = ctx.plant.ee_vel()
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
