"""Damped-least-squares inverse kinematics for the TCP (reset-time only, not in the hot loop).

Iterates q <- q + J# e + (I - J# J) k_null (q_null - q). Each iteration uses only
`mj_kinematics` + `mj_comPos` (no collision detection), clamps the step size, and
clips joints to their ranges. On return, `data.qpos` holds the solution; the caller
must run `mj_forward` before using derived quantities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_rl_bed.control.math_utils import QuatBuffers, quat_error_world

if TYPE_CHECKING:
    from mujoco_rl_bed.sim.plant import FrankaPlant


def solve_tcp_ik(plant: "FrankaPlant", pos: np.ndarray, quat: np.ndarray, q_init: np.ndarray | None = None,
                 q_null: np.ndarray | None = None, max_iters: int = 200, tol: float = 1e-5,
                 damping: float = 1e-3, max_step: float = 0.2, null_gain: float = 0.05,
                 rot_weight: float = 1.0) -> tuple[np.ndarray, float]:
    """Solve for arm joints placing the TCP at (pos, quat).

    Args:
        plant: Plant whose `data.qpos` (arm slice) is overwritten.
        pos: Target TCP position, world frame [m], shape (3,).
        quat: Target TCP orientation (w, x, y, z), shape (4,).
        q_init: Initial guess [rad], shape (n,); None = current arm configuration.
        q_null: Null-space posture target [rad]; None = `handles.q_home`.
        max_iters: Iteration limit.
        tol: Convergence threshold on the weighted 6D error norm [m / rad].
        damping: DLS damping lambda.
        max_step: Max joint step norm per iteration [rad].
        null_gain: Gain of the null-space posture pull.
        rot_weight: Weight of the orientation error relative to position.

    Returns:
        (q, err): solution [rad], shape (n,), and the final 6D error norm.
    """
    m, d, h = plant.model, plant.data, plant.h
    jids = [h.joint_ids[n] for n in h.arm_joint_names]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    q = (plant.q if q_init is None else np.asarray(q_init, dtype=np.float64)).copy()
    qn = h.q_home if q_null is None else np.asarray(q_null, dtype=np.float64)
    qb = QuatBuffers()
    e = np.zeros(6)
    eye6 = np.eye(6)
    eye_n = np.eye(plant.n)
    err = np.inf
    for _ in range(max_iters):
        plant.q[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)  # Jacobians need subtree COMs
        e[:3] = pos - plant.ee_pos
        quat_error_world(e[3:], quat, plant.ee_quat(), qb)
        e[3:] *= rot_weight
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        J = plant.jacobian().copy()
        J[3:] *= rot_weight
        A = J @ J.T + (damping ** 2) * eye6
        Jpinv = J.T @ np.linalg.solve(A, eye6)                     # damped pseudo-inverse (n x 6)
        dq = Jpinv @ e + (eye_n - Jpinv @ J) @ (null_gain * (qn - q))
        nrm = np.linalg.norm(dq)
        if nrm > max_step:
            dq *= max_step / nrm
        q = np.clip(q + dq, lo, hi)
    plant.q[:] = q
    return q, err
