"""Allocation-free math helpers for the controllers (float64).

- `DampedLeastSquares`: Cholesky factorization of (J J^T + lambda^2 I) with in-place
  solves. It lets the controllers apply the damped pseudo-inverse J# = J^T (J J^T + lambda^2 I)^-1
  without ever forming it or calling `np.linalg`, which allocates.
- `damped_pinv`: an allocating reference implementation (tests, offline use).
- `quat_error_world`: orientation error x_d ⊖ x as a world-frame rotation vector, via `mju_subQuat`.
- `apply_rotvec`: rotate a quaternion by a world-frame rotation vector.
- `GravityTorque`: gravity-only joint torques for a serial chain (used when Coriolis
  compensation is disabled).
- `critical_damping`: kd = 2 sqrt(kp).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def critical_damping(kp: np.ndarray | float, out: np.ndarray | None = None) -> np.ndarray:
    """Critical damping for a unit-mass second-order system: kd = 2 sqrt(kp).

    Args:
        kp: Stiffness, scalar or array of shape (k,).
        out: Optional output buffer of shape (k,) (avoids allocation).

    Returns:
        Damping gains with the same shape as `kp`.
    """
    if out is None:
        return 2.0 * np.sqrt(np.asarray(kp, dtype=np.float64))
    np.sqrt(kp, out=out)
    out *= 2.0
    return out


def damped_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    """Damped pseudo-inverse J# = J^T (J J^T + lam^2 I)^-1 (allocating reference version).

    Args:
        J: Jacobian, shape (m, n).
        lam: Damping factor lambda (>= 0).

    Returns:
        J#, shape (n, m).
    """
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (lam * lam) * np.eye(m))


class DampedLeastSquares:
    """In-place damped least squares on a fixed-size Jacobian.

    `factor(J)` computes the Cholesky factor of A = J J^T + lambda^2 I (m x m) in a
    preallocated buffer. `solve(out, b)` then returns A^-1 b. With these:

        J#^T v = A^-1 J v          (m-vector)
        J^T J#^T v                 (used by the null-space projector N^T = I - J^T J#^T)
    """

    def __init__(self, m: int, n: int, lam: float) -> None:
        """Allocate buffers.

        Args:
            m: Task-space dimension (6).
            n: Joint-space dimension (7).
            lam: Damping factor lambda.
        """
        self.m, self.n = m, n
        self.lam2 = float(lam) ** 2
        self._A = np.zeros((m, m))
        self._diag = self._A.reshape(-1)[:: m + 1]  # strided writable view on the diagonal
        self._tmp_m = np.zeros(m)
        self._tmp_m2 = np.zeros(m)

    def factor(self, J: np.ndarray) -> None:
        """Factor A = J J^T + lambda^2 I in place.

        Args:
            J: Jacobian, shape (m, n).
        """
        np.dot(J, J.T, out=self._A)
        self._diag += self.lam2
        mujoco.mju_cholFactor(self._A, 0.0)  # lower-triangular Cholesky, in place

    def solve(self, out: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve A x = b using the stored factor.

        Args:
            out: Output buffer, shape (m,).
            b: Right-hand side, shape (m,).

        Returns:
            `out`.
        """
        mujoco.mju_cholSolve(out, self._A, b)
        return out

    def project_nullspace(self, J: np.ndarray, tau0: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Null-space projection out = (I - J^T J#^T) tau0, using the stored factor of J.

        Args:
            J: Same Jacobian passed to `factor`, shape (m, n).
            tau0: Joint torque to project [Nm], shape (n,).
            out: Output buffer, shape (n,). May not alias `tau0`.

        Returns:
            `out`.
        """
        np.dot(J, tau0, out=self._tmp_m)                     # J tau0
        mujoco.mju_cholSolve(self._tmp_m2, self._A, self._tmp_m)  # J#^T tau0 = A^-1 J tau0
        np.dot(J.T, self._tmp_m2, out=out)                   # J^T J#^T tau0
        np.subtract(tau0, out, out=out)
        return out


class QuatBuffers:
    """Scratch buffers for the quaternion helpers (one instance per owner)."""

    def __init__(self) -> None:
        """Allocate scratch space."""
        self.v3 = np.zeros(3)
        self.q4 = np.zeros(4)
        self.axis = np.zeros(3)


def quat_error_world(out: np.ndarray, q_des: np.ndarray, q: np.ndarray, buf: QuatBuffers) -> np.ndarray:
    """Orientation error x_d ⊖ x as a world-frame rotation vector [rad].

    `mju_subQuat(res, qa, qb)` returns res such that qb * quat(res) = qa, with res in
    qb's local frame and the shortest rotation chosen. We rotate it into the world frame
    with q so it can be multiplied with the world-frame Jacobian.

    Args:
        out: Output buffer, shape (3,).
        q_des: Desired orientation (w, x, y, z), shape (4,).
        q: Current orientation (w, x, y, z), shape (4,).
        buf: Scratch buffers.

    Returns:
        `out`.
    """
    mujoco.mju_subQuat(buf.v3, q_des, q)    # local-frame error
    mujoco.mju_rotVecQuat(out, buf.v3, q)   # -> world frame
    return out


def apply_rotvec(out: np.ndarray, q: np.ndarray, rotvec: np.ndarray, buf: QuatBuffers) -> np.ndarray:
    """Rotate quaternion q by a world-frame rotation vector: out = quat(rotvec) * q.

    Args:
        out: Output quaternion buffer, shape (4,). May alias `q`.
        q: Input orientation (w, x, y, z), shape (4,).
        rotvec: World-frame axis-angle [rad], shape (3,).
        buf: Scratch buffers.

    Returns:
        `out` (normalized).
    """
    angle = math.sqrt(rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2])
    if angle < 1e-12:
        if out is not q:
            out[:] = q
        return out
    np.divide(rotvec, angle, out=buf.axis)
    mujoco.mju_axisAngle2Quat(buf.q4, buf.axis, angle)
    mujoco.mju_mulQuat(out, buf.q4, q)  # left-multiply = rotation in the world frame
    mujoco.mju_normalize4(out)
    return out


class GravityTorque:
    """Gravity-only joint torques for the arm hinge joints (no Coriolis terms).

    For hinge joint i carried by body b_i, the generalized gravity force is
        g_i = - a_i · ((c_i - p_i) x (M_i g))
    where a_i is the joint axis, p_i the joint anchor, c_i and M_i the subtree COM and
    mass of b_i, and g the gravity vector. The sign matches `qfrc_bias`, i.e. the
    torque needed to hold the arm against gravity. Uses `subtree_com`, `xanchor` and `xaxis`, which
    are valid after `mj_step1`. Allocation-free.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, joint_ids: list[int]) -> None:
        """Cache views and buffers.

        Args:
            model: Compiled model.
            data: Simulation data.
            joint_ids: Arm hinge joint ids, in arm order.
        """
        self.m, self.d = model, data
        self.jids = np.asarray(joint_ids, dtype=np.int64)
        self.bids = np.asarray([model.jnt_bodyid[j] for j in joint_ids], dtype=np.int64)
        n = len(joint_ids)
        self._r = np.zeros((n, 3))
        self._F = np.zeros((n, 3))
        self._c = np.zeros((n, 3))
        self._tmp = np.zeros(n)
        self._out = np.zeros(n)
        self._axis = np.zeros((n, 3))
        self._mass = np.zeros((n, 1))

    def compute(self) -> np.ndarray:
        """Compute gravity torques for the current configuration.

        Returns:
            Owned buffer, shape (n,) [Nm].
        """
        m, d = self.m, self.d
        # np.take with out= gathers rows without allocating a new array.
        np.take(d.subtree_com, self.bids, axis=0, out=self._c)
        np.take(d.xanchor, self.jids, axis=0, out=self._r)
        np.subtract(self._c, self._r, out=self._r)                     # lever arm r = c - p
        np.take(m.body_subtreemass, self.bids, out=self._mass[:, 0])
        np.multiply(self._mass, m.opt.gravity, out=self._F)            # F = M g
        r, F, c = self._r, self._F, self._c
        # c = r x F, written component-wise into preallocated columns.
        t = self._tmp
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            np.multiply(r[:, j], F[:, k], out=c[:, i])
            np.multiply(r[:, k], F[:, j], out=t)
            np.subtract(c[:, i], t, out=c[:, i])
        np.take(d.xaxis, self.jids, axis=0, out=self._axis)
        np.multiply(self._axis, c, out=c)
        np.sum(c, axis=1, out=self._out)
        np.negative(self._out, out=self._out)
        return self._out
