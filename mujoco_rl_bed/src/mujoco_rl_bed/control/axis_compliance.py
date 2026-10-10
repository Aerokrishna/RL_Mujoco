"""Axis compliance controller: push along a motion axis with a set force, compliant everywhere else but yaw.

Admittance (outer loop) on top of a stiff Cartesian impedance (inner loop). The outer loop integrates the
dynamics of a virtual compliant frame (position x_c, orientation q_c) every physics tick.

The motion axis is a free 3D vector in the TCP frame (`axis_tcp`, default TCP z = gripper/peg axis). The
translational dynamics live in the *command frame* R_f = R_ref A: the reference orientation (the TCP
frame at reset) rotated so that its z axis is the motion axis. R_f is fixed for the episode (it changes
only when a new reference orientation is set), so the commanded side and push forces keep their meaning
when the tool tilts. In the command frame (z = motion axis):

    m a_t     = [ f_x + F_x - k_lat e_x - d_lat v_x,
                  f_y + F_y - k_lat e_y - d_lat v_y,
                  f_push + F_z         - d_axial v_z ]          force control along z (no spring)
    I alpha_t = [ T_x + k_tilt r_x - d_tilt w_x,
                  T_y + k_tilt r_y - d_tilt w_y,
                        k_yaw  r_z - d_yaw  w_z ]               soft roll/pitch, yaw held (no T_z)

with (F, T) the measured external wrench (force and torque about the TCP, from `set_wrench_source`,
low-pass filtered), e = R_f^T (x_c - p_ref) the offset from the reference position and v = R_f^T v_c.
The rotational line is written in the A-rotated virtual frame (q_c A, whose z is the motion axis on the
tool): r is the rotation from q_c to q_target = q_ref * exp(A [tilt_x, tilt_y, 0]), w the virtual angular
velocity and T the measured torque, all in that frame. Roll/pitch are about the command x, y axes; yaw
is the rotation about the motion axis. The inner loop tracks the virtual frame:

    tau = J^T [ Kp_in (x_c - p) + Kd_in (v_c - v) ; Kr_in (q_c ⊖ q) + Dr_in (w_c - w) ]
          + (I - J^T J#^T) tau_null + qfrc_bias

Why not a plain soft impedance: with tau = J^T F and soft gains, the arm's inertia couples axes (a pure
push along z also accelerates the TCP sideways), which moved the peg ~2 mm off the hole before
contact. The virtual frame has decoupled dynamics, and the stiff inner loop keeps the TCP on it.

In free space the TCP moves along the motion axis at about f_push / d_axial. In contact the virtual frame
balances the measured force: the peg presses with f_push along z and complies laterally and in tilt.
Without a wrench source, F = T = 0 (pure free-space admittance, no contact compliance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np

from mujoco_rl_bed.control.base import ControllerCfg
from mujoco_rl_bed.control.math_utils import DampedLeastSquares, GravityTorque, critical_damping

if TYPE_CHECKING:
    from mujoco_rl_bed.sim.plant import FrankaPlant


@dataclass
class AxisComplianceCfg(ControllerCfg):
    """Configuration for `AxisCompliance`.

    Attributes:
        m_lin: Virtual mass [kg].
        k_lat: Lateral (x/y) spring of the virtual frame back to the reference position [N/m].
        d_lat: Lateral damping [Ns/m].
        d_axial: Damping along z [Ns/m] (free-space speed = f_push / d_axial).
        i_rot: Virtual rotational inertia [kg m^2].
        k_tilt: Roll/pitch spring to the tilt target [Nm/rad].
        d_tilt: Roll/pitch damping [Nms/rad].
        k_yaw: Yaw spring [Nm/rad] (holds yaw).
        d_yaw: Yaw damping [Nms/rad]; None = critical, 2 sqrt(k_yaw i_rot).
        wrench_filter_hz: Low-pass cutoff on the measured wrench [Hz] (0 = no filter).
        kp_in: Inner translational stiffness [N/m].
        kr_in: Inner rotational stiffness [Nm/rad].
        kn: Null-space posture stiffness [Nm/rad].
        q_null: Null-space posture target [rad]; None = the joint configuration at each `reset`.
        compensate_coriolis: Add full `qfrc_bias` (else gravity only).
        use_nullspace: Add the projected posture torque.
        pinv_damping: Damping of the pseudo-inverse used by the null-space projector.
        max_lag: Max distance of the virtual frame from the TCP [m] (anti wind-up when blocked).
        saturate: Clip torques to the actuator limits.
        axis_tcp: Motion axis in the TCP frame (normalized; default TCP z).
    """

    m_lin: float = 1.0
    k_lat: float = 200.0
    d_lat: float = 40.0
    d_axial: float = 100.0
    i_rot: float = 0.02
    k_tilt: float = 3.0
    d_tilt: float = 1.5
    k_yaw: float = 30.0
    d_yaw: float | None = None
    wrench_filter_hz: float = 80.0
    kp_in: float = 3000.0
    kr_in: float = 100.0
    kn: float = 10.0
    q_null: tuple[float, ...] | None = None
    compensate_coriolis: bool = True
    use_nullspace: bool = True
    pinv_damping: float = 1e-3
    max_lag: float = 0.01
    saturate: bool = True
    axis_tcp: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def build(self, plant: "FrankaPlant") -> "AxisCompliance":
        """Instantiate the controller.

        Args:
            plant: Plant to control.

        Returns:
            An `AxisCompliance` controller.
        """
        return AxisCompliance(plant, self)


class AxisCompliance:
    """Admittance-type axis compliance on a stiff impedance inner loop (see module doc)."""

    @staticmethod
    def axis_frame(axis: np.ndarray) -> np.ndarray:
        """Rotation A whose z column is `axis` (minimal rotation from e_z; identity for e_z).

        Args:
            axis: Motion axis in the TCP frame, shape (3,).

        Returns:
            A, shape (3, 3): command-frame axes as columns, in TCP coordinates.
        """
        z = np.asarray(axis, dtype=np.float64)
        n = float(np.linalg.norm(z))
        if n < 1e-9:
            raise ValueError("axis_tcp must be non-zero")
        z = z / n
        ez = np.array([0.0, 0.0, 1.0])
        v = np.cross(ez, z)
        s, c = float(np.linalg.norm(v)), float(z[2])
        if s < 1e-12:
            return np.eye(3) if c > 0.0 else np.diag([1.0, -1.0, -1.0])
        K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + K + K @ K * ((1.0 - c) / s**2)

    setpoint_dim: int = 3 + 4 + 2 + 1 + 2  # x_c, q_c, f_lat, f_push, tilt

    def __init__(self, plant: "FrankaPlant", cfg: AxisComplianceCfg) -> None:
        """Allocate buffers.

        Args:
            plant: Plant to control.
            cfg: Controller configuration.
        """
        self.cfg = cfg
        n = plant.n
        self.n = n
        self.k_lin = np.array([cfg.k_lat, cfg.k_lat, 0.0])
        self.d_lin = np.array([cfg.d_lat, cfg.d_lat, cfg.d_axial])
        d_yaw = 2.0 * np.sqrt(cfg.k_yaw * cfg.i_rot) if cfg.d_yaw is None else cfg.d_yaw
        self.k_rot = np.array([cfg.k_tilt, cfg.k_tilt, cfg.k_yaw])
        self.d_rot = np.array([cfg.d_tilt, cfg.d_tilt, d_yaw])
        self.ext_rot_mask = np.array([1.0, 1.0, 0.0])           # external torque drives roll/pitch only
        self.A = self.axis_frame(cfg.axis_tcp)                  # command axes in the TCP frame (columns)
        self.R_f = np.eye(3)                                    # command frame -> world (fixed per episode)
        self.kp_in, self.kd_in = float(cfg.kp_in), float(critical_damping(cfg.kp_in))
        self.kr_in, self.dr_in = float(cfg.kr_in), float(critical_damping(cfg.kr_in))
        self.kn = np.full(n, float(cfg.kn))
        self.dn = critical_damping(self.kn)
        self._q_null_from_reset = cfg.q_null is None
        self.q_null = np.asarray(plant.h.q_home if cfg.q_null is None else cfg.q_null, dtype=np.float64).copy()
        self.use_nullspace = cfg.use_nullspace
        self.saturate = cfg.saturate
        self.dt = float(plant.dt)
        hz = float(cfg.wrench_filter_hz)
        self._w_alpha = 1.0 if hz <= 0.0 else self.dt / (self.dt + 1.0 / (2.0 * np.pi * hz))
        if cfg.compensate_coriolis:
            self._gravity = None
        else:
            jids = [plant.h.joint_ids[name] for name in plant.h.arm_joint_names]
            self._gravity = GravityTorque(plant.model, plant.data, jids)

        # Reference, commands and the virtual frame.
        self.p_ref = np.zeros(3)
        self.q_ref = np.array([1.0, 0.0, 0.0, 0.0])
        self.q_target = np.array([1.0, 0.0, 0.0, 0.0])
        self.f_lat = np.zeros(2)
        self.f_push = 0.0
        self.tilt = np.zeros(2)
        self.x_c = np.zeros(3)
        self.v_c = np.zeros(3)          # world frame
        self.q_c = np.array([1.0, 0.0, 0.0, 0.0])
        self.w_c = np.zeros(3)          # virtual-frame (local) angular velocity
        self.wrench = np.zeros(6)       # filtered external wrench, world frame [F, T about the TCP]
        self._wrench_fn: Callable[[np.ndarray], None] | None = None
        self._w_raw = np.zeros(6)

        self._dls = DampedLeastSquares(6, n, cfg.pinv_damping)
        self._Rc = np.zeros(9)
        self._Rc3 = self._Rc.reshape(3, 3)
        self._v = np.zeros(6)
        self._e = np.zeros(3)
        self._vt = np.zeros(3)
        self._ft = np.zeros(3)
        self._acc = np.zeros(3)
        self._r = np.zeros(3)
        self._tt = np.zeros(3)
        self._alpha = np.zeros(3)
        self._dq = np.zeros(4)
        self._qtmp = np.zeros(4)
        self._err_r = np.zeros(3)
        self._F = np.zeros(6)
        self._tau = np.zeros(n)
        self._tau_null = np.zeros(n)
        self._tau_proj = np.zeros(n)
        self._tmp_n = np.zeros(n)
        self._tmp3 = np.zeros(3)
        self._tilt_q = np.zeros(4)
        self._Rr = np.zeros(9)
        self._wa = np.zeros(3)
        self._lim = plant.torque_limits.copy()
        self._neg_lim = -self._lim

    # ------------------------------------------------------------------ interface
    def set_wrench_source(self, fn: Callable[[np.ndarray], None] | None) -> None:
        """Register the external-wrench measurement.

        Args:
            fn: `fn(out)` writes the wrench the environment applies to the robot/tool, world frame,
                [force (3) N, torque about the TCP (3) Nm], into `out` (shape (6,)); None = no measurement.
        """
        self._wrench_fn = fn

    def set_target(self, pos: np.ndarray | None = None, quat: np.ndarray | None = None,
                   f_lat: np.ndarray | None = None, f_push: float | None = None,
                   tilt: np.ndarray | None = None) -> None:
        """Set the reference pose and/or the commands (once per policy step).

        Args:
            pos: Reference position for the lateral spring, world frame [m], shape (3,).
            quat: Reference TCP orientation (w, x, y, z), shape (4,); also fixes the command frame
                R_f = R(quat) A (tilt is applied on top of it).
            f_lat: Commanded side force along the command frame's x, y [N], shape (2,).
            f_push: Commanded force along the motion axis [N] (positive = along `axis_tcp`).
            tilt: Tilt target about the command frame's x, y axes [rad], shape (2,).
        """
        if pos is not None:
            np.copyto(self.p_ref, pos)
        if quat is not None:
            np.copyto(self.q_ref, quat)
            mujoco.mju_quat2Mat(self._Rr, self.q_ref)
            np.dot(self._Rr.reshape(3, 3), self.A, out=self.R_f)
        if f_lat is not None:
            np.copyto(self.f_lat, f_lat)
        if f_push is not None:
            self.f_push = float(f_push)
        if tilt is not None:
            np.copyto(self.tilt, tilt)
        if quat is not None or tilt is not None:
            self._tmp3[:] = self.A[:, 0] * self.tilt[0] + self.A[:, 1] * self.tilt[1]   # TCP-frame axis
            angle = float(np.linalg.norm(self._tmp3))
            if angle > 1e-12:
                mujoco.mju_axisAngle2Quat(self._tilt_q, self._tmp3 / angle, angle)
                mujoco.mju_mulQuat(self.q_target, self.q_ref, self._tilt_q)
            else:
                np.copyto(self.q_target, self.q_ref)

    def reset(self, plant: "FrankaPlant") -> None:
        """Start the virtual frame at the current pose (at rest), make it the reference, zero the commands.

        Args:
            plant: Plant after reset + `mj_forward`.
        """
        if self._q_null_from_reset:
            np.copyto(self.q_null, plant.q)
        self.f_lat.fill(0.0)
        self.f_push = 0.0
        self.tilt.fill(0.0)
        np.copyto(self.x_c, plant.ee_pos)
        self.v_c.fill(0.0)
        np.copyto(self.q_c, plant.ee_quat())
        self.w_c.fill(0.0)
        self.wrench.fill(0.0)
        self.set_target(pos=plant.ee_pos, quat=plant.ee_quat())
        self._tau.fill(0.0)

    def setpoint(self, out: np.ndarray) -> np.ndarray:
        """Write [x_c (3), q_c (4), f_lat (2), f_push, tilt (2)] into `out`.

        Args:
            out: Buffer of shape (12,).

        Returns:
            `out`.
        """
        out[0:3] = self.x_c
        out[3:7] = self.q_c
        out[7:9] = self.f_lat
        out[9] = self.f_push
        out[10:12] = self.tilt
        return out

    @property
    def last_torque(self) -> np.ndarray:
        """Most recent commanded torque [Nm], shape (n,) (owned buffer)."""
        return self._tau

    # ------------------------------------------------------------------ hot loop
    def _admittance_step(self) -> None:
        """Advance the virtual frame by one physics step (semi-implicit Euler)."""
        dt = self.dt
        if self._wrench_fn is not None:
            self._wrench_fn(self._w_raw)
            self.wrench += self._w_alpha * (self._w_raw - self.wrench)
        Rf = self.R_f
        # translation, in the (episode-fixed) command frame
        np.subtract(self.x_c, self.p_ref, out=self._tmp3)
        np.dot(Rf.T, self._tmp3, out=self._e)
        np.dot(Rf.T, self.v_c, out=self._vt)
        np.dot(Rf.T, self.wrench[:3], out=self._ft)
        a = self._acc
        a[0] = self.f_lat[0] + self._ft[0] - self.k_lin[0] * self._e[0] - self.d_lin[0] * self._vt[0]
        a[1] = self.f_lat[1] + self._ft[1] - self.k_lin[1] * self._e[1] - self.d_lin[1] * self._vt[1]
        a[2] = self.f_push + self._ft[2] - self.d_lin[2] * self._vt[2]
        a /= self.cfg.m_lin
        np.dot(Rf, a, out=self._tmp3)
        self.v_c += dt * self._tmp3
        self.x_c += dt * self.v_c
        # rotation, in the A-rotated virtual frame (z = motion axis on the tool)
        mujoco.mju_quat2Mat(self._Rc, self.q_c)
        A = self.A
        mujoco.mju_subQuat(self._tmp3, self.q_target, self.q_c)
        np.dot(A.T, self._tmp3, out=self._r)
        np.dot(self._Rc3.T, self.wrench[3:], out=self._tmp3)
        np.dot(A.T, self._tmp3, out=self._tt)
        np.dot(A.T, self.w_c, out=self._wa)
        al = self._alpha
        np.multiply(self.k_rot, self._r, out=al)
        al -= self.d_rot * self._wa
        al += self.ext_rot_mask * self._tt
        al /= self.cfg.i_rot
        np.dot(A, al, out=self._tmp3)
        self.w_c += dt * self._tmp3
        angle = float(np.linalg.norm(self.w_c)) * dt
        if angle > 1e-12:
            mujoco.mju_axisAngle2Quat(self._dq, self.w_c / np.linalg.norm(self.w_c), angle)
            mujoco.mju_mulQuat(self._qtmp, self.q_c, self._dq)
            mujoco.mju_normalize4(self._qtmp)
            np.copyto(self.q_c, self._qtmp)

    def torque(self, plant: "FrankaPlant") -> np.ndarray:
        """Advance the virtual frame and compute the inner impedance torques.

        Args:
            plant: Plant after `mj_step1`.

        Returns:
            Owned buffer of joint torques [Nm], shape (n,).
        """
        self._admittance_step()
        J = plant.jacobian()
        q, qd = plant.q, plant.qd
        tau = self._tau
        np.dot(J, qd, out=self._v)
        # anti wind-up: keep the virtual frame within max_lag of the TCP (e.g. when blocked by contact)
        np.subtract(self.x_c, plant.ee_pos, out=self._tmp3)
        lag = float(np.linalg.norm(self._tmp3))
        if lag > self.cfg.max_lag:
            self._tmp3 *= self.cfg.max_lag / lag
            np.add(plant.ee_pos, self._tmp3, out=self.x_c)
        F = self._F
        F[:3] = self.kp_in * self._tmp3 + self.kd_in * (self.v_c - self._v[:3])
        mujoco.mju_subQuat(self._err_r, self.q_c, plant.ee_quat())          # TCP-frame error
        R = plant.ee_rot.reshape(3, 3)
        np.dot(R, self._err_r, out=self._tmp3)                             # -> world
        mujoco.mju_quat2Mat(self._Rc, self.q_c)
        F[3:] = self.kr_in * self._tmp3 + self.dr_in * (self._Rc3 @ self.w_c - self._v[3:])
        np.dot(J.T, F, out=tau)

        if self.use_nullspace:
            tn = self._tau_null
            np.subtract(self.q_null, q, out=tn)
            np.multiply(self.kn, tn, out=tn)
            np.multiply(self.dn, qd, out=self._tmp_n)
            np.subtract(tn, self._tmp_n, out=tn)
            self._dls.factor(J)
            self._dls.project_nullspace(J, tn, self._tau_proj)
            np.add(tau, self._tau_proj, out=tau)
        if self._gravity is None:
            np.add(tau, plant.qfrc_bias, out=tau)
        else:
            np.add(tau, self._gravity.compute(), out=tau)
        if self.saturate:
            np.maximum(tau, self._neg_lim, out=tau)
            np.minimum(tau, self._lim, out=tau)
        return tau
