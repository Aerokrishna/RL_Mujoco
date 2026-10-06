"""Cartesian impedance controller for the torque-controlled arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mujoco_rl_bed.control.base import ControllerCfg
from mujoco_rl_bed.control.math_utils import (DampedLeastSquares, GravityTorque, QuatBuffers, critical_damping,
                                          quat_error_world)

if TYPE_CHECKING:
    from mujoco_rl_bed.sim.plant import FrankaPlant


@dataclass
class CartesianImpedanceCfg(ControllerCfg):
    """Configuration for `CartesianImpedance`.

    Attributes:
        kp: Task-space stiffness (x, y, z [N/m], rx, ry, rz [Nm/rad]), world frame.
        kd: Task-space damping (6,); None = critical damping 2 sqrt(kp).
        kn: Null-space posture stiffness [Nm/rad], scalar or (n,).
        dn: Null-space posture damping [Nms/rad], scalar or (n,); None = 2 sqrt(kn).
        q_null: Null-space posture target [rad]; None = scene `q_home`.
        compensate_coriolis: True = add full `qfrc_bias` (gravity + Coriolis/centrifugal);
            False = gravity only.
        use_nullspace: Add the projected posture torque.
        pinv_damping: Damping lambda of the pseudo-inverse used in the null-space projector. The
            projection leaks posture torque into task space roughly in proportion to lambda^2:
            1e-2 gave a 0.1-0.2 mm static TCP error (too much for 0.25 mm insertion clearance);
            1e-3 gives about 0.001 mm.
        torque_rate_limit: Max |tau_t - tau_{t-1}| per physics tick [Nm]; None = off.
        saturate: Clip torques to the actuator limits.
    """

    kp: tuple[float, ...] = (600.0, 600.0, 600.0, 50.0, 50.0, 50.0)
    kd: tuple[float, ...] | None = None
    kn: float | tuple[float, ...] = 10.0
    dn: float | tuple[float, ...] | None = None
    q_null: tuple[float, ...] | None = None
    compensate_coriolis: bool = True
    use_nullspace: bool = True
    pinv_damping: float = 1e-3
    torque_rate_limit: float | None = None
    saturate: bool = True

    def build(self, plant: "FrankaPlant") -> "CartesianImpedance":
        """Instantiate the controller.

        Args:
            plant: Plant to control.

        Returns:
            A `CartesianImpedance` controller.
        """
        return CartesianImpedance(plant, self)


class CartesianImpedance:
    """Task-space impedance with null-space posture control.

    Control law (all quantities in the world frame, evaluated each physics tick):

        tau = J^T [ Kp (x_d ⊖ x) - Kd (J qd) ] + (I - J^T J#^T) tau_null + qfrc_bias
        tau_null = kn (q_null - q) - dn qd

    where x_d ⊖ x stacks the position error (p_d - p) and the orientation error
    (from `mju_subQuat`, rotated to the world frame), J is the 6xn TCP Jacobian, J# is
    the damped pseudo-inverse J^T (J J^T + lambda^2 I)^-1, and `qfrc_bias` is gravity
    plus Coriolis (gravity only if `compensate_coriolis=False`). The result is optionally
    rate-limited per tick and saturated to the actuator torque limits.

    Gains can be changed every policy step through `set_target(kp=..., kd=...)`, which
    is what variable-impedance actions use. If `kd` is omitted while `kp` changes, and
    the config uses automatic damping (`kd=None`), Kd is reset to 2 sqrt(Kp).
    """

    setpoint_dim: int = 7 + 6 + 6  # pos(3) quat(4) kp(6) kd(6)

    def __init__(self, plant: "FrankaPlant", cfg: CartesianImpedanceCfg) -> None:
        """Allocate all buffers.

        Args:
            plant: Plant to control.
            cfg: Controller configuration.
        """
        self.cfg = cfg
        n = plant.n
        self.n = n
        self.kp = np.asarray(cfg.kp, dtype=np.float64).copy()
        self.auto_kd = cfg.kd is None
        self.kd = critical_damping(self.kp) if self.auto_kd else np.asarray(cfg.kd, dtype=np.float64).copy()
        self.kp_nominal = self.kp.copy()  # for gain randomization / resets
        self.kd_nominal = self.kd.copy()
        self.kn = np.broadcast_to(np.asarray(cfg.kn, dtype=np.float64), (n,)).copy()
        self.dn = critical_damping(self.kn) if cfg.dn is None else \
            np.broadcast_to(np.asarray(cfg.dn, dtype=np.float64), (n,)).copy()
        self.q_null = np.asarray(plant.h.q_home if cfg.q_null is None else cfg.q_null, dtype=np.float64).copy()
        self.use_nullspace = cfg.use_nullspace
        self.saturate = cfg.saturate
        self.rate = cfg.torque_rate_limit

        # Target.
        self.pos_d = np.zeros(3)
        self.quat_d = np.array([1.0, 0.0, 0.0, 0.0])

        # Bias source.
        if cfg.compensate_coriolis:
            self._gravity = None
        else:
            jids = [plant.h.joint_ids[name] for name in plant.h.arm_joint_names]
            self._gravity = GravityTorque(plant.model, plant.data, jids)

        # Buffers.
        self._dls = DampedLeastSquares(6, n, cfg.pinv_damping)
        self._qb = QuatBuffers()
        self._err = np.zeros(6)
        self._err_p = self._err[:3]
        self._err_r = self._err[3:]
        self._v = np.zeros(6)
        self._F = np.zeros(6)
        self._tau = np.zeros(n)
        self._tau_null = np.zeros(n)
        self._tau_proj = np.zeros(n)
        self._tmp_n = np.zeros(n)
        self._tau_prev = np.zeros(n)
        self._lo = np.zeros(n)
        self._hi = np.zeros(n)
        self._lim = plant.torque_limits.copy()
        self._neg_lim = -self._lim

    # ------------------------------------------------------------------ interface
    def set_target(self, pos: np.ndarray | None = None, quat: np.ndarray | None = None,
                   kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        """Set the pose target and/or gains (copied into owned buffers).

        Args:
            pos: Target TCP position, world frame [m], shape (3,).
            quat: Target TCP orientation (w, x, y, z), shape (4,).
            kp: Stiffness (6,) [N/m, Nm/rad].
            kd: Damping (6,) [Ns/m, Nms/rad]; if None while kp is given and the config
                uses automatic damping, Kd = 2 sqrt(Kp).
        """
        if pos is not None:
            np.copyto(self.pos_d, pos)
        if quat is not None:
            np.copyto(self.quat_d, quat)
        if kp is not None:
            np.copyto(self.kp, kp)
            if kd is None and self.auto_kd:
                critical_damping(self.kp, out=self.kd)
        if kd is not None:
            np.copyto(self.kd, kd)

    def reset(self, plant: "FrankaPlant") -> None:
        """Hold the current TCP pose, restore nominal gains, and clear the rate-limit memory.

        Args:
            plant: Plant after reset + `mj_forward`.
        """
        np.copyto(self.pos_d, plant.ee_pos)
        np.copyto(self.quat_d, plant.ee_quat())
        np.copyto(self.kp, self.kp_nominal)
        np.copyto(self.kd, self.kd_nominal)
        np.copyto(self._tau_prev, plant.qfrc_bias)
        self._tau.fill(0.0)

    def setpoint(self, out: np.ndarray) -> np.ndarray:
        """Write [pos_d(3), quat_d(4), kp(6), kd(6)] into `out`.

        Args:
            out: Buffer of shape (19,).

        Returns:
            `out`.
        """
        out[0:3] = self.pos_d
        out[3:7] = self.quat_d
        out[7:13] = self.kp
        out[13:19] = self.kd
        return out

    @property
    def last_torque(self) -> np.ndarray:
        """Most recent commanded torque [Nm], shape (n,) (owned buffer)."""
        return self._tau

    # ------------------------------------------------------------------ hot loop
    def torque(self, plant: "FrankaPlant") -> np.ndarray:
        """Compute the impedance torque for the current state (no allocation).

        Args:
            plant: Plant after `mj_step1`.

        Returns:
            Owned buffer of joint torques [Nm], shape (n,).
        """
        J = plant.jacobian()
        q, qd = plant.q, plant.qd
        tau = self._tau

        # Task-space error x_d ⊖ x.
        np.subtract(self.pos_d, plant.ee_pos, out=self._err_p)
        quat_error_world(self._err_r, self.quat_d, plant.ee_quat(), self._qb)

        # Wrench F = Kp e - Kd (J qd)
        np.dot(J, qd, out=self._v)
        np.multiply(self.kp, self._err, out=self._F)
        np.multiply(self.kd, self._v, out=self._v)
        np.subtract(self._F, self._v, out=self._F)
        np.dot(J.T, self._F, out=tau)

        # Null-space posture: (I - J^T J#^T) [kn (q_null - q) - dn qd]
        if self.use_nullspace:
            tn = self._tau_null
            np.subtract(self.q_null, q, out=tn)
            np.multiply(self.kn, tn, out=tn)
            np.multiply(self.dn, qd, out=self._tmp_n)
            np.subtract(tn, self._tmp_n, out=tn)
            self._dls.factor(J)
            self._dls.project_nullspace(J, tn, self._tau_proj)
            np.add(tau, self._tau_proj, out=tau)

        # Gravity (+ Coriolis) compensation.
        if self._gravity is None:
            np.add(tau, plant.qfrc_bias, out=tau)
        else:
            np.add(tau, self._gravity.compute(), out=tau)

        # Per-tick rate limit, then saturation.
        if self.rate is not None:
            np.subtract(self._tau_prev, self.rate, out=self._lo)
            np.add(self._tau_prev, self.rate, out=self._hi)
            # maximum/minimum ufuncs instead of clip: clip's Python wrapper allocates per call.
            np.maximum(tau, self._lo, out=tau)
            np.minimum(tau, self._hi, out=tau)
        if self.saturate:
            np.maximum(tau, self._neg_lim, out=tau)
            np.minimum(tau, self._lim, out=tau)
        np.copyto(self._tau_prev, tau)
        return tau
