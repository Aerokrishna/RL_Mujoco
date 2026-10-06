"""Joint-space impedance controller (same interface as `CartesianImpedance`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mujoco_rl_bed.control.base import ControllerCfg
from mujoco_rl_bed.control.math_utils import critical_damping

if TYPE_CHECKING:
    from mujoco_rl_bed.sim.plant import FrankaPlant


@dataclass
class JointImpedanceCfg(ControllerCfg):
    """Configuration for `JointImpedance`.

    Attributes:
        kp: Joint stiffness [Nm/rad], shape (n,) (default: franka_ros joint impedance example).
        kd: Joint damping [Nms/rad], shape (n,); None = 2 sqrt(kp).
        torque_rate_limit: Max |tau_t - tau_{t-1}| per physics tick [Nm]; None = off.
        saturate: Clip torques to the actuator limits.
    """

    kp: tuple[float, ...] = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)
    kd: tuple[float, ...] | None = (50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0)
    torque_rate_limit: float | None = None
    saturate: bool = True

    def build(self, plant: "FrankaPlant") -> "JointImpedance":
        """Instantiate the controller.

        Args:
            plant: Plant to control.

        Returns:
            A `JointImpedance` controller.
        """
        return JointImpedance(plant, self)


class JointImpedance:
    """Joint impedance: tau = Kp (q_d - q) + Kd (qd_d - qd) + qfrc_bias, rate-limited and saturated."""

    def __init__(self, plant: "FrankaPlant", cfg: JointImpedanceCfg) -> None:
        """Allocate buffers.

        Args:
            plant: Plant to control.
            cfg: Controller configuration.
        """
        n = plant.n
        self.n = n
        self.setpoint_dim = 2 * n + 2 * n  # q_d, qd_d, kp, kd
        self.cfg = cfg
        self.kp = np.asarray(cfg.kp, dtype=np.float64).copy()
        self.auto_kd = cfg.kd is None
        self.kd = critical_damping(self.kp) if self.auto_kd else np.asarray(cfg.kd, dtype=np.float64).copy()
        self.kp_nominal = self.kp.copy()
        self.kd_nominal = self.kd.copy()
        self.q_d = plant.h.q_home.copy()
        self.qd_d = np.zeros(n)
        self.rate = cfg.torque_rate_limit
        self.saturate = cfg.saturate
        self._tau = np.zeros(n)
        self._tmp = np.zeros(n)
        self._tau_prev = np.zeros(n)
        self._lo = np.zeros(n)
        self._hi = np.zeros(n)
        self._lim = plant.torque_limits.copy()
        self._neg_lim = -self._lim

    def set_target(self, q: np.ndarray | None = None, qd: np.ndarray | None = None,
                   kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        """Set the joint target and/or gains.

        Args:
            q: Target joint positions [rad], shape (n,).
            qd: Target joint velocities [rad/s], shape (n,).
            kp: Stiffness [Nm/rad], shape (n,).
            kd: Damping [Nms/rad], shape (n,); auto 2 sqrt(kp) if omitted and cfg.kd is None.
        """
        if q is not None:
            np.copyto(self.q_d, q)
        if qd is not None:
            np.copyto(self.qd_d, qd)
        if kp is not None:
            np.copyto(self.kp, kp)
            if kd is None and self.auto_kd:
                critical_damping(self.kp, out=self.kd)
        if kd is not None:
            np.copyto(self.kd, kd)

    def reset(self, plant: "FrankaPlant") -> None:
        """Hold the current configuration and restore nominal gains.

        Args:
            plant: Plant after reset + `mj_forward`.
        """
        np.copyto(self.q_d, plant.q)
        self.qd_d.fill(0.0)
        np.copyto(self.kp, self.kp_nominal)
        np.copyto(self.kd, self.kd_nominal)
        np.copyto(self._tau_prev, plant.qfrc_bias)
        self._tau.fill(0.0)

    def setpoint(self, out: np.ndarray) -> np.ndarray:
        """Write [q_d, qd_d, kp, kd] into `out`.

        Args:
            out: Buffer of shape (4n,).

        Returns:
            `out`.
        """
        n = self.n
        out[0:n] = self.q_d
        out[n:2 * n] = self.qd_d
        out[2 * n:3 * n] = self.kp
        out[3 * n:4 * n] = self.kd
        return out

    @property
    def last_torque(self) -> np.ndarray:
        """Most recent commanded torque [Nm], shape (n,)."""
        return self._tau

    def torque(self, plant: "FrankaPlant") -> np.ndarray:
        """Compute joint impedance torques (no allocation).

        Args:
            plant: Plant after `mj_step1`.

        Returns:
            Owned buffer [Nm], shape (n,).
        """
        tau, tmp = self._tau, self._tmp
        np.subtract(self.q_d, plant.q, out=tau)
        np.multiply(self.kp, tau, out=tau)
        np.subtract(self.qd_d, plant.qd, out=tmp)
        np.multiply(self.kd, tmp, out=tmp)
        np.add(tau, tmp, out=tau)
        np.add(tau, plant.qfrc_bias, out=tau)
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
