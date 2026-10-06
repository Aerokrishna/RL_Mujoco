"""FrankaPlant: torque-level robot interface over one MjModel/MjData pair.

The plant knows nothing about tasks. It owns preallocated buffers and cached numpy
views into `MjData` so the per-tick path (`control_step` plus the accessors a
controller calls) performs no numpy allocation and no name lookups.

View semantics: `q`, `qd`, `ee_pos`, `qfrc_bias` and `ctrl_arm` are *views* into
`MjData` memory. They always reflect the current state and must not be stored as
snapshots; call `.copy()` if a snapshot is needed. Methods such as `jacobian()` and
`ee_quat()` write into plant-owned buffers and return them (overwritten on the next call).
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from mujoco_rl_bed.sim.scene import SceneHandles


class FrankaPlant:
    """Torque API and state accessors for the arm (float64 throughout).

    Attributes:
        model: The env-owned `MjModel` (may be randomized in place by events).
        data: The single `MjData`, reused across resets (never reallocated).
        h: Resolved `SceneHandles`.
        n: Number of arm joints.
    """

    def __init__(self, model: mujoco.MjModel, handles: SceneHandles, data: mujoco.MjData | None = None) -> None:
        """Allocate data (once) and all buffers/views.

        Args:
            model: Compiled scene model.
            handles: Handles returned by `SceneBuilder.build()`.
            data: Optional existing `MjData` for `model`; created if None.
        """
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.h = handles
        self.n = handles.n_arm
        m, d, h = self.model, self.data, handles

        # --- cached views into MjData (basic slicing -> views; MjData memory is fixed for its lifetime)
        self._q = d.qpos[h.arm_qpos]
        self._qd = d.qvel[h.arm_dof]
        self._ctrl_arm = d.ctrl[h.arm_ctrl]
        self._bias = d.qfrc_bias[h.arm_dof]
        self._site_xpos = d.site_xpos[h.tcp_site_id]           # (3,)
        self._site_xmat = d.site_xmat[h.tcp_site_id]           # (9,) row-major rotation
        self._ft_site_xmat = d.site_xmat[h.ft_site_id]         # (9,)
        self._ft_force = d.sensordata[h.ft_force]              # (3,) in ft_site frame
        self._ft_torque = d.sensordata[h.ft_torque]            # (3,) in ft_site frame
        self._tlim = h.torque_limits

        # --- preallocated buffers
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._jacp_arm = self._jacp[:, h.arm_dof]               # views on the arm columns
        self._jacr_arm = self._jacr[:, h.arm_dof]
        self._J = np.zeros((6, self.n))
        self._J_lin = self._J[:3]
        self._J_ang = self._J[3:]
        self._quat = np.zeros(4)
        self._ee_vel = np.zeros(6)
        self._wrench = np.zeros(6)
        self._wrench_f = self._wrench[:3]
        self._wrench_t = self._wrench[3:]
        self._ft_rot = self._ft_site_xmat.reshape(3, 3)       # view, no copy (contiguous row)

        # Gripper ctrl helpers.
        self._gid = h.gripper_act_id
        self._glo, self._ghi = (float(h.gripper_ctrlrange[0]), float(h.gripper_ctrlrange[1]))

    # ------------------------------------------------------------------ state (views)
    @property
    def q(self) -> np.ndarray:
        """Arm joint positions [rad], shape (n,), view into `data.qpos`."""
        return self._q

    @property
    def qd(self) -> np.ndarray:
        """Arm joint velocities [rad/s], shape (n,), view into `data.qvel`."""
        return self._qd

    @property
    def ee_pos(self) -> np.ndarray:
        """TCP position in world frame [m], shape (3,), view into `data.site_xpos`."""
        return self._site_xpos

    @property
    def ee_rot(self) -> np.ndarray:
        """TCP rotation matrix (row-major, flattened), shape (9,), view into `data.site_xmat`."""
        return self._site_xmat

    @property
    def qfrc_bias(self) -> np.ndarray:
        """Arm slice of gravity + Coriolis/centrifugal forces [Nm], shape (n,), view."""
        return self._bias

    @property
    def ctrl_arm(self) -> np.ndarray:
        """Arm motor commands currently applied [Nm], shape (n,), view into `data.ctrl`."""
        return self._ctrl_arm

    @property
    def torque_limits(self) -> np.ndarray:
        """Per-joint torque limits [Nm], shape (n,)."""
        return self._tlim

    @property
    def time(self) -> float:
        """Simulation time [s]."""
        return self.data.time

    @property
    def dt(self) -> float:
        """Physics timestep [s]."""
        return self.model.opt.timestep

    # ------------------------------------------------------------------ derived quantities (buffers)
    def ee_quat(self) -> np.ndarray:
        """TCP orientation as a unit quaternion (w, x, y, z).

        Returns:
            Plant-owned buffer, shape (4,), overwritten on the next call.
        """
        mujoco.mju_mat2Quat(self._quat, self._site_xmat)
        return self._quat

    def jacobian(self) -> np.ndarray:
        """Geometric TCP Jacobian on the arm joints, world frame.

        Rows 0-2 map qd to linear velocity [m/s], rows 3-5 to angular velocity [rad/s].
        Valid after `mj_step1`/`mj_forward` (uses current `site_xpos` and `subtree_com`).

        Returns:
            Plant-owned buffer, shape (6, n), overwritten on the next call.
        """
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.h.tcp_site_id)
        np.copyto(self._J_lin, self._jacp_arm)
        np.copyto(self._J_ang, self._jacr_arm)
        return self._J

    def ee_vel(self, J: np.ndarray | None = None) -> np.ndarray:
        """TCP twist (v, w) in world frame [m/s, rad/s], computed as J qd.

        Args:
            J: Precomputed Jacobian (6, n) to avoid recomputation; None = call `jacobian()`.

        Returns:
            Plant-owned buffer, shape (6,), overwritten on the next call.
        """
        np.dot(self.jacobian() if J is None else J, self._qd, out=self._ee_vel)
        return self._ee_vel

    def wrist_wrench(self, world_frame: bool = False) -> np.ndarray:
        """Flange force/torque sensor reading (force [N], torque [Nm]).

        MuJoCo's force/torque sensors report the wrench exerted on the `ee_body`
        subtree by its parent, in the `ft_site` frame, as of the last `mj_step2`/`mj_forward`.

        Args:
            world_frame: If True, rotate both vectors into the world frame.

        Returns:
            Plant-owned buffer, shape (6,), overwritten on the next call.
        """
        if world_frame:
            np.dot(self._ft_rot, self._ft_force, out=self._wrench_f)   # R_site->world @ f_site
            np.dot(self._ft_rot, self._ft_torque, out=self._wrench_t)
        else:
            np.copyto(self._wrench_f, self._ft_force)
            np.copyto(self._wrench_t, self._ft_torque)
        return self._wrench

    # ------------------------------------------------------------------ actuation
    def set_joint_torque(self, tau: np.ndarray) -> None:
        """Write arm motor torques (clamped by the actuator `ctrlrange` inside MuJoCo).

        Args:
            tau: Joint torques [Nm], shape (n,).
        """
        np.copyto(self._ctrl_arm, tau)

    def set_gripper(self, opening: float) -> None:
        """Command the gripper through its original (position) actuator.

        Args:
            opening: Normalized opening in [0, 1] (0 = closed, 1 = fully open);
                mapped linearly onto the actuator's ctrlrange.
        """
        if self._gid >= 0:
            o = min(max(opening, 0.0), 1.0)
            self.data.ctrl[self._gid] = self._glo + o * (self._ghi - self._glo)

    def control_step(self, torque_fn: Callable[["FrankaPlant"], np.ndarray]) -> None:
        """Advance one physics tick with a torque computed on the current state.

        Sequence: `mj_step1` (kinematics, Jacobian inputs, `qfrc_bias`) ->
        `ctrl = torque_fn(self)` -> `mj_step2` (actuation, constraints, integration).
        The controller therefore sees the state at the beginning of this tick, with no
        one-step lag in the Jacobian or bias terms.

        Args:
            torque_fn: Callable returning joint torques [Nm], shape (n,). It may return a
                buffer it owns; the values are copied into `data.ctrl`.
        """
        m, d = self.model, self.data
        mujoco.mj_step1(m, d)
        np.copyto(self._ctrl_arm, torque_fn(self))
        mujoco.mj_step2(m, d)

    def step_physics(self) -> None:
        """Advance one physics tick with the currently stored `ctrl` (no controller)."""
        mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ reset
    def reset(self, q0: np.ndarray | None = None, qd0: np.ndarray | None = None,
              gripper_open: float | None = None) -> None:
        """Reset the simulation state in place and recompute derived quantities.

        Calls `mj_resetData` (zeroes ctrl, xfrc_applied, time, warm starts), sets the arm
        and finger configuration, then `mj_forward` so all accessors are valid.

        Args:
            q0: Arm joint positions [rad], shape (n,); None = `handles.q_home`.
            qd0: Arm joint velocities [rad/s], shape (n,); None = zeros.
            gripper_open: Initial opening in [0, 1]; None = `handles.gripper_open`.
        """
        m, d, h = self.model, self.data, self.h
        mujoco.mj_resetData(m, d)
        self._q[:] = h.q_home if q0 is None else q0
        if qd0 is not None:
            self._qd[:] = qd0
        o = h.gripper_open if gripper_open is None else gripper_open
        if h.finger_qpos_adr.size:
            lo, hi = h.finger_qpos_range[:, 0], h.finger_qpos_range[:, 1]
            d.qpos[h.finger_qpos_adr] = lo + o * (hi - lo)
        self.set_gripper(o)
        mujoco.mj_forward(m, d)
