"""Scripted policies for the `compliant_peg` tasks.

`CatchRule`: the deployable catch rule for the spiral tasks with a try-insert action (`compliant_peg_s2f/g/h`). It
reads only the actor observation and is the teacher of the warm start (`bc_warmstart.py`).

`ScriptedCorrector`: privileged scripted corrector for `compliant_peg` (validates the residual action space; not
deployable).

`LocalOracle`: privileged local-phase corrector for `compliant_peg_s3` (`spiral_local`): the ceiling of the local
x, y, yaw correction (not deployable).

It reads the true peg tip and grasp tilt and outputs the same 5D residual as the policy:

    f_lat = -K e - Ki ∫e + k_lat (x_c - p_ref)      PI on the tip's lateral error e (command frame),
                                                     plus the lateral spring's pull (feed-forward)
    Δf    = -1 (push f_push_min) while |e| > tol and the tip is above the hole, else 0 (nominal push)
    tilt  = -grasp tilt                              cancels the peg's tilt in the hand

With the defaults it inserts every randomized episode in about 2 s.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mujoco_rl_bed.env.torque_env import TorqueEnv


class CatchRule:
    """Try insert when the peg is caught on the hole edge: |F_xy| > `f_th` for `k` search steps -> try for `hold` steps.

    Works on the `compliant_peg_s2g` / `s2i` actor observation (wrench in [0:6], spiral phase / 2 in [`phase_idx`]:
    15 for s2g, 13 for s2i); returns the try action (`no_try` when not trying, 1 while trying; 0 / 1 for s2i). A `cool`-step cooldown follows every try. Near misses
    where the hole edge catches the peg give a 3-5 N lateral force toward the hole; friction spikes rarely last
    3 steps above 3.5 N (92-93 % of the tries are real catches).
    """

    def __init__(self, f_th: float = 3.5, k: int = 3, hold: int = 15, cool: int = 10, no_try: float = -1.0,
                 phase_idx: int = 15) -> None:
        """Args:
            f_th: Lateral force threshold [N].
            k: Consecutive search steps above the threshold.
            hold: Try length [policy steps] (15 = 0.6 s).
            cool: Cooldown after a try [policy steps].
            no_try: Action when not trying (any value below the try dead zone, 0.3).
            phase_idx: Index of the spiral phase / 2 in the observation.
        """
        self.f_th, self.k, self.hold, self.cool, self.no_try = f_th, k, hold, cool, no_try
        self.phase_idx = phase_idx
        self.reset()

    def reset(self) -> None:
        """Clear the counters (call at every episode start)."""
        self.cnt = self.left = self.cd = 0

    def __call__(self, o: np.ndarray) -> float:
        """Try action for actor observation `o`."""
        if o[self.phase_idx] != 0.5:                       # not in the search phase
            return self.no_try
        self.cnt = self.cnt + 1 if math.hypot(o[0], o[1]) > self.f_th else 0
        if self.left > 0:
            self.left -= 1
            if self.left == 0:
                self.cd = self.cool
        elif self.cd > 0:
            self.cd -= 1
        elif self.cnt >= self.k:
            self.left = self.hold
        return 1.0 if self.left > 0 else self.no_try


class ScriptedCorrector:
    """PI lateral correction + grasp-tilt cancellation through `axis_compliance_residual`."""

    def __init__(self, env: "TorqueEnv", K: float = 400.0, Ki: float = 800.0, tol: float = 0.00015) -> None:
        """Cache handles.

        Args:
            env: A `compliant_peg` env.
            K: Proportional gain on the tip's lateral error [N/m].
            Ki: Integral gain [N/(m s)].
            tol: Lateral error below which the nominal push is applied [m].
        """
        self.env = env
        self.st = env.ctx.state["forge"]
        self.c = env.ctx.controller
        p = env.cfg.task.action.params
        self.f_lat_max = float(p["f_lat_max"])
        self.tilt_max = math.radians(float(p["tilt_max_deg"]))
        self.dt = env.policy_dt
        self.K, self.Ki, self.tol = K, Ki, tol
        self.I = np.zeros(2)

    def reset(self) -> None:
        """Clear the integrator (call after `env.reset`)."""
        self.I.fill(0.0)

    def act(self) -> np.ndarray:
        """Action for the current state.

        Returns:
            Action in [-1, 1]^5.
        """
        st, c, d = self.st, self.c, self.env.ctx.data
        tip = d.site_xpos[st.peg_tip]
        e = (c.R_f.T @ np.r_[tip[:2] - st.hole_tip[:2], 0.0])[:2]
        self.I += e * self.dt
        f = -self.K * e - self.Ki * self.I + c.k_lin[:2] * (c.R_f.T @ (c.x_c - c.p_ref))[:2]
        a = np.zeros(5)
        a[0:2] = np.clip(f / self.f_lat_max, -1.0, 1.0)
        a[2] = -1.0 if np.linalg.norm(e) > self.tol and tip[2] > st.hole_tip[2] - 0.002 else 0.0
        a[3:5] = np.clip(-self.env.ctx.state["grasp_tilt"] / self.tilt_max, -1.0, 1.0)
        return a


class LocalOracle:
    """Privileged local phase for `spiral_local`: steer the true peg tip over the hole axis and turn to the hole yaw.

    Per step in the local phase: the anchor offset integrates the tip's lateral error (command frame, gain `gain`
    per step, clipped to the action range), and the yaw command integrates the peg-to-hole yaw error (wrapped to
    ± half the shape's symmetry angle `sym_deg`: 90° square, 60° hexagon, 45° octagon; a positive yaw command turns
    the peg negatively about world z because the tool axis points down). Outside the local phase it holds the last yaw command. `xy=False` / `yaw=False` disable a part.
    """

    def __init__(self, env: "TorqueEnv", gain: float = 0.3, yaw_gain: float = 0.5, xy: bool = True,
                 yaw: bool = True, sym_deg: float = 90.0) -> None:
        """Cache handles.

        Args:
            env: A `compliant_peg_s3` env.
            gain: Lateral integral gain (fraction of the tip error added to the offset per step).
            yaw_gain: Yaw integral gain (fraction of the yaw error added per step).
            xy: Correct the lateral offset.
            yaw: Correct the yaw.
            sym_deg: Rotational symmetry of the cross-section [deg] (90 square, 60 hexagon, 45 octagon).
        """
        ctx = env.ctx
        self.ctx, self.st, self.sp, self.term = ctx, ctx.state["forge"], ctx.state["spiral"], ctx.state["spiral_term"]
        m = ctx.model
        self.peg_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        self.hole_body = int(np.flatnonzero(m.body_mocapid == self.st.hole_mocap)[0])
        self.gain, self.yaw_gain, self.xy, self.yaw = gain, yaw_gain, xy, yaw
        self.sym = math.radians(sym_deg)
        self.reset()

    def reset(self) -> None:
        """Clear the integrators (call after `env.reset`)."""
        self.off = np.zeros(2)
        self.ycmd = 0.0
        self.n_local = 0

    def yaw_error(self) -> float:
        """Hole yaw minus peg yaw about world z, wrapped to [-sym / 2, sym / 2) [rad]."""
        d = self.ctx.data
        pg, hb = d.geom_xmat[self.peg_geom], d.xmat[self.hole_body]
        dphi = math.atan2(hb[3], hb[0]) - math.atan2(pg[3], pg[0])
        return (dphi + 0.5 * self.sym) % self.sym - 0.5 * self.sym

    def __call__(self, o: np.ndarray | None = None) -> np.ndarray:
        """Local action [a_x, a_y, a_yaw] for the current state."""
        t, sp = self.term, self.sp
        if sp["phase"] != 3:
            return np.array([0.0, 0.0, self.ycmd / t.yaw_max])
        if sp["n_local"] != self.n_local:                   # a new local phase starts at the freeze point
            self.n_local, self.off[:] = sp["n_local"], 0.0
        if self.xy:
            tip, hole = self.ctx.data.site_xpos[self.st.peg_tip], self.st.hole_tip
            e = t._R[:, :2].T @ np.array([hole[0] - tip[0], hole[1] - tip[1], 0.0])
            self.off = np.clip(self.off + self.gain * e, -t.local_max, t.local_max)
        if self.yaw:
            self.ycmd = float(np.clip(self.ycmd - self.yaw_gain * self.yaw_error(), -t.yaw_max, t.yaw_max))
        return np.array([self.off[0] / t.local_max, self.off[1] / t.local_max, self.ycmd / t.yaw_max])
