"""Privileged scripted corrector for `compliant_peg` (validates the residual action space; not deployable).

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

import numpy as np

if TYPE_CHECKING:
    from mujoco_rl_bed.env.torque_env import TorqueEnv


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
