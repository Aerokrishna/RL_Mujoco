"""Scripted policies that use privileged simulator state.

They exist to validate tasks (geometry, contacts, reward/success logic) before RL and
to generate demonstrations. They are not meant to be deployable policies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mujoco_rl_bed.env.torque_env import TorqueEnv


class ScriptedPegInsert:
    """Align above the hole, then descend slowly, with retries (for `forge_peg`).

    Works in the `anchor_relative_pos` action space: the desired TCP position relative to
    the anchor is divided by `anchor_bounds`. Lateral aiming is closed-loop on the true
    peg-tip position, so it compensates for small controller errors and peg tilt.

    Phases:
        align:  TCP over the hole axis, peg tip `hover` above the hole tip, until the
                tip's lateral error < `align_tol`.
        insert: lower the target by at most `descend_step` per policy step, down to the tip
                `press` below the hole floor. A fast descent (only λ-limited) makes the
                impedance-controlled TCP drift laterally by more than the 0.25 mm radial
                clearance, and the peg then sticks on the rim.
        If the tip stays on the rim (in contact, not entering) for `stuck_steps`, go back to align.
    """

    def __init__(self, env: "TorqueEnv", hover: float = 0.002, press: float = 0.004, align_tol: float = 0.0001,
                 descend_step: float = 0.001, stuck_steps: int = 4, xy_offset: tuple[float, float] = (0.0, 0.0)) -> None:
        """Cache task geometry.

        Args:
            env: A `forge_peg` env.
            hover: Peg-tip clearance above the hole tip while aligning [m].
            press: Commanded depth below the hole floor during insertion [m].
            align_tol: Lateral tip error that triggers insertion [m].
            descend_step: Max target descent per policy step during insertion [m].
            stuck_steps: Policy steps on the rim before retrying the alignment.
            xy_offset: Deliberate lateral aim offset [m] (to check that misaligned pegs jam).
        """
        self.env = env
        self.st = env.ctx.state["forge"]
        self.bounds = np.asarray(env.cfg.task.action.anchor_bounds, dtype=np.float64)
        self.hover, self.press, self.align_tol = hover, press, align_tol
        self.descend_step, self.stuck_steps = descend_step, stuck_steps
        self.xy_offset = np.asarray(xy_offset, dtype=np.float64)
        self.act_dim = env.action_space.shape[0]  # 4 with success prediction (a_ET)
        self.alpha = float(env.cfg.task.action.ema_factor)  # action smoothing of the action term
        self.reset()

    def reset(self) -> None:
        """Restart from the align phase."""
        self.inserting = False
        self._stuck = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Compute the action from privileged state (the observation is ignored).

        Args:
            obs: Policy observation (unused).

        Returns:
            Action in [-1, 1], shape (act_dim,), float32. With success prediction, a_ET is +1 when
            the task is solved and -1 otherwise (a perfect, privileged predictor).
        """
        st = self.st
        d = self.env.plant.data
        anchor = self.env.ctx.state["fixed_anchor"]
        depth = st.hole_tip[2] - st.hole_floor[2]
        ee = self.env.plant.ee_pos
        tip = d.site_xpos[st.peg_tip]
        tip_err_xy = tip[:2] - (st.hole_tip[:2] + self.xy_offset)
        hover_tcp_z = st.hole_tip[2] + st.tcp_to_tip + self.hover   # world z of the TCP while hovering

        target = np.empty(3)                     # desired TCP position, world frame
        target[:2] = ee[:2] - tip_err_xy         # move the TCP so that the tip lands on the (offset) axis
        on_rim = st.force_norm > 0.3 and st.tip_height > depth - 0.0005
        if self.inserting:
            self._stuck = self._stuck + 1 if on_rim else 0
            if self._stuck >= self.stuck_steps:
                self.inserting, self._stuck = False, 0
        elif np.linalg.norm(tip_err_xy) < self.align_tol and abs(ee[2] - hover_tcp_z) < 0.001:
            self.inserting = True
        if self.inserting:
            target[2] = max(st.hole_floor[2] + st.tcp_to_tip - self.press, ee[2] - self.descend_step)
        else:
            target[2] = hover_tcp_z
        # The action is the target relative to the anchor, whatever the anchor convention is.
        desired = np.clip((target - anchor) / self.bounds, -1.0, 1.0)
        if self.alpha < 1.0:
            # Invert the EMA s_t = alpha * a_t + (1 - alpha) * s_{t-1} so the applied action equals `desired`
            # (as far as the [-1, 1] action bounds allow).
            prev = self.env.ctx.applied_action[:3]
            desired = np.clip((desired - (1.0 - self.alpha) * prev) / self.alpha, -1.0, 1.0)
        a = np.empty(self.act_dim, dtype=np.float32)
        a[:3] = desired
        if self.act_dim > 3:
            a[3] = 1.0 if st.success else -1.0
        return a
