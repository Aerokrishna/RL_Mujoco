"""Terms of the residual-RL insertion task on top of the axis compliance controller.

Reused from `forge.terms`: `ForgeState` (geometry, keypoint distance, success, contact-force statistics,
episode metrics), the socket reset with a noisy hole estimate, friction randomization, the keypoint
kernels, success bonus, force penalty and the privileged critic terms. New here:

- action term `axis_compliance_residual`: 5D residual [f_x, f_y, Δf_push, tilt_x, tilt_y] for
  `AxisCompliance`, in its command frame (the TCP frame at the start of the episode, fixed while the peg
  tilts); all zeros = the base controller (push f_push_nominal along the start TCP z, no side force, no
  tilt),
- events: peg tilt in the grasp (`cr_randomize_grasp`), start pose with the peg tip just above the
  estimated hole (`cr_reset_ee`),
- observations in the TCP frame: estimated hole position and axis, contact wrench, TCP twist,
- rewards: xy-aligned and axis-aligned bonuses.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from forge.terms import ANCHOR, _st, logistic_kernel
from mujoco_rl_bed.env.cfg import ActionCfg
from mujoco_rl_bed.env.managers.action import ActionTerm, action_term
from mujoco_rl_bed.env.managers.events import event_term
from mujoco_rl_bed.env.managers.observation import obs_term
from mujoco_rl_bed.env.managers.reward import reward_term
from mujoco_rl_bed.env.managers.termination import termination_term
from mujoco_rl_bed.sim.ik import solve_tcp_ik

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context


# ---------------------------------------------------------------------------------- action
@action_term("axis_compliance_residual")
class AxisComplianceResidual(ActionTerm):
    """Residual commands for `AxisCompliance`, a in [-1, 1]^5 (EMA-smoothed with `cfg.ema_factor`):

        f_lat  = s[0:2] * f_lat_max                              side force along x, y [N]
        f_push = clip(f_push_nominal + s[2] * f_push_range,      push along z [N]
                      f_push_min, f_push_max)
        tilt   = s[3:5] * tilt_max_deg                           tilt target about x, y [deg -> rad]

    in the controller's command frame (start TCP frame, fixed for the episode: a sideways force stays
    sideways when the peg tilts, and the push axis is not tilted, so tilt and side force do not overlap).
    `cfg.params`: f_lat_max (5 N), f_push_nominal (8 N), f_push_range (7 N), f_push_min (1 N),
    f_push_max (15 N), tilt_max_deg (10). s = 0 is the base controller.
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Read the limits and allocate buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        p = cfg.params
        self.dim = 5
        self.f_lat_max = float(p.get("f_lat_max", 5.0))
        self.f0 = float(p.get("f_push_nominal", 8.0))
        self.f_range = float(p.get("f_push_range", 7.0))
        self.f_min = float(p.get("f_push_min", 1.0))
        self.f_max = float(p.get("f_push_max", 15.0))
        self.tilt_max = math.radians(float(p.get("tilt_max_deg", 10.0)))
        self._ema = ctx.buffer("action_ema", 1)
        self._ema[0] = float(cfg.ema_factor)
        ctx.buffer("action_max_step", 3)  # read by ForgeState (λ is not used by this controller)
        self._s = np.zeros(5)
        self._tmp = np.zeros(5)
        self._f_lat = np.zeros(2)
        self._tilt = np.zeros(2)

    def apply(self, a: np.ndarray) -> None:
        """Smooth the action and send the commands to the controller.

        Args:
            a: Clipped action, shape (5,).
        """
        alpha = float(self._ema[0])
        np.multiply(a, alpha, out=self._tmp)
        self._s *= 1.0 - alpha
        self._s += self._tmp
        s = self._s
        np.multiply(s[0:2], self.f_lat_max, out=self._f_lat)
        np.multiply(s[3:5], self.tilt_max, out=self._tilt)
        self.ctx.controller.set_target(f_lat=self._f_lat, f_push=min(self.f_max, max(self.f_min, self.f0 + s[2] * self.f_range)),
                                       tilt=self._tilt)

    def reset(self) -> None:
        """Start from the base controller (zero residual) and push with the nominal force."""
        self._s.fill(0.0)
        self.ctx.controller.set_target(f_lat=np.zeros(2), f_push=self.f0, tilt=np.zeros(2))

    def write_applied(self, out: np.ndarray) -> None:
        """Applied (smoothed) action.

        Args:
            out: Buffer of shape (5,).
        """
        np.copyto(out, self._s)


@action_term("axis_compliance_anchor")
class AxisComplianceAnchor(ActionTerm):
    """Lateral anchor shift + push force for `AxisCompliance`, a in [-1, 1]^3 (EMA-smoothed):

        p_ref  = p_start + R_f [s[0:2] * anchor_max, 0]          lateral spring anchor [m]
        f_push = clip(f_push_nominal + s[2] * f_push_range,      push along the motion axis [N]
                      f_push_min, f_push_max)

    p_start is the reference position at reset and R_f the controller's command frame (start TCP frame,
    fixed for the episode). The lateral spring pulls the peg toward the commanded anchor, so the side
    force is k_lat * (anchor - position) and the peg still yields to contact. No side force, no tilt
    command (tilt is left to the compliance). `cfg.params`: anchor_max (4 mm), f_push_nominal (4.5 N),
    f_push_range (3.5 N), f_push_min (1 N), f_push_max (8 N).
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Read the limits and allocate buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        p = cfg.params
        self.dim = 3
        self.anchor_max = float(p.get("anchor_max", 0.004))
        self.f0 = float(p.get("f_push_nominal", 4.5))
        self.f_range = float(p.get("f_push_range", 3.5))
        self.f_min = float(p.get("f_push_min", 1.0))
        self.f_max = float(p.get("f_push_max", 8.0))
        self._ema = ctx.buffer("action_ema", 1)
        self._ema[0] = float(cfg.ema_factor)
        ctx.buffer("action_max_step", 3)  # read by ForgeState (λ is not used by this controller)
        self._s = np.zeros(3)
        self._tmp = np.zeros(3)
        self._p_start = np.zeros(3)
        self._p = np.zeros(3)

    def apply(self, a: np.ndarray) -> None:
        """Smooth the action and send the anchor and push force to the controller.

        Args:
            a: Clipped action, shape (3,).
        """
        alpha = float(self._ema[0])
        np.multiply(a, alpha, out=self._tmp)
        self._s *= 1.0 - alpha
        self._s += self._tmp
        s, c = self._s, self.ctx.controller
        np.dot(c.R_f[:, :2], s[0:2] * self.anchor_max, out=self._p)
        self._p += self._p_start
        c.set_target(pos=self._p, f_push=min(self.f_max, max(self.f_min, self.f0 + s[2] * self.f_range)))

    def reset(self) -> None:
        """Anchor at the start position, nominal push, no side force or tilt."""
        self._s.fill(0.0)
        c = self.ctx.controller
        np.copyto(self._p_start, c.p_ref)
        c.set_target(pos=self._p_start, f_lat=np.zeros(2), f_push=self.f0, tilt=np.zeros(2))

    def write_applied(self, out: np.ndarray) -> None:
        """Applied (smoothed) action.

        Args:
            out: Buffer of shape (3,).
        """
        np.copyto(out, self._s)


@action_term("spiral_residual")
class SpiralResidual(ActionTerm):
    """Scripted spiral search for `AxisCompliance` plus a residual, a in [-1, 1]^3 (EMA-smoothed).

    Base (zero residual), evaluated once per policy step in the command frame R_f (start TCP frame):

        approach  push f_search along R_f z, anchor at the start position (the hole estimate),
                  search gains (ks_lat, ds_lat, ks_tilt, ds_tilt): stiff lateral, stiff tilt;
                  touchdown when the contact force against the push exceeds f_search / 2
        search    Archimedean spiral of the lateral anchor around the start position: pitch, constant anchor
                  speed v, out to r_max and back; drop when the TCP has moved drop_th further along R_f z
                  than at touchdown
        insert    task gains (the controller config: soft lateral, soft tilt), push f_insert, anchor held at
                  the TCP position
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Read the parameters and allocate buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        p = cfg.params
        self.try_action = bool(p.get("try_action", False))
        self.speed_action = bool(p.get("speed_action", True)) and not self.try_action
        self.lateral_action = bool(p.get("lateral_action", True)) or not self.try_action
        self.dim = (3 if self.lateral_action else 1) if self.try_action else (3 if self.speed_action else 2)
        self.try_macro = int(p.get("try_macro", 0))
        if self.try_macro and (not self.try_action or self.lateral_action):
            raise ValueError("try_macro needs try_action=True and lateral_action=False")
        self.n_discrete = 2 if self.try_macro else 0
        self._macro_left = 0
        self.f_try = float(p.get("f_try", 5.0))
        self.try_dead = float(p.get("try_dead", 0.3))
        self.f_search = float(p.get("f_search", 1.0))
        self.f_insert = float(p.get("f_insert", 5.0))
        self.v = float(p.get("v", 0.01))
        self.b = float(p.get("pitch", 0.0008)) / (2.0 * math.pi)
        self.r_max = float(p.get("r_max", 0.0026))
        self.drop_th = float(p.get("drop_th", 0.002))
        self.res_max = float(p.get("res_max", 0.001))
        self.speed_up = float(p.get("speed_max", 2.0)) - 1.0
        self.speed_affine = bool(p.get("speed_affine", False))
        self.speed_lo, self.speed_hi = float(p.get("speed_lo", 0.0)), float(p.get("speed_hi", 1.0))
        self.search_gains = (float(p.get("ks_lat", 2000.0)), float(p.get("ds_lat", 190.0)),
                             float(p.get("ks_tilt", 3.0)), float(p.get("ds_tilt", 1.5)))
        c = ctx.controller
        self.task_gains = (c.k_lin[0], c.d_lin[0], c.k_rot[0], c.d_rot[0])
        self._lg_s, self._lg_t = np.log(self.search_gains), np.log(self.task_gains)
        self._try = np.zeros(4)              # try steps, onsets, try steps / onsets with the tip within 0.5 mm
        self._ema = ctx.buffer("action_ema", 1)
        self._ema[0] = float(cfg.ema_factor)
        ctx.buffer("action_max_step", 3)  # read by ForgeState (λ is not used by this controller)
        self.state = ctx.state["spiral"] = {"phase": 0, "offset": np.zeros(2), "depth": 0.0, "tau": 0.0, "try_t": 0.0}
        ctx.state["spiral_term"] = self
        n = 3 if self.dim == 1 else self.dim  # try-only: internal [0, 0, try]
        self._s = np.zeros(n)
        self._tmp = np.zeros(n)
        self._a3 = np.zeros(3)
        self._m = np.zeros((2, 4))           # per phase (search, insert): sum |rx|, sum |ry|, sum toward-hole, n
        self._sp = np.zeros((2, 2))          # search speed scale: (near < 1 mm, far) x (sum, n)
        ctx.episode_info_hooks.append(self._episode_info)
        self._w = np.zeros(6)
        self._R = np.eye(3)
        self._p0 = np.zeros(3)
        self._hold = np.zeros(3)
        self._p = np.zeros(3)
        self._theta = 0.0
        self._out = True
        self._z_touch = 0.0

    def _gains(self, g: tuple[float, float, float, float]) -> None:
        """Set the lateral and roll/pitch gains of the controller's virtual frame."""
        c = self.ctx.controller
        c.k_lin[:2], c.d_lin[:2], c.k_rot[:2], c.d_rot[:2] = g

    def apply(self, a: np.ndarray) -> None:
        """Advance the base search by one policy step and send base + residual to the controller.

        Args:
            a: Clipped action, shape (dim,).
        """
        if not self.lateral_action:
            self._a3[2] = a[0]
            a = self._a3
        alpha = float(self._ema[0])
        np.multiply(a, alpha, out=self._tmp)
        self._s *= 1.0 - alpha
        self._s += self._tmp
        if self.try_action:
            self._s[2] = a[2]                # the try level switches without smoothing
            self.state["a_try"] = float(a[2])
        s, c, st, R = self._s, self.ctx.controller, self.state, self._R
        ee = self.ctx.plant.ee_pos
        dt = self.ctx.policy_dt
        z = float(R[:, 2] @ ee)
        tau = 0.0
        if st["phase"] == 0:
            peg_wrench_world(self.ctx, self._w)
            if -(R[:, 2] @ self._w[:3]) > 0.5 * self.f_search:
                st["phase"], self._z_touch = 1, z
        if st["phase"] == 1:
            if z - self._z_touch > self.drop_th:
                st["phase"] = 2
                self._macro_left = 0
                np.copyto(self._hold, ee)
                self._gains(self.task_gains)
            else:
                r = self.b * self._theta
                if self._out and r >= self.r_max:
                    self._out = False
                elif not self._out and r <= 0.0:
                    self._out = True
                if self.try_macro:
                    if self._macro_left == 0 and s[2] >= 0.5:
                        self._macro_left = self.try_macro
                    tau = 1.0 if self._macro_left > 0 else 0.0
                    self._macro_left = max(self._macro_left - 1, 0)
                    speed = 1.0 - tau
                    self._gains(self.task_gains if tau == 1.0 else self.search_gains)
                elif self.try_action:
                    tau = min(max((s[2] - self.try_dead) / (1.0 - self.try_dead), 0.0), 1.0)
                    speed = 1.0 - tau
                    if tau == 0.0 or tau == 1.0:     # exact end points (zero action = the scripted spiral)
                        self._gains(self.task_gains if tau == 1.0 else self.search_gains)
                    else:
                        self._gains(tuple(np.exp(self._lg_s + tau * (self._lg_t - self._lg_s))))
                elif not self.speed_action:
                    speed = 1.0
                elif self.speed_affine:
                    speed = self.speed_lo + (self.speed_hi - self.speed_lo) * 0.5 * (1.0 + s[2])
                else:
                    speed = 1.0 + (s[2] if s[2] <= 0.0 else s[2] * self.speed_up)
                st["speed"] = speed
                dth = speed * self.v * dt / math.sqrt(r * r + self.b * self.b)
                self._theta = max(0.0, self._theta + (dth if self._out else -dth))
                r = self.b * self._theta
                st["offset"][:] = r * math.cos(self._theta), r * math.sin(self._theta)
        onset = tau > 0.0 and st["tau"] == 0.0
        if tau > 0.0:
            if onset:
                self._try[1] += 1.0
            st["try_t"] += dt
        else:
            st["try_t"] = 0.0
        st["tau"] = tau
        if st["phase"] >= 1:
            st["depth"] = z - self._z_touch
        base = self._hold if st["phase"] == 2 else self._p0 + R[:, :2] @ st["offset"]
        if st["phase"] >= 1:
            fs = _st(self.ctx)
            to_hole = R[:, :2].T @ (fs.hole_tip - self.ctx.data.site_xpos[fs.peg_tip])
            n = float(np.linalg.norm(to_hole))
            m = self._m[st["phase"] - 1]
            m[0] += abs(s[0]) * self.res_max
            m[1] += abs(s[1]) * self.res_max
            m[2] += (s[0:2] @ to_hole) * self.res_max / n if n > 1e-9 else 0.0
            m[3] += 1.0
            if tau > 0.0:
                self._try[0] += 1.0
                self._try[2] += n < 0.0005
                self._try[3] += onset and n < 0.0005
            if st["phase"] == 1 and "speed" in st:
                j = 0 if n < 0.001 else 1
                self._sp[j, 0] += st["speed"]
                self._sp[j, 1] += 1.0
        if self.try_macro:
            s[2] = tau                       # applied action = the try flag
        np.dot(R[:, :2], s[0:2] * self.res_max, out=self._p)
        self._p += base
        f = self.f_insert if st["phase"] == 2 else self.f_search + tau * (self.f_try - self.f_search)
        c.set_target(pos=self._p, f_push=f)

    def reset(self) -> None:
        """Start the approach: search gains, anchor at the start position, push f_search."""
        self._s.fill(0.0)
        self._m.fill(0.0)
        self._sp.fill(0.0)
        self._try.fill(0.0)
        c, st = self.ctx.controller, self.state
        st["tau"], st["try_t"] = 0.0, 0.0
        self._macro_left = 0
        np.copyto(self._R, c.R_f)
        np.copyto(self._p0, c.p_ref)
        st["phase"], st["depth"] = 0, 0.0
        st["offset"].fill(0.0)
        self._theta, self._out, self._z_touch = 0.0, True, 0.0
        self._gains(self.search_gains)
        c.set_target(pos=self._p0, f_lat=np.zeros(2), f_push=self.f_search, tilt=np.zeros(2))

    def write_applied(self, out: np.ndarray) -> None:
        """Applied (smoothed) residual.

        Args:
            out: Buffer of shape (dim,).
        """
        np.copyto(out, self._s if self.lateral_action else self._s[2:])

    def _episode_info(self, ctx: "Context") -> dict:
        """Residual usage over the episode [mm] (logging only)."""
        out = {}
        for i, ph in enumerate(("search", "insert")):
            m = self._m[i]
            k = max(m[3], 1.0)
            out[f"res_{ph}_absx_mm"] = 1e3 * m[0] / k
            out[f"res_{ph}_absy_mm"] = 1e3 * m[1] / k
            out[f"res_{ph}_toward_hole_mm"] = 1e3 * m[2] / k
        if self.speed_action:
            for j, nm in enumerate(("near", "far")):
                out[f"speed_{nm}_hole"] = self._sp[j, 0] / self._sp[j, 1] if self._sp[j, 1] > 0 else float("nan")
        if self.try_action:
            out["try_steps"] = self._try[0]
            out["try_onsets"] = self._try[1]
            out["try_true_frac"] = self._try[2] / self._try[0] if self._try[0] > 0 else 0.0
            out["try_onset_true_frac"] = self._try[3] / self._try[1] if self._try[1] > 0 else 0.0
        return out


@action_term("spiral_local")
class SpiralLocal(SpiralResidual):
    """`spiral_residual`'s scripted spiral (no residual) plus a local x, y, yaw correction phase after a trigger.

    Phases (`ctx.state["spiral"]["phase"]`): 0 approach, 1 search, 3 local, 2 insert. The action only acts in the
    local phase. A trigger (`ctx.state["spiral"]["trigger"] = True`, set by the caller before a step, e.g. from a
    force rule; read in the search phase only) freezes the spiral and starts the local phase:

        lateral anchor = freeze point + R_f [a_x, a_y] local_max      (search gains: stiff lateral, stiff tilt)
        yaw target     = a_yaw yaw_max about the tool axis             (controller yaw spring)
        push           = f_local

    with a = the EMA-smoothed action (`ema_factor`). A drop (TCP `drop_th` further along R_f z than at touchdown)
    switches to insert as in the spiral; after `t_local` [s] without one the spiral resumes where it stopped (the
    yaw target is kept). `n_local` counts the local phases of the episode. With `start_local` the first local phase
    starts right after touchdown (training on local-phase starts; the start offset comes from the reset events).

    `trigger` selects who starts the local phase: "external" (the caller sets the flag), "classifier" (the hole
    trigger, `hole_trigger.HoleTrigger`, on the env's `trigger` observation group, evaluated in the search phase at
    the start of every step) or "oracle" (true peg tip within `trigger_r` [m] of the axis). After a local phase ends
    the in-env triggers wait `trigger_cool` steps (`trigger_enabled = False` switches them off for an episode).

    With `k_yaw_insert` set, the insert phase makes yaw compliant like roll/pitch: the yaw spring drops to
    `k_yaw_insert` (damping `d_yaw_insert`) and the measured yaw torque acts, so the hole walls can turn a peg that
    entered slightly yawed instead of jamming it (restored at reset).

    Action (3): [a_x, a_y, a_yaw] in [-1, 1].
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Read the parameters (spiral ones as `spiral_residual`) and allocate buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        p = cfg.params
        base = ActionCfg(term=cfg.term, ema_factor=cfg.ema_factor,
                         params={**p, "try_action": False, "speed_action": False, "lateral_action": True,
                                 "res_max": 0.0})
        super().__init__(base, ctx)
        self.dim = 3
        self.local_max = float(p.get("local_max", 0.0015))
        self.yaw_max = math.radians(float(p.get("yaw_max_deg", 12.0)))
        self.t_local = float(p.get("t_local", 3.0))
        self.f_local = float(p.get("f_local", self.f_search))
        self.start_local = bool(p.get("start_local", False))
        self.trigger_mode = str(p.get("trigger", "external"))
        if self.trigger_mode not in ("external", "classifier", "oracle"):
            raise ValueError(f"trigger must be external | classifier | oracle, got '{self.trigger_mode}'")
        self.trigger_cool = int(p.get("trigger_cool", 25))
        self.trigger_r = float(p.get("trigger_r", 0.001))
        self._since = 10**6
        self.trigger_enabled = True
        self._hole_trigger = None
        self.k_yaw_insert = p.get("k_yaw_insert")
        self.d_yaw_insert = float(p.get("d_yaw_insert", 0.4))
        c = ctx.controller
        self._yaw_hold = (float(c.k_rot[2]), float(c.d_rot[2]), float(c.ext_rot_mask[2]))
        if self.trigger_mode == "classifier":
            from compliant_RL.hole_trigger import HoleTrigger

            self._hole_trigger = HoleTrigger()
        self._loc = np.zeros(3)
        self._zero2 = np.zeros(2)
        self._anchor = np.zeros(3)
        self._yaw_cmd = 0.0
        self.state.update(trigger=False, local_t=0.0, n_local=0, yaw_cmd=0.0)

    def apply(self, a: np.ndarray) -> None:
        """Spiral (phases 0-2, as `spiral_residual` with zero residual) or one local-phase step.

        Args:
            a: Clipped action, shape (3,).
        """
        st, R = self.state, self._R
        if self.start_local and st["phase"] == 1 and st["n_local"] == 0:
            st["trigger"] = True
        self._since += 1
        if st["phase"] == 1 and self.trigger_mode != "external" and self.trigger_enabled and \
                self._since > self.trigger_cool:
            if self._hole_trigger is not None:
                st["trigger"] |= self._hole_trigger(self.ctx.obs_mgr.compute("trigger"))
            else:
                fs = _st(self.ctx)
                tip = self.ctx.data.site_xpos[fs.peg_tip]
                st["trigger"] |= math.hypot(tip[0] - fs.hole_tip[0], tip[1] - fs.hole_tip[1]) < self.trigger_r
        if st["phase"] == 1 and st["trigger"]:
            self._yaw_soft(False)
            st["phase"], st["local_t"] = 3, 0.0
            st["n_local"] += 1
            np.copyto(self._anchor, self._p0 + R[:, :2] @ st["offset"])
            self._loc[:2] = 0.0
            self._loc[2] = self._yaw_cmd / self.yaw_max
            self._gains(self.search_gains)
        st["trigger"] = False
        if st["phase"] != 3:
            super().apply(self._zero2)
            self._yaw_soft(st["phase"] == 2)
            return
        c = self.ctx.controller
        ee = self.ctx.plant.ee_pos
        z = float(R[:, 2] @ ee)
        st["local_t"] += self.ctx.policy_dt
        st["depth"] = z - self._z_touch
        if z - self._z_touch > self.drop_th:
            st["phase"] = 2
            np.copyto(self._hold, ee)
            self._gains(self.task_gains)
            self._yaw_soft(True)
            c.set_target(pos=self._hold, f_push=self.f_insert)
            return
        if st["local_t"] >= self.t_local:
            st["phase"] = 1
            self._since = 0
            super().apply(self._zero2)
            return
        alpha = float(self._ema[0])
        self._loc += alpha * (a - self._loc)
        np.dot(R[:, :2], self._loc[:2] * self.local_max, out=self._p)
        self._p += self._anchor
        self._yaw_cmd = st["yaw_cmd"] = float(self._loc[2]) * self.yaw_max
        c.set_target(pos=self._p, f_push=self.f_local, yaw=self._yaw_cmd)

    def _yaw_soft(self, on: bool) -> None:
        """Compliant yaw (insert phase, if `k_yaw_insert` is set) or the controller's held yaw."""
        if self.k_yaw_insert is None:
            return
        c = self.ctx.controller
        if on:
            c.k_rot[2], c.d_rot[2], c.ext_rot_mask[2] = float(self.k_yaw_insert), self.d_yaw_insert, 1.0
        else:
            c.k_rot[2], c.d_rot[2], c.ext_rot_mask[2] = self._yaw_hold

    def reset(self) -> None:
        """Start the approach (as `spiral_residual`) with no yaw command."""
        self._yaw_soft(False)
        super().reset()
        self._loc.fill(0.0)
        self._yaw_cmd = 0.0
        self._since = 10**6
        if self._hole_trigger is not None:
            self._hole_trigger.reset()
        self.state.update(trigger=False, local_t=0.0, n_local=0, yaw_cmd=0.0)
        self.ctx.controller.set_target(yaw=0.0)

    def write_applied(self, out: np.ndarray) -> None:
        """Applied (smoothed) local action.

        Args:
            out: Buffer of shape (3,).
        """
        np.copyto(out, self._loc)

    def _episode_info(self, ctx: "Context") -> dict:
        """Spiral metrics plus the number of local phases."""
        return {**super()._episode_info(ctx), "n_local": float(self.state["n_local"])}


# ---------------------------------------------------------------------------------- events
@event_term("cr_randomize_grasp")
def cr_randomize_grasp(ctx: "Context", tilt_max_deg: float = 3.0) -> None:
    """Reset: tilt the peg in the hand about the hand x and y axes, U(±tilt_max_deg) each, about the TCP.

    Models a grasp that holds the peg at an angle (unobserved by the actor; the critic sees it). The
    rotation is about the TCP (between the fingers), so the tip also moves sideways by about
    tcp_to_tip * angle. Writes `ctx.state["grasp_tilt"]` [rad], shape (2,).

    Args:
        ctx: Context.
        tilt_max_deg: Half-range per axis [deg] (0 = no tilt).
    """
    st = _st(ctx)
    m = ctx.model
    tilt = ctx.buffer("grasp_tilt", 2)
    if "cr_grasp_nominal" not in ctx.state:
        tcp = ctx.handles.tcp_site_id
        if m.site_bodyid[tcp] != m.body_parentid[st.peg_body]:
            raise ValueError("cr_randomize_grasp needs the TCP site and the peg on the same (hand) body")
        ctx.state["cr_grasp_nominal"] = (m.body_quat[st.peg_body].copy(), m.site_pos[tcp].copy())
    quat0, tcp_local = ctx.state["cr_grasp_nominal"]
    lim = math.radians(tilt_max_deg)
    tilt[:] = ctx.rng.uniform(-lim, lim, size=2) if lim > 0.0 else 0.0
    rv = np.array([tilt[0], tilt[1], 0.0])
    ang = float(np.linalg.norm(rv))
    qt = np.array([1.0, 0.0, 0.0, 0.0])
    if ang > 1e-12:
        mujoco.mju_axisAngle2Quat(qt, rv / ang, ang)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, qt)
    R = R.reshape(3, 3)
    m.body_pos[st.peg_body] = tcp_local + R @ (st.peg_pos_nominal - tcp_local)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qt, quat0)
    m.body_quat[st.peg_body] = q


@event_term("cr_reset_ee")
def cr_reset_ee(ctx: "Context", hover: float = 0.004, xy_jitter: float = 0.0005, tip_clearance: float = 0.001,
                ik_tol: float = 1e-3) -> None:
    """Reset: gripper down, TCP placed so the (nominal) peg tip is `hover` above the estimated hole opening.

    This is where the residual policy takes over from an upstream approach that used the hole estimate.
    The actual tip is offset by the estimate error and the grasp tilt. A small xy jitter models
    approach inaccuracy. The height is raised if needed so the actual tip starts at least
    `tip_clearance` above the true hole top. Must run after `forge_reset_fixed` and `cr_randomize_grasp`.

    Args:
        ctx: Context.
        hover: Nominal peg-tip height above the estimated opening [m].
        xy_jitter: Half-range of an extra uniform lateral start offset [m].
        tip_clearance: Minimum actual tip height above the true hole top [m].
        ik_tol: IK error above which the worst error is recorded in the episode metrics [m].
    """
    st = _st(ctx)
    target = ctx.state[ANCHOR].copy()                  # TCP pose with the nominal tip at the estimated opening
    target[2] += hover
    target[:2] += ctx.rng.uniform(-xy_jitter, xy_jitter, size=2)
    target[2] = max(target[2], st.hole_tip[2] + st.tcp_to_tip + tip_clearance)
    _, err = solve_tcp_ik(ctx.plant, target, st.home_quat, q_init=ctx.handles.q_home)
    st.ik_err_max = err if err > ik_tol else 0.0


@event_term("cr_align_hole_yaw")
def cr_align_hole_yaw(ctx: "Context") -> None:
    """Startup: rotate the socket about the world z axis to the TCP yaw at home (yaw is held by the controller).

    For non-round parts (`BoxPeg` / `RectHole`): the peg's cross-section axes are the TCP x, y axes, so the hole
    is turned to match them. Sets the mocap body's default orientation, which `mj_resetData` restores every reset.
    Must run after `forge_init`.

    Args:
        ctx: Context.
    """
    st = _st(ctx)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, st.home_quat)
    yaw = math.atan2(R[3], R[0])                      # TCP x axis in the world xy plane
    q = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
    hole_body = int(np.flatnonzero(ctx.model.body_mocapid == st.hole_mocap)[0])
    ctx.model.body_quat[hole_body] = q
    ctx.data.mocap_quat[st.hole_mocap] = q
    ctx.state["hole_yaw"] = yaw


@event_term("cr_hole_yaw_error")
def cr_hole_yaw_error(ctx: "Context", lo_deg: float = 0.0, hi_deg: float = 10.0, sym_deg: float = 90.0,
                      bins: tuple = ()) -> None:
    """Reset: turn the socket about the world z axis by U(lo_deg, hi_deg), random sign, from its aligned yaw.

    With `bins` ((lo, hi), ...) [deg] a bin is picked uniformly first, then U(lo, hi) in it (e.g. half the episodes
    in the range where the peg is blocked).

    The aligned yaw is the TCP yaw at home (`cr_align_hole_yaw`, startup; 0 without it). Must run after the socket
    reset (`forge_reset_fixed`). Writes `ctx.state["hole_yaw_err"]` [rad] and the cross-section symmetry
    `ctx.state["yaw_sym"]` [rad] used by `hole_yaw_error`.

    Args:
        ctx: Context.
        lo_deg: Min yaw error magnitude [deg].
        hi_deg: Max yaw error magnitude [deg].
        sym_deg: Rotational symmetry of the cross-section [deg] (90 square, 60 hexagon, 45 octagon).
        bins: Optional yaw magnitude bins [deg], replacing (lo_deg, hi_deg).
    """
    ctx.state["yaw_sym"] = math.radians(sym_deg)
    st = _st(ctx)
    if bins:
        lo_deg, hi_deg = bins[int(ctx.rng.integers(len(bins)))]
    err = math.radians(float(ctx.rng.uniform(lo_deg, hi_deg))) * (1.0 if ctx.rng.uniform() < 0.5 else -1.0)
    yaw = float(ctx.state.get("hole_yaw", 0.0)) + err
    ctx.data.mocap_quat[st.hole_mocap] = (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))
    ctx.state["hole_yaw_err"] = err


@event_term("cr_s3_start")
def cr_s3_start(ctx: "Context", p_local: float = 0.5, local_offset_max: float = 0.0018,
                spiral_offset_max: float = 0.004, local_episode_s: float = 6.0) -> None:
    """Reset: per episode either a synthetic local-phase start or the full spiral with the in-env hole trigger.

    Local start (probability `p_local`): tip U(0, local_offset_max) off the hole axis, the local phase starts at
    touchdown, no trigger, the episode is cut after `local_episode_s` (`cr_s3_local_timeout`). Otherwise: hole
    estimate U(0, spiral_offset_max) off, spiral + trigger as `compliant_peg_s3_handoff`. Replaces `cr_shift_ee`
    (same position, after `cr_reset_ee`); needs `spiral_local` with `trigger="classifier"`. Episode metric:
    `local_start` (1 / 0).

    Args:
        ctx: Context.
        p_local: Probability of a synthetic local start.
        local_offset_max: Max start offset of a local start [m].
        spiral_offset_max: Max hole-estimate error of a spiral episode [m].
        local_episode_s: Length of a local-start episode [s].
    """
    term = ctx.state["spiral_term"]
    local = bool(ctx.rng.uniform() < p_local)
    term.start_local, term.trigger_enabled = local, not local
    if "s3_start" not in ctx.state:
        ctx.episode_info_hooks.append(lambda c: {"local_start": 1.0 if c.state["s3_start"]["local"] else 0.0})
    ctx.state["s3_start"] = {"local": local, "t_end": local_episode_s}
    cr_shift_ee(ctx, offset_min=0.0, offset_max=local_offset_max if local else spiral_offset_max)


@termination_term("cr_s3_local_timeout")
def cr_s3_local_timeout(ctx: "Context") -> bool:
    """True once a synthetic local-start episode (`cr_s3_start`) has run its length (use as a time-out)."""
    s = ctx.state.get("s3_start")
    return bool(s and s["local"] and ctx.episode_step * ctx.policy_dt >= s["t_end"] - 1e-9)


@event_term("cr_shift_ee")
def cr_shift_ee(ctx: "Context", offset_min: float = 0.0, offset_max: float = 0.002, ik_tol: float = 1e-3) -> None:
    """Reset: shift the gripper sideways (world xy) by U(offset_min, offset_max) in a random direction.

    Orientation and height are kept. Must run after `cr_reset_ee`. Writes `ctx.state["start_shift"]`
    [m], shape (2,).

    Args:
        ctx: Context.
        offset_min: Min shift magnitude [m].
        offset_max: Max shift magnitude [m] (0 = no shift).
        ik_tol: IK error above which the worst error is recorded in the episode metrics [m].
    """
    st = _st(ctx)
    shift = ctx.buffer("start_shift", 2)
    r = float(ctx.rng.uniform(offset_min, offset_max))
    phi = float(ctx.rng.uniform(0.0, 2.0 * math.pi))
    shift[:] = r * math.cos(phi), r * math.sin(phi)
    if r < 1e-12:
        return
    mujoco.mj_kinematics(ctx.model, ctx.data)
    target = ctx.plant.ee_pos.copy()
    target[:2] += shift
    _, err = solve_tcp_ik(ctx.plant, target, ctx.plant.ee_quat().copy(), q_init=ctx.plant.q)
    st.ik_err_max = max(st.ik_err_max, err if err > ik_tol else 0.0)


@event_term("cr_curriculum_start")
def cr_curriculum_start(ctx: "Context", stages: tuple = ((0.002, 6.0, 0.0026), (0.003, 7.0, 0.0036), (0.004, 8.0, 0.0046)),
                        advance_at: tuple = (0.92, 0.85), window: int = 20) -> None:
    """Reset: start pose for the current curriculum stage (replaces `cr_tilt_ee` + `cr_shift_ee`).

    Stage k = (offset_max [m], tilt_max [deg], spiral r_max [m]) or (offset_min, offset_max, tilt_max, r_max):
    tilt U(0, tilt_max) about the peg tip, then a lateral shift U(offset_min, offset_max) (offset_min = 0 for
    3-tuples) in a random direction, and the spiral's r_max. Per env: the stage advances
    when the success rate (success at any step) over the last `window` episodes reaches advance_at[k]; the
    window then restarts. Episode metrics: `curriculum_stage`. Must run after `cr_reset_ee`; needs the
    `spiral_residual` action term.

    Args:
        ctx: Context.
        stages: Curriculum stages (offset_max, tilt_max_deg, r_max).
        advance_at: Success rate needed to leave stage k (one entry per stage but the last).
        window: Episodes per env in the success window.
    """
    cur = ctx.state.get("curriculum")
    if cur is None:
        cur = ctx.state["curriculum"] = {"stage": 0, "hist": []}

        def hook(c: "Context") -> dict:
            cur["hist"].append(1.0 if _st(c).first_success_step >= 0 else 0.0)
            k = cur["stage"]
            if k < len(advance_at) and len(cur["hist"]) >= window and \
                    float(np.mean(cur["hist"][-window:])) >= advance_at[k]:
                cur["stage"], cur["hist"] = k + 1, []
            return {"curriculum_stage": float(k)}

        ctx.episode_info_hooks.append(hook)
    stage = stages[cur["stage"]]
    offset_min, offset_max, tilt_max, r_max = stage if len(stage) == 4 else (0.0, *stage)
    cr_tilt_ee(ctx, tilt_max_deg=tilt_max)
    cr_shift_ee(ctx, offset_min=offset_min, offset_max=offset_max)
    ctx.state["spiral_term"].r_max = r_max


@event_term("cr_tilt_ee")
def cr_tilt_ee(ctx: "Context", tilt_max_deg: float = 10.0, ik_tol: float = 1e-3, tilt_min_deg: float = 0.0) -> None:
    """Reset: tilt the gripper about the peg tip by a random angle in [tilt_min_deg, tilt_max_deg], random direction.

    The tilt axis is horizontal (a world x/y rotation, no yaw) and passes through the peg tip, so the tip
    keeps its xy position and height. Must run after `cr_reset_ee`. The controller then takes the tilted
    pose as its reference. Writes `ctx.state["start_tilt"]` [rad], shape (2,) (rotation vector x, y).

    Args:
        ctx: Context.
        tilt_max_deg: Max tilt magnitude [deg] (0 = no tilt).
        ik_tol: IK error above which the worst error is recorded in the episode metrics [m].
        tilt_min_deg: Min tilt magnitude [deg] (= tilt_max_deg for a fixed tilt).
    """
    st = _st(ctx)
    m, d = ctx.model, ctx.data
    tilt = ctx.buffer("start_tilt", 2)
    ang = math.radians(tilt_min_deg + (tilt_max_deg - tilt_min_deg) * float(ctx.rng.uniform()))
    phi = float(ctx.rng.uniform(0.0, 2.0 * math.pi))
    tilt[:] = ang * math.cos(phi), ang * math.sin(phi)
    if ang < 1e-12:
        return
    mujoco.mj_kinematics(m, d)
    tip = d.site_xpos[st.peg_tip].copy()
    tcp = ctx.plant.ee_pos.copy()
    qt = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qt, np.array([math.cos(phi), math.sin(phi), 0.0]), ang)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, qt)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, qt, ctx.plant.ee_quat())
    _, err = solve_tcp_ik(ctx.plant, tip + R.reshape(3, 3) @ (tcp - tip), quat, q_init=ctx.plant.q)
    st.ik_err_max = max(st.ik_err_max, err if err > ik_tol else 0.0)


# ---------------------------------------------------------------------------------- observations
def _tcp_R(ctx: "Context") -> np.ndarray:
    """TCP rotation (TCP -> world), shape (3, 3) (a view of site_xmat).

    Args:
        ctx: Context.

    Returns:
        Rotation matrix view.
    """
    return ctx.plant.ee_rot.reshape(3, 3)


@obs_term("cr_hole_rel_tcp", dim=3)
def cr_hole_rel_tcp(ctx: "Context", out: np.ndarray) -> None:
    """Estimated hole opening relative to the TCP, in the TCP frame [m], shape (3,) (noisy estimate)."""
    st = _st(ctx)
    est = ctx.state[ANCHOR] - np.array([0.0, 0.0, st.tcp_to_tip])
    np.dot(_tcp_R(ctx).T, est - ctx.plant.ee_pos, out=out)


@obs_term("cr_hole_axis_tcp", dim=3)
def cr_hole_axis_tcp(ctx: "Context", out: np.ndarray) -> None:
    """Hole axis (world up, the socket is upright) in the TCP frame, shape (3,): the TCP tilt."""
    np.copyto(out, _tcp_R(ctx)[2, :])  # R^T e_z = third row of R


def peg_wrench_world(ctx: "Context", out: np.ndarray) -> None:
    """Contact wrench the socket applies to the peg, world axes: [force (3) N, torque about the TCP (3) Nm].

    From `cfrc_ext[peg]` = [torque about c, force] with c = subtree_com of the peg's root body:
    torque about the TCP = torque_c + (c - p_tcp) x force. The peg only touches the socket, so this is
    what a wrist F/T sensor would read after removing gravity and inertia.

    Args:
        ctx: Context.
        out: Output, shape (6,).
    """
    d = ctx.data
    st = _st(ctx)
    ext = d.cfrc_ext[st.peg_body]
    c = d.subtree_com[ctx.model.body_rootid[st.peg_body]]
    p = ctx.plant.ee_pos
    fx, fy, fz = ext[3], ext[4], ext[5]
    ax, ay, az = c[0] - p[0], c[1] - p[1], c[2] - p[2]
    out[0], out[1], out[2] = fx, fy, fz
    out[3] = ext[0] + ay * fz - az * fy
    out[4] = ext[1] + az * fx - ax * fz
    out[5] = ext[2] + ax * fy - ay * fx


@event_term("cr_init")
def cr_init(ctx: "Context") -> None:
    """Startup: give the axis compliance controller the peg contact wrench as its force measurement.

    Must run after `forge_init`.

    Args:
        ctx: Context.
    """
    if not hasattr(ctx.controller, "set_wrench_source"):
        raise ValueError("compliant_peg needs the AxisCompliance controller")
    ctx.controller.set_wrench_source(lambda out: peg_wrench_world(ctx, out))


@obs_term("cr_wrench_tcp", dim=6, needs_accumulation=True)
def cr_wrench_tcp(ctx: "Context", out: np.ndarray) -> None:
    """Contact wrench on the peg, about the TCP, in the TCP frame [N, Nm], shape (6,), step-averaged."""
    w = ctx.state.get("cr_wrench_buf")
    if w is None:
        w = ctx.state["cr_wrench_buf"] = np.zeros(6)
    peg_wrench_world(ctx, w)
    R = ctx.plant.ee_rot  # row-major: R^T v uses the columns of R
    fx, fy, fz, tx, ty, tz = w
    out[0] = R[0] * fx + R[3] * fy + R[6] * fz
    out[1] = R[1] * fx + R[4] * fy + R[7] * fz
    out[2] = R[2] * fx + R[5] * fy + R[8] * fz
    out[3] = R[0] * tx + R[3] * ty + R[6] * tz
    out[4] = R[1] * tx + R[4] * ty + R[7] * tz
    out[5] = R[2] * tx + R[5] * ty + R[8] * tz


def contact_wrench_tcp_reference(ctx: "Context") -> np.ndarray:
    """Slow reference for `cr_wrench_tcp` (sums the peg's contacts one by one); for tests.

    Args:
        ctx: Context.

    Returns:
        [force, torque about the TCP] on the peg in the TCP frame, shape (6,).
    """
    st = _st(ctx)
    m, d = ctx.model, ctx.data
    w, f6 = np.zeros(6), np.zeros(6)
    tcp = ctx.plant.ee_pos
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if st.peg_body not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]   # force of geom1 on geom2, world frame
        if b1 == st.peg_body:
            fw = -fw
        w[:3] += fw
        w[3:] += np.cross(c.pos - tcp, fw)
    R = _tcp_R(ctx)
    return np.concatenate([R.T @ w[:3], R.T @ w[3:]])


@obs_term("cr_twist_tcp", dim=6)
def cr_twist_tcp(ctx: "Context", out: np.ndarray) -> None:
    """TCP twist in the TCP frame [m/s, rad/s], shape (6,)."""
    v = ctx.plant.ee_vel()
    R = _tcp_R(ctx)
    np.dot(R.T, v[:3], out=out[:3])
    np.dot(R.T, v[3:], out=out[3:])


@obs_term("cr_spring_defl", dim=2)
def cr_spring_defl(ctx: "Context", out: np.ndarray) -> None:
    """Lateral spring deflection of the virtual frame from its anchor, command frame [m], shape (2,).

    k_lat times this is the lateral spring force (sign: virtual frame minus anchor).
    """
    c = ctx.controller
    np.dot(c.R_f[:, :2].T, c.x_c - c.p_ref, out=out)


@obs_term("sp_base_offset", dim=2)
def sp_base_offset(ctx: "Context", out: np.ndarray) -> None:
    """Spiral base offset of the anchor from the start position, command frame [m], shape (2,)."""
    np.copyto(out, ctx.state["spiral"]["offset"])


@obs_term("sp_depth", dim=1)
def sp_depth(ctx: "Context", out: np.ndarray) -> None:
    """TCP displacement along the push axis since touchdown [m] (0 before touchdown), shape (1,)."""
    out[0] = ctx.state["spiral"]["depth"]


@obs_term("sp_phase", dim=1)
def sp_phase(ctx: "Context", out: np.ndarray) -> None:
    """Spiral phase / 2: 0 approach, 0.5 search, 1 insert, shape (1,)."""
    out[0] = 0.5 * ctx.state["spiral"]["phase"]


@obs_term("sp_tcp_lat_vel", dim=2)
def sp_tcp_lat_vel(ctx: "Context", out: np.ndarray) -> None:
    """TCP lateral velocity in the command frame [m/s], shape (2,) (from joint encoders)."""
    np.dot(ctx.controller.R_f[:, :2].T, ctx.plant.ee_vel()[:3], out=out)


@obs_term("sp_tcp_tilt", dim=2)
def sp_tcp_tilt(ctx: "Context", out: np.ndarray) -> None:
    """TCP roll/pitch relative to the command frame (start TCP frame) [rad], small-angle, shape (2,)."""
    Rr = ctx.controller.R_f.T @ _tcp_R(ctx)
    out[0] = 0.5 * (Rr[2, 1] - Rr[1, 2])
    out[1] = 0.5 * (Rr[0, 2] - Rr[2, 0])


@obs_term("sp_try_state", dim=2)
def sp_try_state(ctx: "Context", out: np.ndarray) -> None:
    """Applied try-insert level tau and time since the current try started [s], shape (2,)."""
    sp = ctx.state["spiral"]
    out[0], out[1] = sp["tau"], sp["try_t"]


@obs_term("episode_progress", dim=1)
def episode_progress(ctx: "Context", out: np.ndarray) -> None:
    """Privileged (critic): elapsed fraction of the episode, shape (1,)."""
    out[0] = ctx.episode_step / ctx.max_episode_steps


@obs_term("cr_peg_axis_gt", dim=3)
def cr_peg_axis_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: peg axis (top -> tip) in the world frame, shape (3,); (0, 0, -1) = aligned with the hole."""
    st = _st(ctx)
    np.copyto(out, ctx.data.xmat[st.peg_body].reshape(3, 3)[:, 2])


@obs_term("cr_grasp_tilt_gt", dim=2)
def cr_grasp_tilt_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: peg tilt in the grasp about the hand x, y axes [rad], shape (2,)."""
    np.copyto(out, ctx.buffer("grasp_tilt", 2))


def hole_yaw_error(ctx: "Context") -> float:
    """Hole yaw minus peg yaw about world z [rad], wrapped to ± half the cross-section symmetry.

    The symmetry comes from `cr_hole_yaw_error` (`ctx.state["yaw_sym"]`, default 90°, a square).

    Args:
        ctx: Context.

    Returns:
        Yaw error [rad].
    """
    ids = ctx.state.get("cr_yaw_ids")
    if ids is None:
        m = ctx.model
        ids = ctx.state["cr_yaw_ids"] = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom"),
                                         int(np.flatnonzero(m.body_mocapid == _st(ctx).hole_mocap)[0]))
    d = ctx.data
    pg, hb = d.geom_xmat[ids[0]], d.xmat[ids[1]]
    sym = float(ctx.state.get("yaw_sym", 0.5 * math.pi))
    dphi = math.atan2(hb[3], hb[0]) - math.atan2(pg[3], pg[0])
    return (dphi + 0.5 * sym) % sym - 0.5 * sym


@obs_term("sp_local_offset", dim=2)
def sp_local_offset(ctx: "Context", out: np.ndarray) -> None:
    """TCP lateral position relative to the local-phase freeze point, command frame [m], shape (2,) (0 before the
    first local phase; needs `spiral_local`)."""
    t = ctx.state["spiral_term"]
    if ctx.state["spiral"]["n_local"] == 0:
        out[:] = 0.0
        return
    np.dot(t._R[:, :2].T, ctx.plant.ee_pos - t._anchor, out=out)


@obs_term("sp_tcp_yaw", dim=1)
def sp_tcp_yaw(ctx: "Context", out: np.ndarray) -> None:
    """TCP yaw about the tool axis relative to the command frame (start TCP frame) [rad], small-angle, shape (1,)."""
    Rr = ctx.controller.R_f.T @ _tcp_R(ctx)
    out[0] = 0.5 * (Rr[1, 0] - Rr[0, 1])


@obs_term("sp_local_time", dim=1)
def sp_local_time(ctx: "Context", out: np.ndarray) -> None:
    """Time in the current local phase [s], shape (1,) (0 outside it; needs `spiral_local`)."""
    sp = ctx.state["spiral"]
    out[0] = sp["local_t"] if sp["phase"] == 3 else 0.0


@obs_term("cr_yaw_err_gt", dim=1)
def cr_yaw_err_gt(ctx: "Context", out: np.ndarray) -> None:
    """True hole-minus-peg yaw error (`hole_yaw_error`) [rad], shape (1,) (privileged)."""
    out[0] = hole_yaw_error(ctx)


# ---------------------------------------------------------------------------------- rewards
def peg_axis_angle(ctx: "Context") -> float:
    """Angle between the peg axis and the hole axis [rad].

    Args:
        ctx: Context.

    Returns:
        Angle [rad].
    """
    z = ctx.data.xmat[_st(ctx).peg_body][8]  # world z component of the peg's body z axis (top -> tip)
    return math.acos(max(-1.0, min(1.0, -z)))


@reward_term("cr_xy_aligned")
def cr_xy_aligned(ctx: "Context", tol: float = 0.0005) -> float:
    """1 while the peg tip is within `tol` [m] of the hole axis (laterally)."""
    return 1.0 if _st(ctx).xy_dist < tol else 0.0


@reward_term("sp_try_cost")
def sp_try_cost(ctx: "Context") -> float:
    """Applied try-insert level tau in [0, 1] (use a negative weight)."""
    return ctx.state["spiral"]["tau"]


@reward_term("sp_try_action_cost")
def sp_try_action_cost(ctx: "Context") -> float:
    """(a + 1)^2 / 4 in [0, 1] of the raw try action a (use a negative weight).

    Unlike `sp_try_cost` it is not flat below the try dead zone, so outputs that do nothing still cost a little and
    the policy is pushed to -1 where a try does not help (instead of drifting up to the dead-zone edge).
    """
    return 0.25 * (ctx.state["spiral"].get("a_try", -1.0) + 1.0) ** 2


@reward_term("cr_yaw_align")
def cr_yaw_align(ctx: "Context", a: float = 20.0, b: float = 0.0) -> float:
    """Yaw alignment kernel K_{a,b}(|hole_yaw_error|) (a in 1/rad). The keypoint kernels lie on the peg axis and
    do not see yaw."""
    return logistic_kernel(abs(hole_yaw_error(ctx)), a, b)


@reward_term("cr_kp_xy")
def cr_kp_xy(ctx: "Context", a: float = 1000.0, b: float = 0.0) -> float:
    """Lateral alignment kernel K_{a,b}(xy distance of the peg tip from the hole axis)."""
    return logistic_kernel(_st(ctx).xy_dist, a, b)


@reward_term("cr_place_bonus")
def cr_place_bonus(ctx: "Context", xy_tol: float = 0.0025, depth: float = 0.001) -> float:
    """1 while the peg tip is within `xy_tol` of the hole axis and at least `depth` [m] below the opening.

    Unlike `forge_place_bonus` (tip just below the opening), the depth margin keeps contact penetration
    on the socket top from counting as placed.
    """
    st = _st(ctx)
    tip_z = ctx.data.site_xpos[st.peg_tip][2]
    return 1.0 if st.xy_dist < xy_tol and tip_z < st.hole_tip[2] - depth else 0.0


@reward_term("cr_axis_aligned")
def cr_axis_aligned(ctx: "Context", tol_deg: float = 1.0) -> float:
    """1 while the peg axis is within `tol_deg` of the hole axis."""
    return 1.0 if peg_axis_angle(ctx) < math.radians(tol_deg) else 0.0
