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
                  the TCP position at the drop

    Residual: anchor += R_f [s[0:2] * res_max, 0] (all phases); spiral speed scale 1 + s[2] for s[2] <= 0 and
    1 + s[2] (speed_max - 1) above, so [0, speed_max] with 1 at zero residual (0 = hold the spiral). With
    speed_action=False the action is the lateral residual only (a in [-1, 1]^2) and the spiral runs at v.
    Episode metrics (logging only): mean |residual| x, y [mm] and the mean residual component toward the true
    hole [mm] (positive = steering toward it), in the search and insert phases. Writes `ctx.state["spiral"]` (phase 0/1/2, base offset (2) [m], depth since
    touchdown [m]) for the observations. `cfg.params`: f_search (1 N), f_insert (5 N), v (10 mm/s),
    pitch (0.8 mm), r_max (2.6 mm), drop_th (0.5 mm), res_max (1 mm), speed_action (True), speed_max (2), ks_lat (2000),
    ds_lat (190), ks_tilt (3), ds_tilt (1.5).
    """

    def __init__(self, cfg: ActionCfg, ctx: "Context") -> None:
        """Read the parameters and allocate buffers.

        Args:
            cfg: Action configuration.
            ctx: Shared context.
        """
        super().__init__(cfg, ctx)
        p = cfg.params
        self.speed_action = bool(p.get("speed_action", True))
        self.dim = 3 if self.speed_action else 2
        self.f_search = float(p.get("f_search", 1.0))
        self.f_insert = float(p.get("f_insert", 5.0))
        self.v = float(p.get("v", 0.01))
        self.b = float(p.get("pitch", 0.0008)) / (2.0 * math.pi)
        self.r_max = float(p.get("r_max", 0.0026))
        self.drop_th = float(p.get("drop_th", 0.0005))
        self.res_max = float(p.get("res_max", 0.001))
        self.speed_up = float(p.get("speed_max", 2.0)) - 1.0
        self.search_gains = (float(p.get("ks_lat", 2000.0)), float(p.get("ds_lat", 190.0)),
                             float(p.get("ks_tilt", 3.0)), float(p.get("ds_tilt", 1.5)))
        c = ctx.controller
        self.task_gains = (c.k_lin[0], c.d_lin[0], c.k_rot[0], c.d_rot[0])
        self._ema = ctx.buffer("action_ema", 1)
        self._ema[0] = float(cfg.ema_factor)
        ctx.buffer("action_max_step", 3)  # read by ForgeState (λ is not used by this controller)
        self.state = ctx.state["spiral"] = {"phase": 0, "offset": np.zeros(2), "depth": 0.0}
        ctx.state["spiral_term"] = self
        self._s = np.zeros(self.dim)
        self._tmp = np.zeros(self.dim)
        self._m = np.zeros((2, 4))           # per phase (search, insert): sum |rx|, sum |ry|, sum toward-hole, n
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
            a: Clipped action, shape (3,).
        """
        alpha = float(self._ema[0])
        np.multiply(a, alpha, out=self._tmp)
        self._s *= 1.0 - alpha
        self._s += self._tmp
        s, c, st, R = self._s, self.ctx.controller, self.state, self._R
        ee = self.ctx.plant.ee_pos
        dt = self.ctx.policy_dt
        z = float(R[:, 2] @ ee)
        if st["phase"] == 0:
            peg_wrench_world(self.ctx, self._w)
            if -(R[:, 2] @ self._w[:3]) > 0.5 * self.f_search:
                st["phase"], self._z_touch = 1, z
        if st["phase"] == 1:
            if z - self._z_touch > self.drop_th:
                st["phase"] = 2
                np.copyto(self._hold, ee)
                self._gains(self.task_gains)
            else:
                r = self.b * self._theta
                if self._out and r >= self.r_max:
                    self._out = False
                elif not self._out and r <= 0.0:
                    self._out = True
                speed = 1.0 if not self.speed_action else 1.0 + (s[2] if s[2] <= 0.0 else s[2] * self.speed_up)
                dth = speed * self.v * dt / math.sqrt(r * r + self.b * self.b)
                self._theta = max(0.0, self._theta + (dth if self._out else -dth))
                r = self.b * self._theta
                st["offset"][:] = r * math.cos(self._theta), r * math.sin(self._theta)
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
        np.dot(R[:, :2], s[0:2] * self.res_max, out=self._p)
        self._p += base
        c.set_target(pos=self._p, f_push=self.f_insert if st["phase"] == 2 else self.f_search)

    def reset(self) -> None:
        """Start the approach: search gains, anchor at the start position, push f_search."""
        self._s.fill(0.0)
        self._m.fill(0.0)
        c, st = self.ctx.controller, self.state
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
        np.copyto(out, self._s)

    def _episode_info(self, ctx: "Context") -> dict:
        """Residual usage over the episode [mm] (logging only)."""
        out = {}
        for i, ph in enumerate(("search", "insert")):
            m = self._m[i]
            k = max(m[3], 1.0)
            out[f"res_{ph}_absx_mm"] = 1e3 * m[0] / k
            out[f"res_{ph}_absy_mm"] = 1e3 * m[1] / k
            out[f"res_{ph}_toward_hole_mm"] = 1e3 * m[2] / k
        return out


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
def cr_tilt_ee(ctx: "Context", tilt_max_deg: float = 10.0, ik_tol: float = 1e-3) -> None:
    """Reset: tilt the gripper about the peg tip by a random angle in [0, tilt_max_deg], random direction.

    The tilt axis is horizontal (a world x/y rotation, no yaw) and passes through the peg tip, so the tip
    keeps its xy position and height. Must run after `cr_reset_ee`. The controller then takes the tilted
    pose as its reference. Writes `ctx.state["start_tilt"]` [rad], shape (2,) (rotation vector x, y).

    Args:
        ctx: Context.
        tilt_max_deg: Max tilt magnitude [deg] (0 = no tilt).
        ik_tol: IK error above which the worst error is recorded in the episode metrics [m].
    """
    st = _st(ctx)
    m, d = ctx.model, ctx.data
    tilt = ctx.buffer("start_tilt", 2)
    ang = math.radians(tilt_max_deg) * float(ctx.rng.uniform())
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


@obs_term("cr_peg_axis_gt", dim=3)
def cr_peg_axis_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: peg axis (top -> tip) in the world frame, shape (3,); (0, 0, -1) = aligned with the hole."""
    st = _st(ctx)
    np.copyto(out, ctx.data.xmat[st.peg_body].reshape(3, 3)[:, 2])


@obs_term("cr_grasp_tilt_gt", dim=2)
def cr_grasp_tilt_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: peg tilt in the grasp about the hand x, y axes [rad], shape (2,)."""
    np.copyto(out, ctx.buffer("grasp_tilt", 2))


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
