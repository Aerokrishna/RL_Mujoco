"""FORGE-style insertion terms (Noseworthy et al., "FORGE", arXiv:2408.04587).

Where the paper is silent, values and mechanisms follow NVIDIA's reference implementation in
Isaac Lab (`isaaclab_tasks/direct/forge`, `.../factory`), referred to as "Isaac" below.

Implemented:
- keypoint reward: baseline (Isaac), coarse and fine logistic kernels (App. B, Eq. 8),
  K_{a,b}(x) = 1 / (e^{-ax} + b + e^{ax}),
- bonuses I_place + I_success (Eq. 2),
- excessive-force penalty -β max(0, ||F|| - F_th) (Eq. 3), with F_th randomized per
  episode and observed by the policy (π(a | o, F_th)),
- success prediction (Sec. III-C): action a_ET -> p in [0, 1] (`ctx.state["pred_success"]`),
  penalty -|p - y_t|, switched on only once the success rate reaches `delay_until_ratio`
  (Isaac: 0.25; before that an untrained predictor would cancel the success bonus); early
  termination when p > p_term (evaluation only) and the paper's early-termination metrics,
- Isaac action penalties: ||a_t - a_{t-1}|| on the smoothed action (core `applied_action_rate_norm`)
  and ||p_targ - p_ee|| / λ_nominal (`forge_action_penalty_asset`),
- dynamics randomization (Sec. III-B, Isaac sampling): per-axis Kp (x/÷ (1 + U(0, 0.41)) around 565 N/m
  and 28 Nm/rad, Kd = 2 sqrt(Kp)), per-axis λ (x/÷ (1 + U(0, 0.25)) around 2 cm), action EMA
  α ~ U(0.025, 0.1), wrench dead zone U(0, [5 N, 1 Nm]) per component (re-drawn every 2 s), part
  friction U(0.5, 1.0); all unobserved by the actor and given to the critic,
- initial state: fixed-part pose, hand pose (incl. yaw ±45°) relative to the fixed part, and the
  held part's offset in the hand (x, z ±3 mm, unobserved) (Table II / Isaac),
- observation noise (Isaac): see `task.py` (`ObsCfg.term_cfg`).

Shared per-step state lives in a `ForgeState` (in `ctx.state["forge"]`). It is
updated once per policy step by the `forge_update` step event, so the rewards,
terminations and metrics never recompute geometry.

Contact force: `data.cfrc_ext[peg]` is the external (contact) wrench on the peg body,
[torque, force] with world-frame orientation. It is filled by `mj_rnePostConstraint`,
which MuJoCo runs because the scene has force/torque sensors. The peg collides only
with the socket, so this is exactly the peg-socket contact force (no gravity or
inertia, unlike the wrist F/T sensor).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_rl_bed.env.managers.events import event_term
from mujoco_rl_bed.env.managers.observation import obs_term
from mujoco_rl_bed.env.managers.reward import reward_term
from mujoco_rl_bed.env.managers.termination import termination_term
from mujoco_rl_bed.sim.ik import solve_tcp_ik

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context

ANCHOR = "fixed_anchor"  # ctx.state buffer read by the `anchor_relative_pos` action term


def logistic_kernel(x: float, a: float, b: float) -> float:
    """Bounded distance kernel K_{a,b}(x) = 1 / (e^{-ax} + b + e^{ax}).

    Args:
        x: Distance [m] (>= 0).
        a: Sharpness [1/m].
        b: Offset (peak value is 1 / (2 + b)).

    Returns:
        Kernel value in (0, 1 / (2 + b)].
    """
    ax = min(a * x, 50.0)  # avoid overflow far from the goal
    return 1.0 / (math.exp(-ax) + b + math.exp(ax))


class ForgeState:
    """Resolved ids plus per-episode/per-step quantities for the peg-in-hole task.

    Attributes:
        peg_body: Peg body id.
        kp_sites: Peg keypoint site ids, shape (k,), kp0 = tip.
        kp_offsets: Keypoint distances from the tip along the peg axis [m], shape (k,).
        hole_mocap: Mocap index of the socket body.
        tip_local: Hole-tip site position in the socket body frame [m], shape (3,).
        floor_local: Hole-floor site position in the socket body frame [m], shape (3,).
        tcp_to_tip: Nominal distance from the TCP to the peg tip along the peg axis [m] (the policy's
            belief: used for the anchor, unaffected by the in-hand offset).
        tcp_to_tip_true: Actual TCP-to-tip distance along the axis this episode [m] (nominal + held offset z).
        peg_pos_nominal: Peg body position in the hand frame from the model [m], shape (3,).
        held_offset: Peg offset in the hand frame this episode [m], shape (3,) (unobserved by the actor).
        part_geoms: Geom ids of the peg and the socket (friction randomization).
        friction: Part sliding friction this episode.
        kp: Controller stiffness this episode [N/m (3), Nm/rad (3)], shape (6,).
        lam: λ per axis this episode [m] (the action term's `action_max_step` buffer), shape (3,).
        ema: Action smoothing α this episode (the action term's `action_ema` buffer), shape (1,).
        dead_zone: Wrench dead zone [N (3), Nm (3)], shape (6,) (re-drawn during the episode).
        pred_scale: Success-prediction penalty scale: 0 until the running success rate reaches
            `delay_until_ratio`, then 1 for good (Isaac's delayed penalty).
        success_rate_ema: Running mean of the per-step success label (time constant `gate_window` steps).
        home_quat: TCP orientation at q_home (gripper pointing down), shape (4,).
        hole_tip: True hole-tip position [m], shape (3,).
        hole_floor: True hole-floor position [m], shape (3,).
        anchor: Action/observation reference seen by the policy [m] (`ctx.state[ANCHOR]`), shape (3,):
            the TCP position at which the peg tip sits at the (estimated) hole opening, i.e.
            hole_tip_estimate + (0, 0, tcp_to_tip). Zero action then means "peg tip at the
            opening", and `ee_pos_rel_anchor` is the peg-tip position relative to the opening
            (the peg is rigid and held vertical). With the hole tip itself as the anchor,
            zero action, which is the untrained policy's mean, drove the peg 10 mm into the block,
            pressing at about 12 N (about -200 contact penalty per episode).
        f_th: Force threshold for the current episode [N].
        force: Instantaneous peg contact force (world frame) at the last tick of the step [N], shape (3,).
        force_norm: ||force|| [N] (single tick: spiky on impacts; used only for the peak metric).
        force_mean: Peg contact force averaged over the physics ticks of the policy step [N], shape (3,)
            (the same signal the policy observes through `contact_force`).
        force_mean_norm: ||force_mean|| [N]; used by the excessive-force penalty and F_mean/F_max metrics.
        kp_dist: Mean keypoint distance to the inserted configuration [m].
        xy_dist: Lateral distance between the peg tip and the hole axis [m].
        tip_height: Peg-tip height above the hole floor [m].
        placed: Peg centered and engaged in the hole.
        success: Peg centered and within `success_dist` of the hole floor.
        first_success_step: First policy step with success (-1 if none).
        p_term: Success-prediction threshold for early termination / ET metrics.
        first_pred_step: First policy step with predicted success p > p_term (-1 if none).
        pred_correct: Whether the task was actually solved at `first_pred_step`.
    """

    def __init__(self, ctx: "Context", peg: str, hole: str, place_xy: float, success_dist: float,
                 p_term: float = 0.9, delay_until_ratio: float = 0.25, gate_window: int = 750) -> None:
        """Resolve ids and allocate buffers.

        Args:
            ctx: Context (the plant must be at q_home after `mj_forward`).
            peg: Peg asset name.
            hole: Socket asset name.
            place_xy: Lateral tolerance for "placed"/"success" [m].
            success_dist: Max tip height above the floor for success [m].
            p_term: Success-prediction threshold for early termination and ET metrics.
            delay_until_ratio: Running success rate that switches the prediction penalty on (0 = always on).
            gate_window: Time constant of the running success rate [policy steps].
        """
        m, d, h = ctx.model, ctx.data, ctx.handles
        self.peg_body = h.body_ids[peg]
        self.peg_tip = h.site_ids[f"{peg}_tip"]
        kp = []
        while f"{peg}_kp{len(kp)}" in h.site_ids:
            kp.append(h.site_ids[f"{peg}_kp{len(kp)}"])
        if len(kp) < 2:
            raise ValueError(f"Peg '{peg}' needs >= 2 keypoint sites")
        self.kp_sites = np.asarray(kp, dtype=np.int64)
        # Distance of each keypoint from the tip, from the model (site positions in the peg body frame).
        self.kp_offsets = np.abs(m.site_pos[self.kp_sites, 2] - m.site_pos[self.peg_tip, 2])
        hole_body = h.body_ids[hole]
        self.hole_mocap = int(m.body_mocapid[hole_body])
        if self.hole_mocap < 0:
            raise ValueError(f"Socket body '{hole}' must be a mocap body")
        self.tip_local = m.site_pos[h.site_ids[f"{hole}_tip"]].copy()
        self.floor_local = m.site_pos[h.site_ids[f"{hole}_floor"]].copy()
        self.tcp_to_tip = float(np.linalg.norm(d.site_xpos[self.peg_tip] - d.site_xpos[h.tcp_site_id]))
        self.tcp_to_tip_true = self.tcp_to_tip
        self.peg_pos_nominal = m.body_pos[self.peg_body].copy()
        self.held_offset = np.zeros(3)
        self.part_geoms = np.flatnonzero((m.geom_bodyid == self.peg_body) | (m.geom_bodyid == hole_body))
        self.friction = float(m.geom_friction[self.part_geoms[0], 0])
        self.home_quat = ctx.plant.ee_quat().copy()
        self.home_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(self.home_quat_inv, self.home_quat)  # conjugate = inverse for unit quaternions
        self.place_xy = place_xy
        self.success_dist = success_dist
        self.p_term = float(p_term)
        # Controller parameters of the episode (overwritten by the randomization events if enabled).
        self.kp = np.asarray(ctx.controller.kp_nominal, dtype=np.float64).copy()
        self.lam = ctx.state["action_max_step"]  # owned by the action term (shape (3,))
        self.ema = ctx.state["action_ema"]
        self.dead_zone = np.zeros(6)
        self.pred = ctx.buffer("pred_success", 1)
        self.delay_until_ratio = float(delay_until_ratio)
        self.gate_window = max(1, int(gate_window))
        self.success_rate_ema = 0.0
        self.pred_scale = 1.0 if self.delay_until_ratio <= 0.0 else 0.0
        self.first_pred_step = -1
        self.pred_correct = False

        self.hole_tip = np.zeros(3)
        self.hole_floor = np.zeros(3)
        self.anchor = ctx.buffer(ANCHOR, 3)
        self.f_th = 0.0
        self.force = np.zeros(3)
        self.force_norm = 0.0
        self.force_mean = np.zeros(3)
        self.force_mean_norm = 0.0
        self._f_peak = 0.0
        self._kp = np.zeros((len(kp), 3))
        self._kp_targ = np.zeros((len(kp), 3))
        self.kp_dist = 0.0
        self.xy_dist = 0.0
        self.tip_height = 0.0
        self.placed = False
        self.success = False
        self.first_success_step = -1
        self.ik_err_max = 0.0
        self._f_sum = 0.0
        self._f_max = 0.0
        self._f_n = 0
        self._placed_any = False

    def set_hole_pose(self, ctx: "Context", base_pos: np.ndarray) -> None:
        """Move the socket (mocap) and update the true hole-tip/floor positions.

        Args:
            ctx: Context.
            base_pos: Socket body origin (bottom of the base plate), world frame [m].
        """
        ctx.data.mocap_pos[self.hole_mocap] = base_pos
        np.add(base_pos, self.tip_local, out=self.hole_tip)    # socket is upright (identity orientation)
        np.add(base_pos, self.floor_local, out=self.hole_floor)

    def reset_episode(self) -> None:
        """Clear per-episode statistics."""
        self.first_success_step = -1
        self.first_pred_step = -1
        self.pred_correct = False
        self._f_sum = 0.0
        self._f_max = 0.0
        self._f_peak = 0.0
        self._f_n = 0
        self._placed_any = False

    def update(self, ctx: "Context") -> None:
        """Recompute per-step quantities from the current simulation state.

        Args:
            ctx: Context.
        """
        d = ctx.data
        np.copyto(self.force, d.cfrc_ext[self.peg_body, 3:6])
        f = self.force
        self.force_norm = math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
        if ctx.episode_step > 0:
            np.copyto(self.force_mean, ctx.obs_mgr.accumulated("contact_force"))
        else:  # reset state: no ticks accumulated yet
            np.copyto(self.force_mean, self.force)
        fm = self.force_mean
        self.force_mean_norm = math.sqrt(fm[0] * fm[0] + fm[1] * fm[1] + fm[2] * fm[2])

        # Keypoints vs. their inserted configuration: tip on the hole floor, axis upright.
        np.take(d.site_xpos, self.kp_sites, axis=0, out=self._kp)
        self._kp_targ[:] = self.hole_floor
        self._kp_targ[:, 2] += self.kp_offsets
        self.kp_dist = float(np.mean(np.linalg.norm(self._kp - self._kp_targ, axis=1)))

        tip = d.site_xpos[self.peg_tip]
        dx, dy = tip[0] - self.hole_tip[0], tip[1] - self.hole_tip[1]
        self.xy_dist = math.sqrt(dx * dx + dy * dy)
        self.tip_height = float(tip[2] - self.hole_floor[2])
        centered = self.xy_dist < self.place_xy
        self.placed = centered and tip[2] < self.hole_tip[2]
        self.success = centered and self.tip_height < self.success_dist

        k = ctx.episode_step
        if k > 0:  # statistics over policy steps (not the reset state)
            self._f_sum += self.force_mean_norm
            self._f_max = max(self._f_max, self.force_mean_norm)
            self._f_peak = max(self._f_peak, self.force_norm)
            self._f_n += 1
            self._placed_any = self._placed_any or self.placed
            # Delayed prediction penalty (Isaac: on once the batch success fraction >= delay_until_ratio;
            # here: per env, from a running mean of the success label over ~gate_window steps).
            self.success_rate_ema += ((1.0 if self.success else 0.0) - self.success_rate_ema) / self.gate_window
            if self.pred_scale == 0.0 and self.success_rate_ema >= self.delay_until_ratio:
                self.pred_scale = 1.0
            if self.success and self.first_success_step < 0:
                self.first_success_step = k
            if self.pred[0] > self.p_term and self.first_pred_step < 0:
                self.first_pred_step = k
                self.pred_correct = self.success

    def episode_info(self, ctx: "Context") -> dict:
        """Episode metrics (paper Table I: duration, F_mean, F_max).

        contact_force_mean / _max are over policy steps of the step-averaged force norm;
        contact_force_peak is the max single-tick (end-of-step) force, which includes impact spikes.
        Early termination (paper Table I): et_triggered (p > p_term at some step), et_correct
        (solved when it first triggered; precision = mean over triggered episodes), et_recall_hit
        (succeeded and triggered correctly; recall = mean over successful episodes), et_delay
        [s] (trigger time - first success time, for correct triggers; -1 otherwise).

        Args:
            ctx: Context.

        Returns:
            Dict merged into `info` at episode end.
        """
        n = max(self._f_n, 1)
        return {
            "contact_force_mean": self._f_sum / n,
            "contact_force_max": self._f_max,
            "contact_force_peak": self._f_peak,
            "time_to_success": self.first_success_step * ctx.policy_dt if self.first_success_step >= 0 else -1.0,
            "ever_success": self.first_success_step >= 0,
            "ever_placed": self._placed_any,
            "force_threshold": self.f_th,
            "ik_err_max": self.ik_err_max,
            "et_triggered": self.first_pred_step >= 0,
            "et_correct": self.first_pred_step >= 0 and self.pred_correct,
            "et_recall_hit": self.first_success_step >= 0 and self.first_pred_step >= 0 and self.pred_correct,
            "et_delay": ((self.first_pred_step - self.first_success_step) * ctx.policy_dt
                         if self.first_pred_step >= 0 and self.pred_correct and self.first_success_step >= 0 else -1.0),
            "pred_success_final": float(self.pred[0]),
            "ctrl_kp": float(np.mean(self.kp[:3])),
            "ctrl_lambda": float(np.mean(self.lam)),
            "ctrl_ema": float(self.ema[0]),
            "dead_zone_force": float(np.mean(self.dead_zone[:3])),
            "part_friction": self.friction,
            "held_offset_x": float(self.held_offset[0]),
            "held_offset_z": float(self.held_offset[2]),
            "success_pred_scale": self.pred_scale,
            "success_rate_ema": self.success_rate_ema,
        }


def _st(ctx: "Context") -> ForgeState:
    """Fetch the task state (created by `forge_init`).

    Args:
        ctx: Context.

    Returns:
        The `ForgeState`.
    """
    return ctx.state["forge"]


# ---------------------------------------------------------------------------------- events
@event_term("forge_init")
def forge_init(ctx: "Context", peg: str = "peg", hole: str = "hole", place_xy: float = 0.0025,
               success_dist: float = 0.001, p_term: float = 0.9, delay_until_ratio: float = 0.25,
               gate_window: int = 750) -> None:
    """Startup: create the `ForgeState` and register the episode-metrics hook.

    Args:
        ctx: Context (plant at q_home).
        peg: Peg asset name.
        hole: Socket asset name.
        place_xy: Lateral tolerance for place/success [m] (paper "Place Dist." 2.5 mm).
        success_dist: Tip height above the hole floor for success [m] (paper: within 1 mm of the base).
        p_term: Success-prediction threshold used for the early-termination metrics.
        delay_until_ratio: Success rate at which the prediction penalty switches on (Isaac: 0.25; 0 = always).
        gate_window: Time constant of the running success rate [policy steps] (750 = 5 episodes).
    """
    ctx.obs_mgr.accumulated("contact_force")  # fail early: the penalty needs the step-averaged force term
    if "action_max_step" not in ctx.state or "action_ema" not in ctx.state:
        raise ValueError("forge_peg needs the `anchor_relative_pos` action term")
    st = ForgeState(ctx, peg, hole, place_xy, success_dist, p_term, delay_until_ratio, gate_window)
    ctx.state["forge"] = st
    ctx.episode_info_hooks.append(st.episode_info)


@event_term("forge_reset_fixed")
def forge_reset_fixed(ctx: "Context", lo: tuple[float, float, float] = (0.55, -0.05, 0.0),
                      hi: tuple[float, float, float] = (0.65, 0.05, 0.1), pos_noise_std: float = 0.0,
                      pos_noise_xy: float = 0.0) -> None:
    """Reset: sample the socket pose and the policy's (optionally noisy) anchor.

    The anchor is the estimated hole tip raised by the TCP-to-peg-tip distance (see `ForgeState.anchor`).

    Args:
        ctx: Context.
        lo: Lower corner of the socket base position [m] (paper Table II "Fixed").
        hi: Upper corner [m].
        pos_noise_std: Std of the per-episode hole-position estimate noise [m] (paper: 2.5 mm; v1: 0).
        pos_noise_xy: Additional horizontal estimate error of exactly this size [m], random direction
            (for controlled evaluation sweeps; 0 = none).
    """
    st = _st(ctx)
    st.reset_episode()
    st.set_hole_pose(ctx, ctx.rng.uniform(lo, hi))
    st.anchor[:] = st.hole_tip
    st.anchor[2] += st.tcp_to_tip  # reference = TCP pose with the peg tip at the opening
    if pos_noise_std > 0.0:
        st.anchor += ctx.rng.normal(0.0, pos_noise_std, size=3)
    if pos_noise_xy > 0.0:
        ang = ctx.rng.uniform(0.0, 2.0 * math.pi)
        st.anchor[0] += pos_noise_xy * math.cos(ang)
        st.anchor[1] += pos_noise_xy * math.sin(ang)


@event_term("forge_reset_ee")
def forge_reset_ee(ctx: "Context", xy_range: float = 0.02, z_range: tuple[float, float] = (0.037, 0.057),
                   yaw_range: float = 0.785, tip_clearance: float = 0.001, ik_tol: float = 1e-3) -> None:
    """Reset: place the TCP above the true hole tip by IK (gripper pointing down, random yaw).

    The TCP target is hole_tip + (U(±xy_range), U(±xy_range), U(z_range)) (paper Table II "Hand: x, y
    (rel)" and "Hand: z (rel)"; Isaac `hand_init_pos` 0.047 ± 0.01) with the gripper yawed by U(±yaw_range)
    about the world z axis (Isaac `hand_init_orn_noise` 0.785 rad). The height is raised if needed so the
    peg tip, including its in-hand offset, starts at least `tip_clearance` above the hole tip. Must run
    after `forge_reset_fixed` and `forge_randomize_held`.

    Args:
        ctx: Context.
        xy_range: Lateral half-range [m].
        z_range: Height range of the TCP above the hole tip [m].
        yaw_range: Yaw half-range [rad] (0 = always the nominal orientation).
        tip_clearance: Minimum initial peg-tip height above the hole tip [m].
        ik_tol: IK error above which the worst error is recorded in the episode metrics [m].
    """
    st = _st(ctx)
    r = ctx.rng
    target = st.hole_tip + np.array([r.uniform(-xy_range, xy_range), r.uniform(-xy_range, xy_range),
                                     r.uniform(*z_range)])
    target[2] = max(target[2], st.hole_tip[2] + st.tcp_to_tip_true + tip_clearance)
    quat = st.home_quat
    if yaw_range > 0.0:
        yaw = r.uniform(-yaw_range, yaw_range)
        qz = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])  # world-z rotation
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, qz, st.home_quat)
    _, err = solve_tcp_ik(ctx.plant, target, quat, q_init=ctx.handles.q_home)
    st.ik_err_max = err if err > ik_tol else 0.0


@event_term("forge_sample_threshold")
def forge_sample_threshold(ctx: "Context", lo: float = 5.0, hi: float = 10.0) -> None:
    """Reset: sample the episode's force threshold F_th ~ U(lo, hi) [N] (paper Table II).

    Args:
        ctx: Context.
        lo: Lower bound [N].
        hi: Upper bound [N].
    """
    _st(ctx).f_th = float(ctx.rng.uniform(lo, hi))


def _isaac_multiplier(rng: np.random.Generator, noise: np.ndarray) -> np.ndarray:
    """Isaac `get_random_prop_gains` factor: m = 1 + U(0, noise), inverted (1 / m) with probability 1/2.

    Args:
        rng: Random generator.
        noise: Per-component noise level, shape (k,).

    Returns:
        Multipliers, shape (k,).
    """
    m = 1.0 + rng.uniform(0.0, 1.0, size=noise.shape) * noise
    return np.where(rng.uniform(size=noise.shape) > 0.5, 1.0 / m, m)


@event_term("forge_randomize_controller")
def forge_randomize_controller(ctx: "Context",
                               kp_default: tuple[float, ...] = (565.0, 565.0, 565.0, 28.0, 28.0, 28.0),
                               kp_noise: tuple[float, ...] = (0.41, 0.41, 0.41, 0.41, 0.41, 0.41),
                               lam_default: float = 0.02, lam_noise: tuple[float, float, float] = (0.25, 0.25, 0.25),
                               ema_range: tuple[float, float] = (0.025, 0.1)) -> None:
    """Reset: sample the controller gains, λ and the action EMA α (Isaac Lab FORGE `_reset_idx`).

    Per component: Kp = kp_default * m, m = 1 + U(0, kp_noise) or its inverse (x: [400, 797] N/m around 565;
    paper Table II: [400, 800]); Kd = 2 sqrt(Kp) when the controller uses automatic damping. Per axis:
    λ = lam_default * m with lam_noise 0.25 ([1.6, 2.5] cm, Table II). α ~ U(ema_range). The maximum
    commandable force per axis is λ Kp (paper: 6.4 to 20 N). Runs before the controller reset, which
    applies the nominal gains. Zero noise / an equal range disables a part.

    Args:
        ctx: Context.
        kp_default: Nominal stiffness [N/m (3), Nm/rad (3)].
        kp_noise: Stiffness noise level per component.
        lam_default: Nominal λ [m].
        lam_noise: λ noise level per axis.
        ema_range: Range of the action smoothing α (Isaac FORGE: [0.025, 0.1]).
    """
    st = _st(ctx)
    c = ctx.controller
    r = ctx.rng
    st.kp[:] = np.asarray(kp_default, dtype=np.float64) * _isaac_multiplier(r, np.asarray(kp_noise, dtype=np.float64))
    c.kp_nominal[:] = st.kp
    if getattr(c, "auto_kd", False):
        c.kd_nominal[:] = 2.0 * np.sqrt(st.kp)
    st.lam[:] = lam_default * _isaac_multiplier(r, np.asarray(lam_noise, dtype=np.float64))
    lo, hi = ema_range
    st.ema[0] = float(r.uniform(lo, hi)) if hi > lo else float(lo)
    if not 0.0 < st.ema[0] <= 1.0:
        raise ValueError(f"ema_range must lie in (0, 1], got {ema_range}")


@event_term("forge_randomize_dead_zone")
def forge_randomize_dead_zone(ctx: "Context",
                              max_dz: tuple[float, ...] = (5.0, 5.0, 5.0, 1.0, 1.0, 1.0)) -> None:
    """Reset / interval: sample the controller's wrench dead zone dz ~ U(0, max_dz) per component.

    Paper Sec. III-B (robot dynamics randomization, [0, 5] N) and Isaac (`default_dead_zone`
    [5, 5, 5, 1, 1, 1], drawn at reset and every 2 s). Commanded wrench components below dz are zeroed,
    larger ones reduced by dz (see `CartesianImpedance.set_dead_zone`).

    Args:
        ctx: Context.
        max_dz: Upper bound per component [N (3), Nm (3)] (zeros disable the dead zone).
    """
    st = _st(ctx)
    st.dead_zone[:] = ctx.rng.uniform(0.0, 1.0, size=6) * np.asarray(max_dz, dtype=np.float64)
    ctx.controller.set_dead_zone(st.dead_zone)


@event_term("forge_randomize_friction")
def forge_randomize_friction(ctx: "Context", lo: float = 0.5, hi: float = 1.0) -> None:
    """Reset: sample the sliding friction of the peg and socket, μ ~ U(lo, hi) (paper Table II, 8 mm peg).

    Both parts get the same μ (MuJoCo uses the larger of the two geoms' coefficients). Isaac's PhysX
    setup (held 0.75, fixed static U(0.25, 1.25), averaged) gives the same static range [0.5, 1.0];
    MuJoCo has no separate static/dynamic coefficient.

    Args:
        ctx: Context.
        lo: Lower bound.
        hi: Upper bound.
    """
    st = _st(ctx)
    st.friction = float(ctx.rng.uniform(lo, hi)) if hi > lo else float(lo)
    ctx.model.geom_friction[st.part_geoms, 0] = st.friction


@event_term("forge_randomize_held")
def forge_randomize_held(ctx: "Context", lo: tuple[float, float, float] = (-0.003, 0.0, -0.003),
                         hi: tuple[float, float, float] = (0.003, 0.0, 0.003)) -> None:
    """Reset: offset the peg in the hand by U(lo, hi) in the hand frame (Isaac `held_asset_pos_noise`).

    Hand frame: z along the peg axis (out of the gripper, so +z lowers the tip), y the finger-closing
    direction (kept 0: the fingers center the peg), x the remaining in-plane direction. The policy is not
    told: the anchor still assumes the nominal grasp, so this adds unobserved lateral and vertical error
    (the critic sees it). The peg stays rigidly attached (no slip during the episode).

    Args:
        ctx: Context.
        lo: Lower offset bound [m].
        hi: Upper offset bound [m].
    """
    st = _st(ctx)
    st.held_offset[:] = ctx.rng.uniform(lo, hi)
    ctx.model.body_pos[st.peg_body] = st.peg_pos_nominal + st.held_offset
    st.tcp_to_tip_true = st.tcp_to_tip + float(st.held_offset[2])


@event_term("forge_update")
def forge_update(ctx: "Context") -> None:
    """Step: recompute the shared per-step quantities (run as a `step` event)."""
    _st(ctx).update(ctx)


# ---------------------------------------------------------------------------------- observations
@obs_term("ee_pos_rel_anchor", dim=3)
def ee_pos_rel_anchor(ctx: "Context", out: np.ndarray) -> None:
    """TCP position relative to the anchor [m], shape (3,).

    For `forge_peg` this is the peg-tip position relative to the (estimated) hole opening, assuming the
    nominal grasp (the in-hand offset from `forge_randomize_held` is not included).
    """
    np.subtract(ctx.plant.ee_pos, ctx.buffer(ANCHOR, 3), out=out)


@obs_term("ee_quat_rel_nominal", dim=4)
def ee_quat_rel_nominal(ctx: "Context", out: np.ndarray) -> None:
    """TCP orientation relative to the nominal gripper-down orientation, q_home^-1 * q (w >= 0), shape (4,).

    The fixed part is upright, so this is the TCP orientation in the fixed-part frame
    up to the constant nominal grasp rotation. It stays near identity, so the w >= 0
    canonicalization is continuous. (The raw down-pointing quaternion has w ≈ 0, where
    the sign flips.)
    """
    st = _st(ctx)
    mujoco.mju_mulQuat(out, st.home_quat_inv, ctx.plant.ee_quat())
    if out[0] < 0.0:
        np.negative(out, out=out)


@obs_term("contact_force", dim=3, needs_accumulation=True)
def contact_force(ctx: "Context", out: np.ndarray) -> None:
    """Peg contact force (world frame) [N], averaged over the policy step, shape (3,)."""
    np.copyto(out, ctx.data.cfrc_ext[_st(ctx).peg_body, 3:6])


@obs_term("force_threshold", dim=1)
def force_threshold(ctx: "Context", out: np.ndarray) -> None:
    """Episode force threshold F_th [N], shape (1,)."""
    out[0] = _st(ctx).f_th


@obs_term("peg_tip_rel_hole_gt", dim=3)
def peg_tip_rel_hole_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: true peg-tip position relative to the true hole tip [m], shape (3,)."""
    st = _st(ctx)
    np.subtract(ctx.data.site_xpos[st.peg_tip], st.hole_tip, out=out)


@obs_term("anchor_error_gt", dim=3)
def anchor_error_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: error of the policy's anchor (estimate - true) [m], shape (3,).

    Zero without position noise; with `pos_noise_std` it is the per-episode hole-pose estimation
    error that the actor must infer from contact (the critic sees it, as in the paper's asymmetric
    actor-critic).
    """
    st = _st(ctx)
    np.subtract(st.anchor, st.hole_tip, out=out)
    out[2] -= st.tcp_to_tip  # anchor = hole tip estimate + (0, 0, tcp_to_tip)


@obs_term("controller_params_gt", dim=16)
def controller_params_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: [Kp (6) [N/m, Nm/rad], λ (3) [m], EMA α (1), dead zone (6) [N, Nm]], shape (16,).

    Isaac's critic state has task_prop_gains, pos/rot_threshold and ema_factor; the dead zone is added
    because it changes during the episode and the actor cannot observe it.
    """
    st = _st(ctx)
    out[0:6] = st.kp
    out[6:9] = st.lam
    out[9] = st.ema[0]
    out[10:16] = st.dead_zone


@obs_term("part_params_gt", dim=4)
def part_params_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: [part friction μ, peg offset in the hand (3) [m]], shape (4,)."""
    st = _st(ctx)
    out[0] = st.friction
    out[1:4] = st.held_offset


@obs_term("success_gt", dim=1)
def success_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: true success label y_t (1 = inserted), shape (1,)."""
    out[0] = 1.0 if _st(ctx).success else 0.0


@obs_term("hole_tip_gt", dim=3)
def hole_tip_gt(ctx: "Context", out: np.ndarray) -> None:
    """Privileged: true hole-tip position [m], shape (3,)."""
    np.copyto(out, _st(ctx).hole_tip)


# ---------------------------------------------------------------------------------- rewards
@reward_term("forge_kp_baseline")
def forge_kp_baseline(ctx: "Context", a: float = 5.0, b: float = 4.0) -> float:
    """Baseline keypoint reward K_{a,b}(d_kp) (Isaac Factory/FORGE `kp_baseline`: a=5, b=4; not in the paper)."""
    return logistic_kernel(_st(ctx).kp_dist, a, b)


@reward_term("forge_kp_coarse")
def forge_kp_coarse(ctx: "Context", a: float = 50.0, b: float = 2.0) -> float:
    """Coarse keypoint reward K_{a,b}(d_kp) (paper App. B, 8 mm peg: a=50, b=2)."""
    return logistic_kernel(_st(ctx).kp_dist, a, b)


@reward_term("forge_kp_fine")
def forge_kp_fine(ctx: "Context", a: float = 100.0, b: float = 0.0) -> float:
    """Fine keypoint reward K_{a,b}(d_kp) (paper App. B, 8 mm peg: a=100, b=0)."""
    return logistic_kernel(_st(ctx).kp_dist, a, b)


@reward_term("forge_place_bonus")
def forge_place_bonus(ctx: "Context") -> float:
    """I_place: 1 while the peg is centered (xy < place_xy) and its tip is inside the hole."""
    return 1.0 if _st(ctx).placed else 0.0


@reward_term("forge_success_bonus")
def forge_success_bonus(ctx: "Context") -> float:
    """I_success: 1 while the peg is centered and its tip is within success_dist of the hole floor."""
    return 1.0 if _st(ctx).success else 0.0


@reward_term("forge_contact_penalty")
def forge_contact_penalty(ctx: "Context") -> float:
    """max(0, ||F|| - F_th) [N] (use weight -β; paper β = 0.2).

    F is the ground-truth peg contact force averaged over the policy step's physics ticks, the
    same signal the policy observes (noise-free). A single end-of-step tick is dominated by
    stiff-contact impact spikes (40-60 N on light touches), which made the penalty swamp the
    task reward and taught the policy to avoid the hole altogether.
    """
    st = _st(ctx)
    return max(0.0, st.force_mean_norm - st.f_th)


@reward_term("forge_success_pred_error")
def forge_success_pred_error(ctx: "Context") -> float:
    """scale * |p - y_t|: predicted success vs. true label (use weight -1; paper Eq. 7).

    scale (`ForgeState.pred_scale`) is 0 until the running success rate reaches `delay_until_ratio`
    (Isaac's delayed penalty), then 1.
    """
    st = _st(ctx)
    if st.pred_scale == 0.0:
        return 0.0
    return st.pred_scale * abs(float(st.pred[0]) - (1.0 if st.success else 0.0))


@reward_term("forge_action_penalty_asset")
def forge_action_penalty_asset(ctx: "Context", lam_nominal: float = 0.02) -> float:
    """||p_targ - p_ee|| / λ_nominal, the unclipped target's distance from the TCP (Isaac `action_penalty_asset`).

    Use a negative weight (Isaac: -0.001). Isaac adds |Δyaw| / rot_threshold; this task has no yaw action.

    Args:
        ctx: Context.
        lam_nominal: Normalizer [m] (Isaac: the nominal `pos_action_threshold`, 0.02).
    """
    d = ctx.state["action_target_delta"]
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / lam_nominal


# ---------------------------------------------------------------------------------- terminations
@termination_term("forge_predicted_success")
def forge_predicted_success(ctx: "Context", p_term: float = 0.9) -> bool:
    """Early termination: the policy predicts success with p > p_term (deployment-style stop)."""
    return float(_st(ctx).pred[0]) > p_term


@termination_term("forge_success")
def forge_success(ctx: "Context") -> bool:
    """Peg inserted (success condition of `ForgeState`)."""
    return _st(ctx).success
