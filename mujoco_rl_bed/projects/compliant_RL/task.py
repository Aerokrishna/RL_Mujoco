"""`compliant_peg`: residual RL for fine peg insertion on top of an axis compliance controller.

The base controller (`AxisCompliance`) pushes the peg along the TCP z axis with a set force and is
compliant laterally and in roll/pitch (stiff in yaw). The policy adds a residual in the controller's
command frame (the TCP frame at the start of the episode, fixed while the peg tilts): side force ±5 N
along x, y, push force 8 ± 7 N (in [1, 15] N) along z, and a tilt target ±10° about x, y held by the weak
rotational spring. All zeros = base controller only.

- Parts and success: as `forge_peg` (8 mm peg, Ø8.5 mm hole, 25 mm deep; tip within 1 mm of the floor
  and within 2.5 mm of the axis).
- Start: gripper down, peg tip 2 mm above the *estimated* hole opening. Estimate error: N(0, 1 mm) per
  axis per episode. Peg tilted in the grasp by U(±3°) about the hand x and y axes (unobserved).
- Timing: 25 Hz policy (2 ms x 20), 8 s episodes (200 steps), no early termination in training.
- Actor observation (29): estimated hole position (3) and hole axis (3) in the TCP frame, contact wrench
  about the TCP in the TCP frame (6, step mean), TCP twist in the TCP frame (6), previous applied action
  (5); the wrench and twist also with their previous-step values (history 2). Noise: position 0.25 mm,
  axis 0.002, force 1 N, torque 0.05 Nm, velocity 0.02 m/s and 0.1 rad/s.
- Critic: true peg-tip offset, peg axis, clean wrench/twist, estimate error, grasp tilt, success label,
  friction, world contact force.
- Reward: K_coarse(50, 2) + K_fine(100, 0) + 0.25 I(tip within 0.5 mm of the axis) + 0.25 I(axis within
  1°) + I_success - 0.2 max(0, ||F|| - 10 N) - 0.05 ||s_t - s_{t-1}||.
- Randomization: friction U[0.5, 1.0].
"""

from __future__ import annotations

from mujoco_rl_bed.control.axis_compliance import AxisComplianceCfg
from mujoco_rl_bed.env.cfg import (ActionCfg, EventTermCfg, ObsCfg, ObsTermCfg, RewardTermCfg, TaskCfg,
                                   TerminationTermCfg)
from mujoco_rl_bed.sim.assets import BoxPeg, Peg, RectHole, RoundHole
from mujoco_rl_bed.sim.scene import SceneCfg
from mujoco_rl_bed.tasks.registry import register_task

POLICY_TERMS = ["cr_hole_rel_tcp", "cr_hole_axis_tcp", "cr_wrench_tcp", "cr_twist_tcp", "last_action"]
CRITIC_TERMS = ["peg_tip_rel_hole_gt", "cr_peg_axis_gt", "cr_wrench_tcp", "cr_twist_tcp", "last_action",
                "anchor_error_gt", "cr_grasp_tilt_gt", "success_gt", "part_params_gt", "contact_force"]


@register_task("compliant_peg")
def compliant_peg() -> TaskCfg:
    """Return the compliant residual insertion task config.

    Returns:
        A new `TaskCfg`.
    """
    return TaskCfg(
        scene=SceneCfg(assets=[Peg(name="peg"), RoundHole(name="hole", inner_radius=0.00425)], gripper_open=0.0625),
        # Slow approach (8 N / 250 Ns/m ≈ 32 mm/s), heavy virtual mass and a 30 Hz wrench filter so the
        # impact on the hole rim does not kick the peg sideways; a stiff inner loop (8000 N/m) keeps the
        # free-space TCP within ~0.2 mm of the virtual frame (3000 N/m drifted 0.35 mm > clearance).
        controller=AxisComplianceCfg(m_lin=2.0, k_lat=200.0, d_lat=40.0, d_axial=250.0, i_rot=0.02, k_tilt=0.1,
                                     d_tilt=0.4, k_yaw=30.0, kp_in=8000.0, kr_in=100.0, wrench_filter_hz=30.0),
        action=ActionCfg(term="axis_compliance_residual", ema_factor=0.5,
                         params={"f_lat_max": 10.0, "f_push_nominal": 10.0, "f_push_range": 7.0,
                                 "f_push_min": 0.1, "f_push_max": 15.0, "tilt_max_deg": 10.0}),
        obs=ObsCfg(
            groups={"policy": list(POLICY_TERMS), "critic": list(CRITIC_TERMS)},
            term_cfg={
                "cr_hole_rel_tcp": ObsTermCfg(noise_std=0.00025),
                "cr_hole_axis_tcp": ObsTermCfg(noise_std=0.002),
                "cr_wrench_tcp": ObsTermCfg(noise_std=(1.0, 1.0, 1.0, 0.05, 0.05, 0.05), history=2),
                "cr_twist_tcp": ObsTermCfg(noise_std=(0.02, 0.02, 0.02, 0.1, 0.1, 0.1), history=2),
            },
        ),
        rewards={
            "kp_coarse": RewardTermCfg(weight=1.0, func="forge_kp_coarse", params={"a": 50.0, "b": 2.0}),
            "kp_fine": RewardTermCfg(weight=1.0, func="forge_kp_fine", params={"a": 100.0, "b": 0.0}),
            "xy_aligned": RewardTermCfg(weight=0.25, func="cr_xy_aligned", params={"tol": 0.0005}),
            "axis_aligned": RewardTermCfg(weight=0.25, func="cr_axis_aligned", params={"tol_deg": 1.0}),
            "success_bonus": RewardTermCfg(weight=1.0, func="forge_success_bonus"),
            "contact_penalty": RewardTermCfg(weight=-0.2, func="forge_contact_penalty"),
            "action_grad": RewardTermCfg(weight=-0.05, func="applied_action_rate_norm"),
        },
        terminations={
            "unstable": TerminationTermCfg(),
            "success": TerminationTermCfg(func="forge_success", success=True, ends_episode=False),
        },
        events={
            "init": EventTermCfg(mode="startup", func="forge_init",
                                 params={"peg": "peg", "hole": "hole", "place_xy": 0.0025, "success_dist": 0.001}),
            "cr_init": EventTermCfg(mode="startup", func="cr_init"),  # wrench measurement -> controller
            "reset_fixed": EventTermCfg(mode="reset", func="forge_reset_fixed",
                                        params={"lo": (0.55, -0.05, 0.0), "hi": (0.65, 0.05, 0.1),
                                                "pos_noise_std": 0.001}),
            "reset_threshold": EventTermCfg(mode="reset", func="forge_sample_threshold", params={"lo": 10.0, "hi": 10.0}),
            "randomize_friction": EventTermCfg(mode="reset", func="forge_randomize_friction",
                                               params={"lo": 0.5, "hi": 1.0}),
            "randomize_grasp": EventTermCfg(mode="reset", func="cr_randomize_grasp", params={"tilt_max_deg": 3.0}),
            "reset_ee": EventTermCfg(mode="reset", func="cr_reset_ee",
                                     params={"hover": 0.002, "xy_jitter": 0.0005, "tip_clearance": 0.001}),
            "update": EventTermCfg(mode="step", func="forge_update"),
        },
        episode_length_s=28.0,
        env_defaults={"decimation": 20},
    )


S1_POLICY_TERMS = ["cr_hole_rel_tcp", "cr_hole_axis_tcp", "cr_wrench_tcp", "cr_twist_tcp", "cr_spring_defl", "last_action"]
S1_CRITIC_TERMS = CRITIC_TERMS + ["cr_spring_defl"]


@register_task("compliant_peg_s1")
def compliant_peg_s1() -> TaskCfg:
    """Stage 1 of the lateral-search curriculum: exact hole pose, peg tip starts 2 mm off the hole axis.

    - Start: gripper down, no tilt, peg tip 2 mm above the hole opening and exactly 2 mm off its axis
      (random direction). The hole estimate is exact, so the policy sees the true tip-to-hole vector.
    - Action (3): lateral anchor shift of the xy spring, ±4 mm, and push force in [1, 8] N
      (`axis_compliance_anchor`); no tilt command (the compliance corrects tilt).
    - Actor observation (35): hole position (3, exact up to 0.25 mm sensor noise) and axis (3) in the TCP
      frame, contact wrench and TCP twist (6 + 6, history 2, sensor noise), lateral spring deflection (2),
      previous applied action (3). Critic: as `compliant_peg` + spring deflection.
    - Reward: K_coarse(50, 2) + K_fine(100, 0) + 0.5 K(1000, 0)(tip xy distance) + I_place (tip ≥ 1 mm inside) + I_success
      - 0.2 max(0, ||F|| - 10 N) - 0.1 ||s_t - s_{t-1}||.
    - Controller: `compliant_peg` gains (k_lat 200 N/m: a 4 mm anchor shift pulls with 0.8 N, so the peg is
      best moved before contact or with a light push; 1000 N/m went unstable in contact, 450 N peaks).
      A privileged P-controller through this action inserts 19/20 in about 2.9 s (return ~410; zero action ~57).
    - 25 Hz policy, 8 s episodes, friction U[0.5, 1.0], no early termination.

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg()
    cfg.action = ActionCfg(term="axis_compliance_anchor", ema_factor=0.5,
                           params={"anchor_max": 0.004, "f_push_nominal": 4.5, "f_push_range": 3.5,
                                   "f_push_min": 1.0, "f_push_max": 8.0})
    cfg.obs.groups = {"policy": list(S1_POLICY_TERMS), "critic": list(S1_CRITIC_TERMS)}
    cfg.rewards = {
        "kp_coarse": RewardTermCfg(weight=1.0, func="forge_kp_coarse", params={"a": 50.0, "b": 2.0}),
        "kp_fine": RewardTermCfg(weight=1.0, func="forge_kp_fine", params={"a": 100.0, "b": 0.0}),
        "kp_xy": RewardTermCfg(weight=0.5, func="cr_kp_xy", params={"a": 1000.0, "b": 0.0}),
        "place_bonus": RewardTermCfg(weight=1.0, func="cr_place_bonus", params={"depth": 0.001}),
        "success_bonus": RewardTermCfg(weight=1.0, func="forge_success_bonus"),
        "contact_penalty": RewardTermCfg(weight=-0.2, func="forge_contact_penalty"),
        "action_grad": RewardTermCfg(weight=-0.1, func="applied_action_rate_norm"),
    }
    ev = cfg.events
    ev["reset_fixed"].params["pos_noise_std"] = 0.0
    ev["randomize_grasp"].params["tilt_max_deg"] = 0.0
    ev["reset_ee"].params.update({"hover": 0.002, "xy_jitter": 0.0})
    events = {}
    for k, v in ev.items():
        events[k] = v
        if k == "reset_ee":
            events["shift_ee"] = EventTermCfg(mode="reset", func="cr_shift_ee",
                                              params={"offset_min": 0.002, "offset_max": 0.002})
    cfg.events = events
    cfg.episode_length_s = 8.0
    return cfg


S2_POLICY_TERMS = ["cr_wrench_tcp", "sp_base_offset", "sp_depth", "cr_spring_defl", "sp_phase", "last_action"]
S2_CRITIC_TERMS = ["peg_tip_rel_hole_gt", "cr_peg_axis_gt", "success_gt", "part_params_gt", "contact_force"] + S2_POLICY_TERMS


@register_task("compliant_peg_s2")
def compliant_peg_s2() -> TaskCfg:
    """Stage 2: spiral search + residual RL; the hole estimate is 2 mm off, the peg starts tilted.

    - Start: no grasp tilt, gripper tilted about the peg tip by U(0, 6°) (random horizontal axis), peg tip
      2 mm above the hole opening and exactly 2 mm off its axis (random direction). The start position is the
      (wrong) hole estimate: the spiral is centred on it.
    - Base (zero residual, `spiral_residual`): approach at 1 N, spiral of the lateral anchor in the start TCP
      frame (pitch 0.8 mm, 10 mm/s, out to 2.6 mm) with stiff search gains (k_lat 2000, k_tilt 3), on a
      0.5 mm drop switch to the controller gains (k_lat 200, k_tilt 0.1) and push 5 N. Scripted, this inserts
      57-62 % of episodes within 10 s.
    - Action (3): lateral residual on the anchor (±1 mm) and spiral speed scale ([0, 2]).
    - Actor observation (15): contact wrench about the TCP in the TCP frame (6, step mean, noise 1 N / 0.05 Nm),
      spiral base offset (2), depth since touchdown (1), lateral spring deflection (2), phase (1), previous
      applied action (3). Critic: true tip offset from the hole, peg axis, success label, friction, world contact
      force (needed by the contact penalty) + the
      actor terms (clean).
    - Reward: K_coarse(50, 2) + K_fine(100, 0) + 0.5 K(1000, 0)(tip xy distance) + I_place (tip ≥ 1 mm inside)
      + I_success - 0.2 max(0, ||F|| - 10 N) - 0.1 ||s_t - s_{t-1}|| - 0.01 ||a||^2.
    - 25 Hz policy, 10 s episodes, friction U[0.5, 1.0], no early termination.

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s1()
    cfg.action = ActionCfg(term="spiral_residual", ema_factor=0.5,
                           params={"f_search": 1.0, "f_insert": 5.0, "v": 0.01, "pitch": 0.0008, "r_max": 0.0026,
                                   "drop_th": 0.0005, "res_max": 0.001, "ks_lat": 2000.0, "ds_lat": 190.0,
                                   "ks_tilt": 3.0, "ds_tilt": 1.5})
    cfg.obs = ObsCfg(
        groups={"policy": list(S2_POLICY_TERMS), "critic": list(S2_CRITIC_TERMS)},
        term_cfg={"cr_wrench_tcp": ObsTermCfg(noise_std=(1.0, 1.0, 1.0, 0.05, 0.05, 0.05))},
    )
    cfg.rewards["residual_l2"] = RewardTermCfg(weight=-0.01, func="action_l2")
    events = {}
    for k, v in cfg.events.items():
        if k == "shift_ee":
            events["tilt_ee"] = EventTermCfg(mode="reset", func="cr_tilt_ee", params={"tilt_max_deg": 6.0})
        events[k] = v
    cfg.events = events
    cfg.episode_length_s = 10.0
    return cfg


@register_task("compliant_peg_s2b")
def compliant_peg_s2b() -> TaskCfg:
    """Stage 2b: `compliant_peg_s2` with a larger hole-estimate error and tilt.

    - Start: tilt U(0, 8°), peg tip U(0, 4 mm) off the hole axis (random direction), 2 mm above the opening.
    - Spiral out to 4.6 mm (4 mm + tilt drift + margin); residual ±1.5 mm, spiral speed scale [0, 3].
    - Same observations (15), action size (3), rewards and gains as `compliant_peg_s2`, so a stage-2 run can
      initialize it (`init_from`). Episode length: 20 s (searching out to 4 mm takes ~6-9 s at 10 mm/s).

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2()
    cfg.action.params.update({"r_max": 0.0046, "res_max": 0.0015, "speed_max": 3.0})
    cfg.events["tilt_ee"].params["tilt_max_deg"] = 8.0
    cfg.events["shift_ee"].params.update({"offset_min": 0.0, "offset_max": 0.004})
    cfg.episode_length_s = 20.0
    return cfg


@register_task("compliant_peg_s2_box")
def compliant_peg_s2_box() -> TaskCfg:
    """`compliant_peg_s2` with a cuboid peg in a rectangular hole of the same size (yaw held).

    - Peg: 8 x 8 mm cross-section (the cylinder's diameter), 50 mm long; hole 8.5 x 8.5 mm (0.25 mm clearance per
      side, as the round hole), 25 mm deep, in the same solid 6 x 6 cm block. Change `half_x`/`half_y` and
      `inner_half_x`/`inner_half_y` together for a non-square rectangle.
    - The hole is turned to the gripper yaw at startup (`cr_align_hole_yaw`): the controller holds yaw, so there is
      no yaw error to correct. Everything else (start, spiral base, observations, actions, rewards) as
      `compliant_peg_s2`, so its policies run unchanged.

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2()
    cfg.scene.assets = [BoxPeg(name="peg", half_x=0.004, half_y=0.004),
                        RectHole(name="hole", inner_half_x=0.00425, inner_half_y=0.00425)]
    events = {}
    for k, v in cfg.events.items():
        events[k] = v
        if k == "init":
            events["align_hole_yaw"] = EventTermCfg(mode="startup", func="cr_align_hole_yaw")
    cfg.events = events
    return cfg


S2C_POLICY_TERMS = list(S2_POLICY_TERMS)


@register_task("compliant_peg_s2c")
def compliant_peg_s2c() -> TaskCfg:
    """Stage 2c: lateral residual only, on a curriculum from small to large hole-estimate errors.

    - Action (2): lateral residual on the spiral anchor (±1 mm); spiral speed fixed at 10 mm/s.
    - Curriculum (`cr_curriculum_start`, per env): offset U(0, 2 / 3 / 4 mm), tilt U(0, 6 / 7 / 8°), spiral r_max
      2.6 / 3.6 / 4.6 mm; advance at a success rate of 0.92 (stage 1) / 0.85 (stage 2) over the env's last 20
      episodes (zero-residual success with 20 s episodes: 97.5 % / 85 % / 75 %).
    - Actor observation (26): contact wrench with history 3 (18), spiral base offset (2), depth since touchdown
      (1), lateral spring deflection (2), phase (1), previous applied action (2). Critic as `compliant_peg_s2`.
    - Rewards as `compliant_peg_s2`; 20 s episodes; episode metrics include the curriculum stage and the residual
      usage (mean |residual| and its component toward the true hole, per phase).

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2()
    cfg.action.params.update({"speed_action": False, "res_max": 0.001, "r_max": 0.0026})
    cfg.obs.term_cfg["cr_wrench_tcp"] = ObsTermCfg(noise_std=(1.0, 1.0, 1.0, 0.05, 0.05, 0.05), history=3)
    events = {}
    for k, v in cfg.events.items():
        if k in ("tilt_ee", "shift_ee"):
            continue
        events[k] = v
        if k == "reset_ee":
            events["curriculum"] = EventTermCfg(mode="reset", func="cr_curriculum_start")
    cfg.events = events
    cfg.episode_length_s = 20.0
    return cfg


@register_task("compliant_peg_s2d")
def compliant_peg_s2d() -> TaskCfg:
    """Stage 2d: `compliant_peg_s2c` with a stricter curriculum that only advances on real correction.

    Stages (offset, tilt, spiral r_max): exactly 2 mm / 6° / 2.6 mm, exactly 3 mm / 7° / 3.6 mm, exactly 4 mm /
    8° / 4.6 mm, then U(0, 4 mm) / 8° / 4.6 mm (final). Zero-residual success with 20 s episodes: 82.5 / 57.5 /
    42.5 / 75 %. A stage is left when the env's success rate over its last 50 episodes reaches 0.95 / 0.80 /
    0.65, i.e. 12-23 points above the scripted spiral. Everything else as `compliant_peg_s2c`.

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2c()
    cfg.events["curriculum"].params.update({
        "stages": ((0.002, 0.002, 6.0, 0.0026), (0.003, 0.003, 7.0, 0.0036), (0.004, 0.004, 8.0, 0.0046),
                   (0.0, 0.004, 8.0, 0.0046)),
        "advance_at": (0.95, 0.80, 0.65), "window": 50})
    return cfg


@register_task("compliant_peg_s2e")
def compliant_peg_s2e() -> TaskCfg:
    """Stage 2e: dwell-capable search. Lateral residual + spiral slow-down/pause on the full error range.

    - Start: offset U(0, 4 mm), tilt U(0, 8°) (as `compliant_peg_s2b`), 20 s episodes, spiral r_max 4.6 mm.
    - Action (3): lateral residual on the anchor (±1.5 mm, enough for a 1 mm local search) and the spiral speed
      (1 + a) / 2 x 10 mm/s: zero residual runs a slower 5 mm/s spiral, -1 pauses it, +1 runs it at 10 mm/s.
    - Actor observation (27): contact wrench with history 3 (the proximity cue: held-out ROC AUC ~0.70 for
      "tip 0.25-1 mm from the axis" vs farther), spiral offset, depth, spring deflection, phase, previous action.
    - Reward as `compliant_peg_s2` with the close-to-hole kernel doubled (0.5 -> 1.0 K(1000, 0)(tip xy distance)),
      so dwelling where the wrench suggests the hole pays. Episode metrics add the mean speed scale with the tip
      within 1 mm of the axis vs farther (logging only).

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2b()
    cfg.action.params.update({"speed_affine": True, "speed_lo": 0.0, "speed_hi": 1.0, "v": 0.01, "res_max": 0.0015})
    cfg.obs.term_cfg["cr_wrench_tcp"] = ObsTermCfg(noise_std=(1.0, 1.0, 1.0, 0.05, 0.05, 0.05), history=3)
    cfg.rewards["kp_xy"] = RewardTermCfg(weight=1.0, func="cr_kp_xy", params={"a": 1000.0, "b": 0.0})
    return cfg


S2F_POLICY_TERMS = ["cr_wrench_tcp", "cr_spring_defl", "sp_tcp_lat_vel", "sp_tcp_tilt", "sp_base_offset", "sp_depth",
                    "sp_phase", "sp_try_state", "last_action"]
S2F_CRITIC_TERMS = ["peg_tip_rel_hole_gt", "cr_peg_axis_gt", "success_gt", "part_params_gt", "contact_force",
                    "episode_progress"] + S2F_POLICY_TERMS


@register_task("compliant_peg_s2f")
def compliant_peg_s2f() -> TaskCfg:
    """Stage 2f: catch-and-insert. The spiral (1.5 N) plus a lateral residual and a "try insert" action.

    Near misses where the hole edge catches the peg (tip 0-0.35 mm from the axis) give a 3-5 N lateral force and a
    2-2.7 mm spring deflection toward the hole that survive the sensor noise; a lateral nudge does not drop a tilted
    peg hung on the rim, but pausing with soft gains and a 5 N push does. Scripted reference through this action
    (|F_lat| > 3.5 N for 3 steps -> try for 0.6 s, anchor onto the virtual frame): 50/60 vs 49/60 zero residual
    (seeds 2000-2059).

    - Start as `compliant_peg_s2b`: offset U(0, 4 mm), tilt U(0, 8°), 20 s episodes, spiral r_max 4.6 mm, f_search 1.5 N.
    - Action (3): lateral residual on the anchor (±1 mm) and try level tau (dead zone below a = 0.3): spiral speed
      1 - tau, push 1.5 + 3.5 tau N, search -> task gains (log-interpolated). Zero = the scripted spiral.
    - Actor observation (20): contact wrench (6, noise 1 N / 0.05 Nm), spring deflection (2), TCP lateral velocity
      (2, noise 1 mm/s) and roll/pitch from the start (2, noise 1 mrad), spiral offset (2), depth (1), phase (1),
      try state (tau, time in try) (2), previous applied action (3). Meant for a recurrent (LSTM) policy.
    - Critic: actor terms (clean) + true tip offset, peg axis, success label, part parameters, contact force,
      episode progress.
    - Reward: as `compliant_peg_s2` - 0.1 tau per step.

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2b()
    cfg.action.params.update({"f_search": 1.5, "res_max": 0.001, "try_action": True, "f_try": 5.0, "try_dead": 0.3})
    cfg.obs = ObsCfg(
        groups={"policy": list(S2F_POLICY_TERMS), "critic": list(S2F_CRITIC_TERMS)},
        term_cfg={"cr_wrench_tcp": ObsTermCfg(noise_std=(1.0, 1.0, 1.0, 0.05, 0.05, 0.05)),
                  "sp_tcp_lat_vel": ObsTermCfg(noise_std=0.001),
                  "sp_tcp_tilt": ObsTermCfg(noise_std=0.001)},
    )
    cfg.rewards["try_cost"] = RewardTermCfg(weight=-0.1, func="sp_try_cost")
    return cfg


@register_task("compliant_peg_s2g")
def compliant_peg_s2g() -> TaskCfg:
    """Stage 2g: `compliant_peg_s2f` with the try level as the only action and a 5x higher try cost.

    Run G (s2f, LSTM, from scratch) learnt to try often and long (2 tries, ~3.6 s paused per episode, 5-8 % of the
    try steps at the hole) and fell to 35/60 at 1M steps (zero residual 49/60): blanket trying paid off in
    small-offset episodes. Here the lateral residual is removed (it added nothing to the scripted rule) and a full
    try costs 0.5 per step, so a 0.6 s false try costs ~7.5, about one step of success reward x 2.5 s.

    - Action (1): try level tau (dead zone below a = 0.3), as `compliant_peg_s2f`; zero = the scripted spiral.
    - Actor observation (19): as `compliant_peg_s2f` (previous applied action is 1-D). Critic as `compliant_peg_s2f`.
    - Reward: as `compliant_peg_s2` - 0.5 tau per step.

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2f()
    cfg.action.params["lateral_action"] = False
    cfg.rewards["try_cost"].weight = -0.5
    return cfg


@register_task("compliant_peg_s2h")
def compliant_peg_s2h() -> TaskCfg:
    """Stage 2h: `compliant_peg_s2g` at 0.1 mm clearance (hole radius 4.1 mm) with an action cost on the try output.

    At 0.1 mm the spiral (pitch 0.8 mm, ~1 mm tracking lag) mostly passes the hole without dropping in: plain spiral
    11/60, scripted rule 14/60, BC warm start 16/60, run H (s2g, many small tries) 24/60 at 4.6 mm reach (seeds
    2000-2059, error 0-5 mm). Training here should teach where slowing/trying pays instead of everywhere.

    - Start: hole-estimate error U(0, 5 mm), tilt U(0, 8°); spiral r_max 5.6 mm (all starts reachable); 20 s episodes.
    - Action, observations, critic as `compliant_peg_s2g`.
    - Reward: as `compliant_peg_s2g` - 0.05 (a + 1)^2 / 4 per step on the raw try action (no flat region below the
      dead zone, so a policy that does not try drifts to -1).

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2g()
    cfg.scene.assets = [Peg(name="peg"), RoundHole(name="hole", inner_radius=0.0041)]
    cfg.action.params["r_max"] = 0.0056
    cfg.events["shift_ee"].params.update({"offset_min": 0.0, "offset_max": 0.005})
    cfg.rewards["try_action_cost"] = RewardTermCfg(weight=-0.05, func="sp_try_action_cost")
    return cfg


S2I_POLICY_TERMS = ["cr_wrench_tcp", "cr_spring_defl", "sp_tcp_lat_vel", "sp_tcp_tilt", "sp_depth", "sp_phase",
                    "sp_try_state", "last_action"]
S2I_CRITIC_TERMS = ["peg_tip_rel_hole_gt", "cr_peg_axis_gt", "success_gt", "part_params_gt", "contact_force",
                    "episode_progress", "sp_base_offset"] + S2I_POLICY_TERMS


@register_task("compliant_peg_s2i")
def compliant_peg_s2i() -> TaskCfg:
    """Stage 2i: `compliant_peg_s2g` with a binary push decision and no spiral-offset input.

    Offline classifiers on plain-spiral rollouts (tip within 0.35 mm of the axis = catch, 600 episodes, held-out
    episodes) found the catch detectable well beyond the scripted rule (catch events found: rule 28 %, all actor
    inputs 75 %); depth and lateral velocity carry most of it, the spiral offset nothing (a position prior only).

    - Action: Discrete(2), 0 = keep searching, 1 = full try (pause, task gains, 5 N) for 15 steps (0.6 s), during which
      the action is ignored (`try_macro`).
    - Actor observation (17): contact wrench (6, noise 1 N / 0.05 Nm), spring deflection (2), TCP lateral velocity (2),
      roll/pitch (2), depth (1), phase (1), try state (2), try flag of the previous step (1).
    - Critic: privileged terms + spiral offset + the actor terms (clean). Reward as `compliant_peg_s2g`.

    Returns:
        A new `TaskCfg`.
    """
    cfg = compliant_peg_s2g()
    cfg.action.params["try_macro"] = 15
    cfg.obs.groups = {"policy": list(S2I_POLICY_TERMS), "critic": list(S2I_CRITIC_TERMS)}
    return cfg
