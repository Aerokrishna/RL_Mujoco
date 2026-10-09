"""`forge_peg`: FORGE 8 mm round peg insertion (arXiv:2408.04587), close to Isaac Lab's FORGE.

Values follow the paper's Table II (8 mm peg). Where the paper is silent they follow NVIDIA's
reference implementation (Isaac Lab `direct/forge` + `direct/factory`, "Isaac" below).

- Parts: 8 mm peg attached to the hand (rigid during an episode, but offset in the hand by
  U(±3 mm) in x and z per episode, unobserved), socket = solid 6x6 cm block with a Ø8.5 mm hole
  (0.25 mm radial / 0.5 mm diametrical clearance, as in the paper), 25 mm deep. Success = peg tip
  within 1 mm of the hole floor and centered (2.5 mm).
- Timing: 15 Hz policy (sim_dt 2 ms x decimation 33 = 15.15 Hz), 150 steps = 9.9 s (paper: 15 Hz, 10 s).
- Initial state: socket base x∈[0.55, 0.65], y∈[-0.05, 0.05], z∈[0, 0.1] m; TCP at hole tip
  + (U±2 cm, U±2 cm, U[3.7, 5.7] cm), gripper down with yaw U(±45°) (IK).
- Action: 3D TCP target = anchor + s * 5 cm, s = EMA-smoothed action, anchor = TCP pose with the peg tip
  at the (noisy) hole opening. Clipped per axis to within λ of the TCP (Eq. 5). Orientation held.
  4th dim a_ET = success prediction (smoothed too, as in Isaac).
- Dynamics randomization (per episode, critic-only): Kp x/÷(1+U(0, 0.41)) around (565 N/m, 28 Nm/rad)
  per axis, Kd = 2 sqrt(Kp); λ x/÷(1+U(0, 0.25)) around 2 cm per axis; EMA α ~ U[0.025, 0.1]; wrench dead
  zone U(0, [5 N, 1 Nm]) per component, re-drawn every 2 s; part friction U[0.5, 1.0].
- Observation (actor, 21): peg-tip pos rel. hole opening (3, via the anchor), TCP quat rel. nominal (4),
  TCP twist (6), contact force (3, step mean), F_th (1), previous applied action (4). Noise (Isaac):
  position 0.25 mm, orientation 0.1° (as quaternion components), velocity as finite differences of the
  noisy pose over Isaac's 1/120 s physics step (0.042 m/s, 0.17 rad/s), force 1 N; hole estimate
  N(0, `pos_noise_std`) per episode (Isaac 1 mm, paper 2.5 mm). Critic: clean state, see CRITIC_TERMS.
- Reward: K(5, 4) + K_coarse(50, 2) + K_fine(100, 0) + I_place + I_success - 0.2 max(0, ||F|| - F_th)
  - |p - y| (on once the success rate reaches 25%) - 0.1 ||s_t - s_{t-1}|| - 0.001 ||p_targ - p_ee|| / 2 cm,
  F_th ~ U[5, 10] N per episode.
- No early termination in training; `is_success` reports success at the final step and `ever_success`
  at any step. Early stop at p > 0.9 only when enabled (evaluation).

Deviations from Isaac: force = step-mean peg contact force (Isaac: wrist force, EMA 0.25 per physics
step); roll/pitch rates stay observed (Isaac zeroes them); no yaw action (round peg); held-part mass
is not randomized (the controller's gravity compensation would cancel it exactly).
"""

from __future__ import annotations

from mujoco_rl_bed.control.cartesian_impedance import CartesianImpedanceCfg
from mujoco_rl_bed.env.cfg import (ActionCfg, EventTermCfg, ObsCfg, ObsTermCfg, RewardTermCfg, TaskCfg,
                                   TerminationTermCfg)
from mujoco_rl_bed.sim.assets import Peg, RoundHole
from mujoco_rl_bed.sim.scene import SceneCfg
from mujoco_rl_bed.tasks.registry import register_task

POLICY_TERMS = ["ee_pos_rel_anchor", "ee_quat_rel_nominal", "ee_vel", "contact_force", "force_threshold", "last_action"]
# Clean critic state (asymmetric=true): ground-truth peg-tip offset instead of the noisy-relative
# position, the actor's other terms without noise, the hole-estimate error (explains what the actor
# believes), the true success label, joint positions, and every randomized dynamics parameter.
CRITIC_TERMS = ["peg_tip_rel_hole_gt", "ee_quat_rel_nominal", "ee_vel", "contact_force", "force_threshold",
                "last_action", "anchor_error_gt", "success_gt", "joint_pos", "controller_params_gt",
                "part_params_gt"]

# Observation noise (Isaac `ForgeObsRandCfg`): fingertip position 0.25 mm, rotation 0.1°, force 1 N.
# Isaac's velocities are finite differences of the noisy pose over its 1/120 s physics step:
# lin. std = sqrt(2) * 0.25 mm * 120 = 0.042 m/s; ang. per axis = sqrt(2) * 0.1° / sqrt(3) * 120 = 0.17 rad/s.
# A 0.1° rotation about a random axis moves each quaternion vector component by 0.1° / (2 sqrt(3)).
POS_NOISE = 0.00025
QUAT_NOISE = (0.0, 0.0005, 0.0005, 0.0005)
VEL_NOISE = (0.042, 0.042, 0.042, 0.17, 0.17, 0.17)
FORCE_NOISE = 1.0

# Overrides that switch off the dynamics randomization and observation noise (scripted expert, tests,
# ablations). The scripted expert cannot compensate the dead zone.
NO_DR = {
    "task.events.dead_zone.params.max_dz": "(0,0,0,0,0,0)",
    "task.events.dead_zone_interval.params.max_dz": "(0,0,0,0,0,0)",
    "task.events.randomize_controller.params.kp_noise": "(0,0,0,0,0,0)",
    "task.events.randomize_controller.params.lam_noise": "(0,0,0)",
    "task.events.randomize_controller.params.ema_range": "(0.2,0.2)",
    "task.events.randomize_friction.params.lo": "0.75",
    "task.events.randomize_friction.params.hi": "0.75",
    "task.events.randomize_held.params.lo": "(0,0,0)",
    "task.events.randomize_held.params.hi": "(0,0,0)",
    "task.events.reset_ee.params.yaw_range": "0.0",
    "task.obs.term_cfg.ee_pos_rel_anchor.noise_std": "0.0",
    "task.obs.term_cfg.ee_quat_rel_nominal.noise_std": "0.0",
    "task.obs.term_cfg.ee_vel.noise_std": "0.0",
    "task.obs.term_cfg.contact_force.noise_std": "0.0",
}


@register_task("forge_peg")
def forge_peg() -> TaskCfg:
    """Return the FORGE peg-insertion task config.

    Returns:
        A new `TaskCfg`.
    """
    return TaskCfg(
        scene=SceneCfg(
            assets=[Peg(name="peg"), RoundHole(name="hole", inner_radius=0.00425)],  # 0.25 mm radial (paper)
            gripper_open=0.0625,  # fingers close to ~8 mm (visual only: the peg is fixed to the hand)
        ),
        # Isaac FORGE nominal gains; randomized per episode by `randomize_controller`.
        controller=CartesianImpedanceCfg(kp=(565.0, 565.0, 565.0, 28.0, 28.0, 28.0)),
        action=ActionCfg(term="anchor_relative_pos", anchor="fixed_anchor", anchor_bounds=(0.05, 0.05, 0.05),
                         max_step=0.02, pos_lo=(0.3, -0.3, 0.0), pos_hi=(0.85, 0.3, 0.5),
                         success_prediction=True,   # 4th action dim a_ET -> predicted success p
                         ema_factor=0.0625,         # overwritten per episode by randomize_controller
                         ema_prediction=True),      # Isaac smooths all action dims, a_ET included
        obs=ObsCfg(
            groups={
                "policy": list(POLICY_TERMS),
                "critic": list(CRITIC_TERMS),
                "dataset": POLICY_TERMS + ["peg_tip_rel_hole_gt", "hole_tip_gt", "success_gt", "joint_pos",
                                           "joint_vel"],
            },
            term_cfg={  # noise only in the `policy` group (ObsCfg.noisy_groups)
                "ee_pos_rel_anchor": ObsTermCfg(noise_std=POS_NOISE),
                "ee_quat_rel_nominal": ObsTermCfg(noise_std=QUAT_NOISE),
                "ee_vel": ObsTermCfg(noise_std=VEL_NOISE),
                "contact_force": ObsTermCfg(noise_std=FORCE_NOISE),
            },
        ),
        rewards={
            "kp_baseline": RewardTermCfg(weight=1.0, func="forge_kp_baseline", params={"a": 5.0, "b": 4.0}),
            "kp_coarse": RewardTermCfg(weight=1.0, func="forge_kp_coarse", params={"a": 50.0, "b": 2.0}),
            "kp_fine": RewardTermCfg(weight=1.0, func="forge_kp_fine", params={"a": 100.0, "b": 0.0}),
            "place_bonus": RewardTermCfg(weight=1.0, func="forge_place_bonus"),
            "success_bonus": RewardTermCfg(weight=1.0, func="forge_success_bonus"),
            "contact_penalty": RewardTermCfg(weight=-0.2, func="forge_contact_penalty"),
            # -|p - y| (Eq. 7), weight 1 as in the paper/Isaac, but zero until the running success rate
            # reaches `init.params.delay_until_ratio` (an untrained predictor would cancel the success bonus).
            "success_pred": RewardTermCfg(weight=-1.0, func="forge_success_pred_error"),
            # Isaac action penalties: ||s_t - s_{t-1}|| on the smoothed action, ||p_targ - p_ee|| / 2 cm.
            "action_grad": RewardTermCfg(weight=-0.1, func="applied_action_rate_norm"),
            "action_asset": RewardTermCfg(weight=-0.001, func="forge_action_penalty_asset",
                                          params={"lam_nominal": 0.02}),
        },
        terminations={
            "unstable": TerminationTermCfg(),
            "success": TerminationTermCfg(func="forge_success", success=True, ends_episode=False),
            # Deployment-style early stop (p > p_term). Off during training, as in the paper; enable for
            # eval with task.terminations.early_term.ends_episode=true.
            "early_term": TerminationTermCfg(func="forge_predicted_success", ends_episode=False,
                                             params={"p_term": 0.9}),
        },
        events={
            # Order matters: init (startup) -> socket pose -> threshold -> dynamics -> held offset -> EE pose.
            "init": EventTermCfg(mode="startup", func="forge_init",
                                 params={"peg": "peg", "hole": "hole", "place_xy": 0.0025, "success_dist": 0.001,
                                         "p_term": 0.9, "delay_until_ratio": 0.25, "gate_window": 750}),
            "reset_fixed": EventTermCfg(mode="reset", func="forge_reset_fixed",
                                        params={"lo": (0.55, -0.05, 0.0), "hi": (0.65, 0.05, 0.1),
                                                "pos_noise_std": 0.001}),  # Isaac 1 mm; paper 2.5 mm
            "reset_threshold": EventTermCfg(mode="reset", func="forge_sample_threshold",
                                            params={"lo": 5.0, "hi": 10.0}),
            "randomize_controller": EventTermCfg(
                mode="reset", func="forge_randomize_controller",
                params={"kp_default": (565.0, 565.0, 565.0, 28.0, 28.0, 28.0),
                        "kp_noise": (0.41, 0.41, 0.41, 0.41, 0.41, 0.41),
                        "lam_default": 0.02, "lam_noise": (0.25, 0.25, 0.25), "ema_range": (0.025, 0.1)}),
            "dead_zone": EventTermCfg(mode="reset", func="forge_randomize_dead_zone",
                                      params={"max_dz": (5.0, 5.0, 5.0, 1.0, 1.0, 1.0)}),
            "dead_zone_interval": EventTermCfg(mode="interval", interval_s=2.0, func="forge_randomize_dead_zone",
                                               params={"max_dz": (5.0, 5.0, 5.0, 1.0, 1.0, 1.0)}),
            "randomize_friction": EventTermCfg(mode="reset", func="forge_randomize_friction",
                                               params={"lo": 0.5, "hi": 1.0}),
            "randomize_held": EventTermCfg(mode="reset", func="forge_randomize_held",
                                           params={"lo": (-0.003, 0.0, -0.003), "hi": (0.003, 0.0, 0.003)}),
            "reset_ee": EventTermCfg(mode="reset", func="forge_reset_ee",
                                     params={"xy_range": 0.02, "z_range": (0.037, 0.057), "yaw_range": 0.785}),
            "update": EventTermCfg(mode="step", func="forge_update"),
        },
        episode_length_s=9.9,               # 150 policy steps at 15.15 Hz
        env_defaults={"decimation": 33},    # 2 ms x 33 = 66 ms policy step (paper: 15 Hz)
    )
