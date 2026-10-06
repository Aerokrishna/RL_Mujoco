"""`forge_peg`: first-pass FORGE-style 8 mm round peg insertion (arXiv:2408.04587).

Values follow the paper's Table II (8 mm peg) unless noted:
- Parts: 8 mm peg rigidly attached to the hand (v1: no grasp slip or in-hand offset),
  socket = solid 6x6 cm block with a Ø8.5 mm hole (0.25 mm radial / 0.5 mm diametrical clearance,
  as in the paper), 25 mm deep,
  top flush with the hole opening. Success = peg tip
  within 1 mm of the hole floor and centered.
- Initial state: socket base x∈[0.55, 0.65], y∈[-0.05, 0.05], z∈[0, 0.1] m; TCP at hole tip
  + (U±2 cm, U±2 cm, U[3.7, 5.7] cm), gripper pointing down (IK).
- Action: 3D TCP target = anchor + a * 5 cm, where the anchor is the TCP pose with the peg tip at
  the (estimated) hole opening (zero action = peg tip at the opening). Clipped per axis to
  within λ = 2 cm of the TCP (Eq. 5). Kp = 600 N/m (constant, mid of [400, 800]),
  Kd = 2 sqrt(Kp). Orientation held.
- Observation (symmetric default: actor and critic see the same vector; with `asymmetric=true` the
  critic additionally gets the true peg-tip offset and the hole-estimate error, as in the paper):
  peg-tip pos rel. hole opening (3, via the anchor), TCP quat rel. nominal gripper-down pose (4), TCP twist (6), contact force (3, step mean),
  F_th (1), previous action incl. a_ET (4) = 21. Critic (asymmetric=true): clean state, see CRITIC_TERMS.
- Reward: K_coarse(50, 2) + K_fine(100, 0) + I_place + I_success - 0.2 max(0, ||F|| - F_th),
  F_th ~ U[5, 10] N per episode.
- Smoothing (not in the paper): EMA on the position action (alpha 0.2) and an action-rate penalty
  (-0.02 ||a_t - a_t-1||^2) against chatter.
- Success prediction: 4th action a_ET -> p in [0, 1], reward -0.1 |p - y_t| (Eq. 7); early stop at p > 0.9
  only when enabled (evaluation).
- Episode: 150 policy steps (paper's step count; 7.5 s at 20 Hz). No early termination in training;
  `is_success` reports success at the final step and `ever_success` at any step.
- Not yet: EE/force observation noise (the hooks exist, set to 0), dynamics randomization (Kp, λ,
  friction, dead zone). Hole-position noise is opt-in via task.events.reset_fixed.params.pos_noise_std.
"""

from __future__ import annotations

from mujoco_rl_bed.control.cartesian_impedance import CartesianImpedanceCfg
from mujoco_rl_bed.env.cfg import (ActionCfg, EventTermCfg, ObsCfg, RewardTermCfg, TaskCfg,
                                   TerminationTermCfg)
from mujoco_rl_bed.sim.assets import Peg, RoundHole
from mujoco_rl_bed.sim.scene import SceneCfg
from mujoco_rl_bed.tasks.registry import register_task

POLICY_TERMS = ["ee_pos_rel_anchor", "ee_quat_rel_nominal", "ee_vel", "contact_force", "force_threshold", "last_action"]
# Clean critic state (asymmetric=true): ground-truth peg-tip offset instead of the noisy-relative
# position, the actor's other (currently noise-free) terms, the hole-estimate error (explains what the
# actor believes) and the true success label. The critic does not see the actor block.
CRITIC_TERMS = ["peg_tip_rel_hole_gt", "ee_quat_rel_nominal", "ee_vel", "contact_force", "force_threshold",
                "last_action", "anchor_error_gt", "success_gt"]


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
        controller=CartesianImpedanceCfg(kp=(600.0, 600.0, 600.0, 100.0, 100.0, 100.0)),
        action=ActionCfg(term="anchor_relative_pos", anchor="fixed_anchor", anchor_bounds=(0.05, 0.05, 0.05),
                         max_step=0.02, pos_lo=(0.3, -0.3, 0.0), pos_hi=(0.85, 0.3, 0.5),
                         success_prediction=True,   # 4th action dim a_ET -> predicted success p
                         ema_factor=0.2),           # action smoothing (not in the paper): -76% TCP jerk vs. raw
        obs=ObsCfg(groups={
            "policy": list(POLICY_TERMS),
            "critic": list(CRITIC_TERMS),
            "dataset": POLICY_TERMS + ["peg_tip_rel_hole_gt", "hole_tip_gt", "success_gt", "joint_pos", "joint_vel"],
        }),
        rewards={
            "kp_coarse": RewardTermCfg(weight=1.0, func="forge_kp_coarse", params={"a": 50.0, "b": 2.0}),
            "kp_fine": RewardTermCfg(weight=1.0, func="forge_kp_fine", params={"a": 100.0, "b": 0.0}),
            "place_bonus": RewardTermCfg(weight=1.0, func="forge_place_bonus"),
            "success_bonus": RewardTermCfg(weight=1.0, func="forge_success_bonus"),
            "contact_penalty": RewardTermCfg(weight=-0.2, func="forge_contact_penalty"),
            # -|p - y| (Eq. 7). Weight 0.1: at 1.0 an untrained predictor (always p=0) cancels the success
            # bonus and the policy learns to stop short of full insertion.
            "success_pred": RewardTermCfg(weight=-0.1, func="forge_success_pred_error"),
            # Not in the paper: penalize step-to-step action changes (chatter / vibration).
            "action_rate": RewardTermCfg(weight=-0.02, func="action_rate_l2"),
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
            # Order matters: init (startup) -> socket pose -> threshold -> EE pose (needs the socket).
            "init": EventTermCfg(mode="startup", func="forge_init",
                                 params={"peg": "peg", "hole": "hole", "place_xy": 0.0025, "success_dist": 0.001,
                                         "p_term": 0.9}),
            "reset_fixed": EventTermCfg(mode="reset", func="forge_reset_fixed",
                                        params={"lo": (0.55, -0.05, 0.0), "hi": (0.65, 0.05, 0.1),
                                                "pos_noise_std": 0.0}),
            "reset_threshold": EventTermCfg(mode="reset", func="forge_sample_threshold",
                                            params={"lo": 5.0, "hi": 10.0}),
            "reset_ee": EventTermCfg(mode="reset", func="forge_reset_ee",
                                     params={"xy_range": 0.02, "z_range": (0.037, 0.057)}),
            "update": EventTermCfg(mode="step", func="forge_update"),
        },
        episode_length_s=7.5,
    )
