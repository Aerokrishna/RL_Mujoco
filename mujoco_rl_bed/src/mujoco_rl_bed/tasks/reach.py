"""`reach`: move the TCP to a random target position (cheap smoke test of the whole pipeline).

- Action: `delta_ee_pose` without rotation (3 dims, 2 cm per step); orientation held.
- Target: uniform in x∈[0.35, 0.65], y∈[-0.25, 0.25], z∈[0.15, 0.5] m (world = base frame),
  shown by a mocap marker.
- Reward: coarse + fine tanh closeness, small action-rate penalty.
- Episode: 5 s (100 policy steps at 20 Hz); success = within 1 cm (no early termination,
  so the policy learns to hold the target).
"""

from __future__ import annotations

from mujoco_rl_bed.control.cartesian_impedance import CartesianImpedanceCfg
from mujoco_rl_bed.env.cfg import ActionCfg, EventTermCfg, ObsCfg, RewardTermCfg, TaskCfg, TerminationTermCfg
from mujoco_rl_bed.sim.assets import TargetMarker
from mujoco_rl_bed.sim.scene import SceneCfg
from mujoco_rl_bed.tasks.registry import register_task

TARGET_LO = (0.35, -0.25, 0.15)
TARGET_HI = (0.65, 0.25, 0.5)
SUCCESS_THRESHOLD = 0.01  # [m]


@register_task("reach")
def reach() -> TaskCfg:
    """Return the reach task config.

    Returns:
        A new `TaskCfg`.
    """
    policy_terms = ["joint_pos_rel", "joint_vel", "ee_pos", "target_pos", "target_rel", "last_action"]
    return TaskCfg(
        scene=SceneCfg(assets=[TargetMarker(name="target")]),
        controller=CartesianImpedanceCfg(),
        action=ActionCfg(term="delta_ee_pose", pos_scale=0.02, rotation=False),
        obs=ObsCfg(groups={
            "policy": policy_terms,
            "critic": policy_terms + ["ee_vel"],
            "dataset": ["joint_pos", "joint_vel", "ee_pose", "ee_vel", "target_pos"],
        }),
        rewards={
            "ee_target_tanh": RewardTermCfg(weight=1.0, params={"std": 0.1}),
            "ee_target_tanh_fine": RewardTermCfg(weight=0.5, func="ee_target_tanh", params={"std": 0.02}),
            "action_rate_l2": RewardTermCfg(weight=-0.01),
        },
        terminations={
            "unstable": TerminationTermCfg(),
            # Reports success (TCP within 1 cm at episode end) without ending the episode.
            "success": TerminationTermCfg(func="ee_target_reached", success=True, ends_episode=False,
                                          params={"threshold": SUCCESS_THRESHOLD}),
        },
        events={
            "reset_joints_by_offset": EventTermCfg(mode="reset", params={"offset": 0.1}),
            "sample_target_pos": EventTermCfg(mode="reset", params={"lo": TARGET_LO, "hi": TARGET_HI,
                                                                    "marker": "target"}),
        },
        episode_length_s=5.0,
        params={"success_threshold": SUCCESS_THRESHOLD},
    )
