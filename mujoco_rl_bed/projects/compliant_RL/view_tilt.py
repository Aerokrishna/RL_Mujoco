"""Watch the axis compliance controller alone (zero residual) insert a peg that starts tilted.

Each episode the gripper is tilted about the peg tip by a random angle in [0, tilt_max] (random
horizontal axis, no yaw), so the tip starts exactly over the hole: no xy offset, no hole-estimate error,
no grasp tilt. The controller pushes with `force` N along the motion axis, is soft in x/y and roll/pitch
and stiff in yaw (gains from `compliant_peg`).

Usage (`conda activate dqn` first; works from any directory):
    python projects/compliant_RL/view_tilt.py                         # 15 episodes, 5 N, up to 10°
    python projects/compliant_RL/view_tilt.py episodes=5 tilt_max=5 force=8
    python projects/compliant_RL/view_tilt.py render=false            # headless, print the table only
    python projects/compliant_RL/view_tilt.py tilt_max=0 offset=0.5   # no tilt, tip starts 0.5 mm off the hole axis

The reference is the tilted start pose: the push is along the tilted peg (tool) axis, the soft tilt
spring holds the start tilt and only contact torques straighten the peg.

`offset` [mm] shifts the gripper sideways (world xy, random direction) by U(offset_min, offset) after the tilt;
offset_min defaults to offset (fixed magnitude).

Other keys: seed (0), realtime (true), episode_s (8). Close the viewer window to stop early.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # projects/ (for `compliant_RL`, `forge`)

import compliant_RL  # noqa: E402,F401  (registers terms and tasks)
from compliant_RL.terms import peg_axis_angle  # noqa: E402
from mujoco_rl_bed.env.cfg import EventTermCfg  # noqa: E402
from mujoco_rl_bed.env.torque_env import TorqueEnv  # noqa: E402
from mujoco_rl_bed.tasks.registry import make_env_cfg  # noqa: E402
from mujoco_rl_bed.utils.config import parse_cli  # noqa: E402


def make_tilt_env(force: float, tilt_max: float, episode_s: float, render: bool, realtime: bool,
                  offset: float = 0.0, offset_min: float | None = None) -> TorqueEnv:
    """`compliant_peg` with the start tilt and shift events (offsets in m), all other randomization off,
    nominal push = `force`."""
    ov = {"task.events.reset_fixed.params.pos_noise_std": "0",
          "task.events.randomize_grasp.params.tilt_max_deg": "0",
          "task.events.reset_ee.params.xy_jitter": "0",
          "task.events.randomize_friction.params.lo": "0.75", "task.events.randomize_friction.params.hi": "0.75",
          "task.action.params.f_push_nominal": str(force),
          "task.episode_length_s": str(episode_s),
          "render": str(render).lower(), "realtime": str(realtime).lower()}
    cfg = make_env_cfg("compliant_peg", ov)
    events = {}
    for k, v in cfg.task.events.items():
        events[k] = v
        if k == "reset_ee":
            events["tilt_ee"] = EventTermCfg(mode="reset", func="cr_tilt_ee", params={"tilt_max_deg": tilt_max})
            events["shift_ee"] = EventTermCfg(mode="reset", func="cr_shift_ee", params={
                "offset_min": offset if offset_min is None else offset_min, "offset_max": offset})
    cfg.task.events = events
    return TorqueEnv(cfg)


def main(argv: list[str]) -> None:
    """Run the episodes and print one line per episode plus a summary.

    Args:
        argv: key=value arguments.
    """
    a = parse_cli(argv)
    episodes, seed = int(a.get("episodes", 15)), int(a.get("seed", 0))
    force, tilt_max = float(a.get("force", 5.0)), float(a.get("tilt_max", 10.0))
    render = str(a.get("render", "true")).lower() == "true"
    offset = float(a.get("offset", 0.0))
    offset_min = float(a.get("offset_min", offset))
    env = make_tilt_env(force, tilt_max, float(a.get("episode_s", 8.0)), render,
                        str(a.get("realtime", "true")).lower() == "true", 1e-3 * offset, 1e-3 * offset_min)
    st, c, d = env.ctx.state["forge"], env.ctx.controller, env.ctx.data
    zero = np.zeros(env.action_space.shape[0])
    print(f"axis compliance only: push {force:g} N along the tool axis, start tilt <= {tilt_max:g} deg, "
          f"start offset {offset_min:g}-{offset:g} mm, "
          f"k_lat {c.cfg.k_lat:g} N/m, d_lat {c.cfg.d_lat:g} Ns/m, k_tilt {c.cfg.k_tilt:g} Nm/rad, k_yaw {c.cfg.k_yaw:g} Nm/rad")
    print(f"{'ep':>3} {'off0':>7} {'tilt0':>7} {'tilt_end':>8} {'xy_err':>7} {'depth':>6} {'F_max':>6} {'t_ins':>6}  result")
    n_ok = 0
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        tilt0 = math.degrees(float(np.linalg.norm(env.ctx.state["start_tilt"])))
        off0 = 1e3 * float(np.linalg.norm(d.site_xpos[st.peg_tip][:2] - st.hole_tip[:2]))
        z_top = d.site_xpos[st.peg_tip][2]
        t_ins, k = None, 0
        while True:
            _, _, term, trunc, info = env.step(zero)
            k += 1
            if t_ins is None and st.success:
                t_ins = k * env.policy_dt
            if term or trunc or (render and not env._viewer.is_running()):
                break
        tip = d.site_xpos[st.peg_tip]
        xy = 1e3 * float(np.linalg.norm(tip[:2] - st.hole_tip[:2]))
        depth = 1e3 * (st.hole_tip[2] - tip[2])
        ok = t_ins is not None
        n_ok += ok
        print(f"{ep:>3} {off0:>5.2f}mm {tilt0:>6.1f}° {math.degrees(peg_axis_angle(env.ctx)):>7.1f}° {xy:>5.2f}mm {depth:>4.1f}mm "
              f"{info.get('contact_force_max', float('nan')):>5.1f}N "
              f"{(f'{t_ins:.2f}s' if ok else '-'):>6}  {'inserted' if ok else 'FAILED'}")
        if render and not env._viewer.is_running():
            break
    print(f"inserted {n_ok}/{ep + 1}  (start tip height above hole: {1e3 * (z_top - st.hole_tip[2]):.1f} mm)")
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
