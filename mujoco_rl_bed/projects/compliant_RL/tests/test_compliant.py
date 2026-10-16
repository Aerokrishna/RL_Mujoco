"""compliant_peg tests: axis compliance command frame, base controller insertion without randomization,
privileged scripted corrector with randomization, action mapping."""

from __future__ import annotations

import numpy as np

import compliant_RL  # noqa: F401  (registers terms and tasks)
from compliant_RL.scripted import ScriptedCorrector
from mujoco_rl_bed.control.axis_compliance import AxisCompliance
from mujoco_rl_bed.tasks.registry import make_env

NO_DR = {"task.events.reset_fixed.params.pos_noise_std": "0", "task.events.randomize_grasp.params.tilt_max_deg": "0",
         "task.events.reset_ee.params.xy_jitter": "0"}


def run(env, policy=None, seed: int = 0) -> dict:
    """Run one episode (zero residual unless `policy`) and return the final info with `ever_success`."""
    env.reset(seed=seed)
    if policy is not None:
        policy.reset()
    ever = False
    while True:
        a = np.zeros(5) if policy is None else policy.act()
        _, _, term, trunc, info = env.step(a)
        ever |= bool(info.get("is_success"))
        if term or trunc:
            return {**info, "ever_success": ever}


def test_axis_frame():
    """A maps e_z to the motion axis and is a rotation."""
    for axis in [(0, 0, 1), (1, 0, 0), (0.3, -0.4, 0.8), (0, 0, -1)]:
        A = AxisCompliance.axis_frame(np.array(axis, dtype=float))
        np.testing.assert_allclose(A[:, 2], np.array(axis) / np.linalg.norm(axis), atol=1e-12)
        np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-12)
        assert np.linalg.det(A) > 0.0
    np.testing.assert_allclose(AxisCompliance.axis_frame(np.array([0.0, 0.0, 1.0])), np.eye(3))


def test_command_frame_fixed_under_tilt():
    """A commanded tilt rotates the tool but not the command frame: free-space motion stays on the start axis."""
    env = make_env("compliant_peg", NO_DR)
    env.reset(seed=0)
    c = env.ctx.controller
    R0 = c.R_f.copy()
    p0 = env.ctx.plant.ee_pos.copy()
    for _ in range(3):
        env.step(np.array([0.0, 0.0, 0.0, 1.0, 0.0]))     # tilt 10° about x, still in free space
    np.testing.assert_allclose(c.R_f, R0)
    d = env.ctx.plant.ee_pos - p0
    lateral = np.linalg.norm(d - (d @ R0[:, 2]) * R0[:, 2])
    assert d @ R0[:, 2] > 0.002 and lateral < 0.001


def test_action_mapping():
    """Side force ±10 N, push 10 ± 7 N clipped to 15 N, tilt ±10°; all-zero action is the nominal 10 N push."""
    env = make_env("compliant_peg", NO_DR)
    env.reset(seed=0)
    c = env.ctx.controller
    env.step(np.zeros(5))
    assert c.f_push == 10.0 and np.all(c.f_lat == 0.0) and np.all(c.tilt == 0.0)
    for _ in range(30):
        env.step(np.array([1.0, -1.0, 1.0, 1.0, -1.0]))
    np.testing.assert_allclose(c.f_lat, [10.0, -10.0], atol=1e-6)
    np.testing.assert_allclose(np.degrees(c.tilt), [10.0, -10.0], atol=1e-4)
    assert abs(c.f_push - 15.0) < 1e-6
    for _ in range(30):
        env.step(np.array([0.0, 0.0, -1.0, 0.0, 0.0]))
    assert abs(c.f_push - 3.0) < 1e-6


def test_base_controller_inserts_without_randomization():
    """With the hole estimate exact and no grasp tilt, the base controller (zero residual) inserts."""
    env = make_env("compliant_peg", NO_DR)
    info = run(env, seed=1)
    assert info["ever_success"] and info["contact_force_max"] < 15.0


def test_scripted_corrector_with_randomization():
    """The privileged corrector solves randomized episodes through the 5D residual action space."""
    env = make_env("compliant_peg")
    pol = ScriptedCorrector(env)
    assert all(run(env, pol, seed=s)["ever_success"] for s in range(3))


def test_s2f_zero_action_is_spiral():
    """`compliant_peg_s2f` with zero action follows the `compliant_peg_s2b` spiral at 1.5 N."""
    env = make_env("compliant_peg_s2f")
    ref = make_env("compliant_peg_s2b", {"task.action.params.f_search": "1.5"})
    env.reset(seed=3)
    ref.reset(seed=3)
    for _ in range(100):
        env.step(np.zeros(3))
        ref.step(np.zeros(3))
    np.testing.assert_allclose(env.plant.q, ref.plant.q, atol=1e-6)


def test_s2f_try_action():
    """A full try in the search phase pauses the spiral, switches to the task gains and pushes f_try."""
    env = make_env("compliant_peg_s2f")
    env.reset(seed=3)
    c, sp, term = env.ctx.controller, env.ctx.state["spiral"], env.ctx.state["spiral_term"]
    while sp["phase"] != 1:
        env.step(np.zeros(3))
    off = sp["offset"].copy()
    env.step(np.array([0.0, 0.0, 1.0]))
    env.step(np.array([0.0, 0.0, 1.0]))
    assert sp["tau"] == 1.0 and sp["try_t"] > 0.0
    np.testing.assert_allclose(sp["offset"], off)
    np.testing.assert_allclose((c.k_lin[0], c.d_lin[0], c.k_rot[0], c.d_rot[0]), term.task_gains, rtol=1e-9)
    assert c.f_push == term.f_try
    env.step(np.zeros(3))
    assert sp["tau"] == 0.0 and c.k_lin[0] == term.search_gains[0]


def _yaw_deg(env) -> float:
    """Peg yaw about world z [deg]."""
    import mujoco

    m, d = env.ctx.model, env.ctx.data
    g = d.geom_xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")]
    return float(np.degrees(np.arctan2(g[3], g[0])))


def test_s3_trigger_starts_local_phase_and_yaw_target_turns_peg():
    """spiral_local: a trigger in the search phase freezes the spiral; the yaw action turns the peg (sign: the tool
    axis points down, so +yaw command = -yaw about world z)."""
    env = make_env("compliant_peg_s3", {"task.events.hole_yaw_error.params.hi_deg": "0"})
    sp = env.ctx.state["spiral"]
    env.reset(seed=1)
    for t in range(110):
        if t == 50:
            sp["trigger"] = True
        env.step(np.array([0.0, 0.0, 0.5 if t >= 50 else 0.0]))
        if t == 49:
            assert sp["phase"] == 1 and sp["n_local"] == 0
            yaw0 = _yaw_deg(env)
    assert sp["phase"] == 3 and sp["n_local"] == 1
    assert abs(np.degrees(sp["yaw_cmd"]) - 6.0) < 1e-6
    assert abs(_yaw_deg(env) - yaw0 + 6.0) < 0.2


def test_s3_local_oracle_inserts():
    """compliant_peg_s3_local: the privileged LocalOracle corrects offset and yaw and inserts; zero action rarely."""
    from compliant_RL.scripted import LocalOracle

    env = make_env("compliant_peg_s3_local", {"task.events.shift_ee.params.offset_max": "0.0012",
                                              "task.events.hole_yaw_error.params.lo_deg": "8.0",
                                              "task.events.hole_yaw_error.params.hi_deg": "8.0"})
    st, sp = env.ctx.state["forge"], env.ctx.state["spiral"]
    orc = LocalOracle(env)
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        orc.reset()
        while True:
            obs, _, term, trunc, _ = env.step(orc(obs))
            if term or trunc:
                break
        assert sp["n_local"] == 1
        assert st.first_success_step >= 0, f"seed {seed}"


def test_hole_trigger_loads_and_reads_trigger_group():
    """The shipped hole-trigger classifier loads and scores the `trigger` observation group of compliant_peg_s3."""
    from compliant_RL.hole_trigger import HoleTrigger

    env = make_env("compliant_peg_s3", {})
    env.reset(seed=0)
    trig = HoleTrigger()
    for _ in range(5):
        env.step(np.zeros(3))
        fired = trig(env.compute_obs("trigger"))
        assert isinstance(fired, bool) and 0.0 <= trig.p <= 1.0


def test_s3_handoff_in_env_trigger_and_spaces_match_s3_local():
    """compliant_peg_s3_handoff: the in-env hole trigger starts local phases during the spiral; its spaces match
    compliant_peg_s3_local, so a run trained there can be fine-tuned here."""
    env = make_env("compliant_peg_s3_handoff", {"obs_mode": "asymmetric"})
    ref = make_env("compliant_peg_s3_local", {"obs_mode": "asymmetric"})
    assert env.observation_space.shape == ref.observation_space.shape
    assert env.action_space.shape == ref.action_space.shape and env.policy_obs_dim == ref.policy_obs_dim
    sp = env.ctx.state["spiral"]
    n_local = 0
    for seed in range(4):
        env.reset(seed=seed)
        for _ in range(300):
            _, _, term, trunc, _ = env.step(np.zeros(3))
            if term or trunc or sp["phase"] == 2:
                break
        n_local += sp["n_local"]
    assert n_local >= 1
