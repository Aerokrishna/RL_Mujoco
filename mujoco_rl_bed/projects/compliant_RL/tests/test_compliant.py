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
    """Push force covers [1, 15] N; all-zero action is the nominal 8 N push."""
    env = make_env("compliant_peg", NO_DR)
    env.reset(seed=0)
    c = env.ctx.controller
    env.step(np.zeros(5))
    assert c.f_push == 8.0 and np.all(c.f_lat == 0.0) and np.all(c.tilt == 0.0)
    for _ in range(30):
        env.step(np.array([1.0, -1.0, 1.0, 1.0, -1.0]))
    np.testing.assert_allclose(c.f_lat, [5.0, -5.0], atol=1e-6)
    np.testing.assert_allclose(np.degrees(c.tilt), [10.0, -10.0], atol=1e-4)
    assert abs(c.f_push - 15.0) < 1e-6
    for _ in range(30):
        env.step(np.array([0.0, 0.0, -1.0, 0.0, 0.0]))
    assert abs(c.f_push - 1.0) < 1e-6


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
