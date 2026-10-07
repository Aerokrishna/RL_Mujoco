"""forge_peg tests: geometry/contacts (scripted expert), reward kernels, observations, action clipping."""

from __future__ import annotations

import numpy as np
import pytest

import forge  # noqa: F401  (registers forge terms and tasks)
from forge.scripted import ScriptedPegInsert
from forge.terms import logistic_kernel
from mujoco_rl_bed.tasks.registry import make_env


def run_scripted(seed: int, xy_offset: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Run one scripted episode and return the final info.

    Args:
        seed: Episode seed.
        xy_offset: Lateral aim offset [m].

    Returns:
        Episode-end info dict.
    """
    env = make_env("forge_peg")
    pol = ScriptedPegInsert(env, xy_offset=xy_offset)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, te, tr, info = env.step(pol(obs))
        if te or tr:
            return info


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_expert_inserts(seed: int) -> None:
    """An aligned, slow insertion succeeds with low contact force (the hole fits the peg)."""
    info = run_scripted(seed)
    assert info["is_success"] and info["ever_success"]
    assert info["contact_force_max"] < 15.0  # step-averaged; brief contacts while descending reach ~10 N


@pytest.mark.parametrize("seed", [0, 1])
def test_misaligned_peg_jams(seed: int) -> None:
    """Aiming 1 mm off-axis (radial clearance is 0.25 mm) never inserts."""
    info = run_scripted(seed, xy_offset=(0.001, 0.0))
    assert not info["ever_success"]


def test_reset_state_and_observation_layout() -> None:
    """Reset places the peg above the hole without contact; obs = 20 dims, orientation ≈ nominal."""
    env = make_env("forge_peg")
    obs, _ = env.reset(seed=3)
    st = env.ctx.state["forge"]
    assert obs.shape == (21,) and env.action_space.shape == (4,)  # 3D position + success prediction
    assert st.force_norm == 0.0 and env.plant.data.ncon == 0
    assert 5.0 <= st.f_th <= 10.0 and obs[16] == pytest.approx(st.f_th, rel=1e-6)
    np.testing.assert_allclose(obs[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-3)  # quat rel. nominal pose
    rel = obs[0:3]  # peg tip relative to the hole opening (anchor = hole tip + TCP-to-tip offset)
    assert np.all(np.abs(rel[:2]) <= 0.0201) and 0.0019 <= rel[2] <= 0.0221
    np.testing.assert_allclose(rel, env.plant.data.site_xpos[st.peg_tip] - st.hole_tip, atol=1e-6)
    assert st.ik_err_max == 0.0


def test_zero_action_hovers_at_opening() -> None:
    """Zero action brings the peg tip to the hole opening without ramming the block."""
    env = make_env("forge_peg")
    env.reset(seed=5)
    st = env.ctx.state["forge"]
    for _ in range(60):
        env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
    tip = env.plant.data.site_xpos[st.peg_tip]
    assert np.linalg.norm(tip - st.hole_tip) < 0.003
    assert st.force_norm < 5.0


def test_penalty_uses_step_averaged_force() -> None:
    """The penalty's force equals the step-averaged contact force the policy observes (not one tick)."""
    env = make_env("forge_peg")
    env.reset(seed=6)
    st = env.ctx.state["forge"]
    sl = dict(env.obs_mgr.layout["policy"])["contact_force"]
    seen_contact = False
    for _ in range(40):
        obs, *_ = env.step(np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32))  # press down onto the rim/hole
        np.testing.assert_allclose(obs[sl], st.force_mean, rtol=1e-5, atol=1e-4)
        seen_contact |= st.force_mean_norm > 0.0
    assert seen_contact


def test_action_clipped_within_lambda() -> None:
    """The position target never leaves the λ box around the TCP."""
    env = make_env("forge_peg")
    env.reset(seed=4)
    lam = env.cfg.task.action.max_step
    for a in ([-1, -1, -1, -1], [1, 1, 1, 1], [0.3, -0.7, -1, 0.2]):
        ee = env.plant.ee_pos.copy()
        env.action_mgr.apply(np.asarray(a, dtype=np.float32))
        assert np.all(np.abs(env.controller.pos_d - ee) <= lam + 1e-12)


def test_logistic_kernel_values() -> None:
    """K_{a,b}(0) = 1 / (2 + b) and decreases with distance."""
    assert logistic_kernel(0.0, 50.0, 2.0) == pytest.approx(0.25)
    assert logistic_kernel(0.0, 100.0, 0.0) == pytest.approx(0.5)
    assert logistic_kernel(0.01, 100.0, 0.0) < logistic_kernel(0.001, 100.0, 0.0)
    assert logistic_kernel(10.0, 100.0, 0.0) >= 0.0  # no overflow far away


def test_asymmetric_critic_terms() -> None:
    """Critic block is clean ground truth: true peg-tip offset, estimate error, success label."""
    env = make_env("forge_peg", {"obs_mode": "asymmetric"})
    obs, _ = env.reset(seed=7)
    p = env.policy_obs_dim
    lay = dict(env.obs_mgr.layout["critic"])
    crit = obs[p:]
    assert p == 21 and obs.shape == (p + env.obs_mgr.dim("critic"),)
    assert "ee_pos_rel_anchor" not in lay  # the noisy-relative position is not in the critic block
    np.testing.assert_allclose(crit[lay["anchor_error_gt"]], 0.0, atol=1e-6)
    np.testing.assert_allclose(crit[lay["peg_tip_rel_hole_gt"]], obs[0:3], atol=1e-5)  # no noise: same as actor's
    assert crit[lay["success_gt"]][0] == 0.0

    noisy = make_env("forge_peg", {"obs_mode": "asymmetric", "task.events.reset_fixed.params.pos_noise_std": "0.0025"})
    obs, _ = noisy.reset(seed=7)
    st = noisy.ctx.state["forge"]
    crit = obs[p:]
    err = crit[lay["anchor_error_gt"]]
    assert np.linalg.norm(err) > 1e-4
    # actor sees peg tip relative to the noisy estimate; true offset = actor view + estimate error
    np.testing.assert_allclose(crit[lay["peg_tip_rel_hole_gt"]], obs[0:3] + err, atol=1e-5)
    np.testing.assert_allclose(err, st.anchor - st.hole_tip - [0, 0, st.tcp_to_tip], atol=1e-6)


def test_success_prediction_perfect_predictor() -> None:
    """With a perfect a_ET (scripted expert), the prediction penalty is ~0 and ET metrics are ideal."""
    info = run_scripted(0)
    assert info["is_success"] and info["et_triggered"] and info["et_correct"] and info["et_recall_hit"]
    assert 0.0 <= info["et_delay"] <= 0.1  # predicts success within about one policy step of achieving it
    assert info["reward_terms"]["success_pred"] > -1.5  # at most ~1 step of lag over 150 steps


def test_success_prediction_penalty_and_early_termination() -> None:
    """Predicting success while unsolved is penalized; with ends_episode=true it terminates the episode."""
    env = make_env("forge_peg")
    env.reset(seed=8)
    _, r_yes, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))   # p = 1, not solved
    env.reset(seed=8)
    _, r_no, *_ = env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))   # p = 0, not solved
    w = -env.cfg.task.rewards["success_pred"].weight
    assert r_no - r_yes == pytest.approx(w, abs=1e-6)  # -w |p - y|

    ev = make_env("forge_peg", {"task.terminations.early_term.ends_episode": "true"})
    ev.reset(seed=8)
    _, _, term, trunc, info = ev.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert term and not trunc and info["termination"] == "early_term" and info["et_triggered"]
    assert not info["et_correct"] and not info["is_success"]  # a false-positive early stop


def test_action_smoothing_suppresses_chatter() -> None:
    """With EMA smoothing, a policy flipping x between -1 and +1 barely moves the target; without, it jumps."""
    def target_swing(ema: str | None) -> float:
        env = make_env("forge_peg", {"task.action.ema_factor": ema} if ema else None)
        env.reset(seed=9)
        xs = []
        for k in range(20):
            sgn = 1.0 if k % 2 == 0 else -1.0
            env.step(np.array([sgn, 0.0, 0.0, -1.0], dtype=np.float32))
            xs.append(env.controller.pos_d[0])
        return float(np.ptp(xs[10:]))  # steady-state peak-to-peak target swing [m]

    smooth, raw = target_swing(None), target_swing("1.0")  # task default vs. no smoothing
    assert raw > 0.02 and smooth < 0.4 * raw


def test_applied_action_and_action_rate_penalty() -> None:
    """last_action observes the smoothed action; the action-rate penalty is 0 on step 1 and > 0 on a flip."""
    env = make_env("forge_peg")
    env.reset(seed=10)
    lay = dict(env.obs_mgr.layout["policy"])
    a1 = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    obs, *_, info = env.step(a1)
    assert env.reward_mgr._sums[env.reward_mgr.names.index("action_rate")] == 0.0  # no history on step 1
    alpha = env.cfg.task.action.ema_factor
    applied = obs[lay["last_action"]]
    assert applied[3] == pytest.approx(-1.0)  # success prediction is not smoothed
    assert abs(applied[0]) < 1.0 and applied[0] == pytest.approx(env.ctx.applied_action[0], abs=1e-6)
    env.step(-a1)
    i = env.reward_mgr.names.index("action_rate")
    w = env.cfg.task.rewards["action_rate"].weight
    assert env.reward_mgr._sums[i] == pytest.approx(w * (4.0 + 0.0 + 0.0 + 4.0), abs=1e-6)  # ||-a1 - a1||^2 = 8
    assert 0.0 < alpha < 1.0


def test_controller_randomization() -> None:
    """Kp and λ are resampled each episode within the paper's ranges, applied, and visible to the critic only."""
    env = make_env("forge_peg", {"obs_mode": "asymmetric"})
    st = env.ctx.state["forge"]
    seen = []
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        c = env.controller
        assert 400.0 <= st.kp <= 800.0 and 0.016 <= st.lam <= 0.025
        np.testing.assert_allclose(c.kp[:3], st.kp)                       # applied by the controller reset
        np.testing.assert_allclose(c.kd[:3], 2.0 * np.sqrt(st.kp))         # critical damping follows
        assert env.ctx.state["action_max_step"][0] == st.lam               # λ used by the action term
        crit = obs[env.policy_obs_dim:]
        lay = dict(env.obs_mgr.layout["critic"])
        np.testing.assert_allclose(crit[lay["controller_params_gt"]], [st.kp, st.lam], rtol=1e-5)
        assert "controller_params_gt" not in dict(env.obs_mgr.layout["policy"])
        seen.append(st.kp)
    assert np.ptp(seen) > 50.0  # actually varies between episodes
    env.ctx.state["action_max_step"][0] = st.lam
    ee = env.plant.ee_pos.copy()
    env.action_mgr.apply(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32))
    assert np.all(np.abs(env.controller.pos_d - ee) <= st.lam + 1e-9)  # clip uses the sampled λ
