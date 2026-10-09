"""forge_peg tests: geometry/contacts (scripted expert), reward kernels, observations, action clipping,
dynamics randomization (Isaac Lab FORGE style), observation noise and the delayed prediction penalty."""

from __future__ import annotations

import numpy as np
import pytest

import forge  # noqa: F401  (registers forge terms and tasks)
from forge.scripted import ScriptedPegInsert
from forge.task import NO_DR
from forge.terms import logistic_kernel
from mujoco_rl_bed.tasks.registry import make_env


def make(overrides: dict | None = None, dr: bool = False):
    """forge_peg env, by default without dynamics randomization / observation noise (`NO_DR`).

    Args:
        overrides: Extra overrides (applied last).
        dr: Keep the task's randomization and noise.

    Returns:
        The env.
    """
    return make_env("forge_peg", {**({} if dr else NO_DR), **(overrides or {})})


def run_scripted(seed: int, xy_offset: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Run one scripted episode and return the final info.

    Args:
        seed: Episode seed.
        xy_offset: Lateral aim offset [m].

    Returns:
        Episode-end info dict.
    """
    env = make()
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
    """Reset places the peg above the hole without contact; obs = 21 dims, orientation ≈ nominal."""
    env = make({"task.events.reset_fixed.params.pos_noise_std": "0.0"})
    obs, _ = env.reset(seed=3)
    st = env.ctx.state["forge"]
    assert obs.shape == (21,) and env.action_space.shape == (4,)  # 3D position + success prediction
    assert st.force_norm == 0.0 and env.plant.data.ncon == 0
    assert 5.0 <= st.f_th <= 10.0 and obs[16] == pytest.approx(st.f_th, rel=1e-6)
    np.testing.assert_allclose(obs[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-3)  # quat rel. nominal pose
    rel = obs[0:3]  # peg tip relative to the hole opening (anchor = hole tip + TCP-to-tip offset)
    assert np.all(np.abs(rel[:2]) <= 0.0201) and 0.0009 <= rel[2] <= 0.0221
    np.testing.assert_allclose(rel, env.plant.data.site_xpos[st.peg_tip] - st.hole_tip, atol=1e-6)
    assert st.ik_err_max == 0.0


def test_zero_action_hovers_at_opening() -> None:
    """Zero action brings the peg tip to the hole opening without ramming the block."""
    env = make({"task.events.reset_fixed.params.pos_noise_std": "0.0"})
    env.reset(seed=5)
    st = env.ctx.state["forge"]
    for _ in range(45):
        env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
    tip = env.plant.data.site_xpos[st.peg_tip]
    assert np.linalg.norm(tip - st.hole_tip) < 0.003
    assert st.force_norm < 5.0


def test_penalty_uses_step_averaged_force() -> None:
    """The penalty's force equals the step-averaged contact force the policy observes (not one tick)."""
    env = make()
    env.reset(seed=6)
    st = env.ctx.state["forge"]
    sl = dict(env.obs_mgr.layout["policy"])["contact_force"]
    seen_contact = False
    for _ in range(30):
        obs, *_ = env.step(np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32))  # press down onto the rim/hole
        np.testing.assert_allclose(obs[sl], st.force_mean, rtol=1e-5, atol=1e-4)
        seen_contact |= st.force_mean_norm > 0.0
    assert seen_contact


def test_action_clipped_within_lambda() -> None:
    """The position target never leaves the per-axis λ box around the TCP."""
    env = make(dr=True)
    env.reset(seed=4)
    lam = env.ctx.state["action_max_step"]
    assert lam.shape == (3,) and np.ptp(lam) > 0.0  # randomized per axis
    for a in ([-1, -1, -1, -1], [1, 1, 1, 1], [0.3, -0.7, -1, 0.2]):
        ee = env.plant.ee_pos.copy()
        env.action_mgr.apply(np.asarray(a, dtype=np.float32))
        assert np.all(np.abs(env.controller.pos_d - ee) <= lam + 1e-12)
        assert np.all(np.abs(env.ctx.state["action_target_delta"]) <= 0.1 + 0.05)  # unclipped target - TCP


def test_logistic_kernel_values() -> None:
    """K_{a,b}(0) = 1 / (2 + b) and decreases with distance."""
    assert logistic_kernel(0.0, 50.0, 2.0) == pytest.approx(0.25)
    assert logistic_kernel(0.0, 5.0, 4.0) == pytest.approx(1.0 / 6.0)  # Isaac baseline kernel
    assert logistic_kernel(0.0, 100.0, 0.0) == pytest.approx(0.5)
    assert logistic_kernel(0.01, 100.0, 0.0) < logistic_kernel(0.001, 100.0, 0.0)
    assert logistic_kernel(10.0, 100.0, 0.0) >= 0.0  # no overflow far away


def test_asymmetric_critic_terms() -> None:
    """Critic block is clean ground truth: true peg-tip offset, estimate error, success label."""
    env = make({"obs_mode": "asymmetric", "task.events.reset_fixed.params.pos_noise_std": "0.0"})
    obs, _ = env.reset(seed=7)
    p = env.policy_obs_dim
    lay = dict(env.obs_mgr.layout["critic"])
    crit = obs[p:]
    assert p == 21 and obs.shape == (p + env.obs_mgr.dim("critic"),)
    assert "ee_pos_rel_anchor" not in lay  # the noisy-relative position is not in the critic block
    np.testing.assert_allclose(crit[lay["anchor_error_gt"]], 0.0, atol=1e-6)
    np.testing.assert_allclose(crit[lay["peg_tip_rel_hole_gt"]], obs[0:3], atol=1e-5)  # no noise: same as actor's
    assert crit[lay["success_gt"]][0] == 0.0

    noisy = make({"obs_mode": "asymmetric", "task.events.reset_fixed.params.pos_noise_std": "0.0025"})
    obs, _ = noisy.reset(seed=7)
    st = noisy.ctx.state["forge"]
    crit = obs[p:]
    err = crit[lay["anchor_error_gt"]]
    assert np.linalg.norm(err) > 1e-4
    # actor sees peg tip relative to the noisy estimate; true offset = actor view + estimate error
    np.testing.assert_allclose(crit[lay["peg_tip_rel_hole_gt"]], obs[0:3] + err, atol=1e-5)
    np.testing.assert_allclose(err, st.anchor - st.hole_tip - [0, 0, st.tcp_to_tip], atol=1e-6)


def test_success_prediction_perfect_predictor() -> None:
    """With a perfect a_ET (scripted expert), the ET metrics are ideal; the smoothed prediction lags a little."""
    info = run_scripted(0)
    assert info["is_success"] and info["et_triggered"] and info["et_correct"] and info["et_recall_hit"]
    assert 0.0 <= info["et_delay"] <= 1.0  # α = 0.2 smoothing of a_ET: about 8 steps from p = 0 to p > 0.9


def test_success_prediction_penalty_and_early_termination() -> None:
    """Predicting success while unsolved is penalized; with ends_episode=true it terminates the episode."""
    ov = {"task.events.init.params.delay_until_ratio": "0.0", "task.action.ema_prediction": "false"}
    env = make(ov)
    env.reset(seed=8)
    _, r_yes, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))   # p = 1, not solved
    env.reset(seed=8)
    _, r_no, *_ = env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))   # p = 0, not solved
    w = -env.cfg.task.rewards["success_pred"].weight
    assert w == 1.0 and r_no - r_yes == pytest.approx(w, abs=1e-6)  # -w |p - y| (same action-rate norm)

    ev = make({**ov, "task.terminations.early_term.ends_episode": "true"})
    ev.reset(seed=8)
    _, _, term, trunc, info = ev.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert term and not trunc and info["termination"] == "early_term" and info["et_triggered"]
    assert not info["et_correct"] and not info["is_success"]  # a false-positive early stop


def test_success_prediction_penalty_is_delayed() -> None:
    """The prediction penalty is off until the running success rate reaches delay_until_ratio, then stays on."""
    env = make({"task.action.ema_prediction": "false"})
    env.reset(seed=8)
    st = env.ctx.state["forge"]
    i = env.reward_mgr.names.index("success_pred")
    env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))  # wrong prediction, gate closed
    assert st.pred_scale == 0.0 and env.reward_mgr._sums[i] == 0.0
    st.success_rate_ema = 0.3  # as if 30% of recent steps were successes (>= 0.25)
    env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert st.pred_scale == 1.0 and env.reward_mgr._sums[i] == pytest.approx(-1.0)
    env.reset(seed=9)
    assert st.pred_scale == 1.0  # latched across episodes


def test_action_smoothing_suppresses_chatter() -> None:
    """With EMA smoothing, a policy flipping x between -1 and +1 barely moves the target; without, it jumps."""
    def target_swing(ema: str | None) -> float:
        env = make({"task.events.randomize_controller.params.ema_range": f"({ema},{ema})"} if ema else None)
        env.reset(seed=9)
        xs = []
        for k in range(20):
            sgn = 1.0 if k % 2 == 0 else -1.0
            env.step(np.array([sgn, 0.0, 0.0, -1.0], dtype=np.float32))
            xs.append(env.controller.pos_d[0])
        return float(np.ptp(xs[10:]))  # steady-state peak-to-peak target swing [m]

    smooth, raw = target_swing(None), target_swing("1.0")  # task default vs. no smoothing
    assert raw > 0.02 and smooth < 0.4 * raw


def test_applied_action_and_action_grad_penalty() -> None:
    """last_action observes the smoothed action (a_ET included); the Isaac action-grad penalty is
    ||s_t - s_{t-1}|| on it, measured from the hold action on the first step."""
    env = make()
    env.reset(seed=10)
    lay = dict(env.obs_mgr.layout["policy"])
    s0 = env.ctx.applied_action.copy()
    assert s0[3] == -1.0  # smoothed a_ET starts at p = 0
    a1 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs, *_ = env.step(a1)
    alpha = float(env.ctx.state["action_ema"][0])
    s1 = alpha * a1 + (1.0 - alpha) * s0
    np.testing.assert_allclose(env.ctx.applied_action, s1, atol=1e-9)
    np.testing.assert_allclose(obs[lay["last_action"]], s1, atol=1e-6)
    assert env.ctx.state["pred_success"][0] == pytest.approx(0.5 * (s1[3] + 1.0))
    i = env.reward_mgr.names.index("action_grad")
    w = env.cfg.task.rewards["action_grad"].weight
    assert w == -0.1 and env.reward_mgr._sums[i] == pytest.approx(w * np.linalg.norm(s1 - s0), abs=1e-9)


def test_controller_randomization() -> None:
    """Kp, λ (per axis) and α are resampled each episode in Isaac's ranges, applied, and critic-only."""
    env = make_env("forge_peg", {"obs_mode": "asymmetric"})
    st = env.ctx.state["forge"]
    seen = []
    for seed in range(8):
        obs, _ = env.reset(seed=seed)
        c = env.controller
        assert np.all((st.kp[:3] >= 565.0 / 1.41 - 1e-6) & (st.kp[:3] <= 565.0 * 1.41 + 1e-6))
        assert np.all((st.kp[3:] >= 28.0 / 1.41 - 1e-6) & (st.kp[3:] <= 28.0 * 1.41 + 1e-6))
        assert np.all((st.lam >= 0.016 - 1e-9) & (st.lam <= 0.025 + 1e-9))
        assert 0.025 <= st.ema[0] <= 0.1
        np.testing.assert_allclose(c.kp, st.kp)                       # applied by the controller reset
        np.testing.assert_allclose(c.kd, 2.0 * np.sqrt(st.kp))         # critical damping follows
        crit = obs[env.policy_obs_dim:]
        lay = dict(env.obs_mgr.layout["critic"])
        np.testing.assert_allclose(crit[lay["controller_params_gt"]],
                                   np.r_[st.kp, st.lam, st.ema, st.dead_zone], rtol=1e-5)
        assert "controller_params_gt" not in dict(env.obs_mgr.layout["policy"])
        seen.append(st.kp[0])
    assert np.ptp(seen) > 50.0  # actually varies between episodes
    ee = env.plant.ee_pos.copy()
    env.action_mgr.apply(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32))
    assert np.all(np.abs(env.controller.pos_d - ee) <= st.lam + 1e-9)  # clip uses the sampled λ


def test_dead_zone() -> None:
    """The controller zeroes wrench components below the dead zone and shrinks the others; the dead zone is
    drawn at reset and re-drawn every 2 s."""
    env = make(dr=True)
    env.reset(seed=11)
    st, c = env.ctx.state["forge"], env.controller
    assert np.all(st.dead_zone >= 0.0) and np.all(st.dead_zone <= [5, 5, 5, 1, 1, 1])
    np.testing.assert_allclose(c.dead_zone, st.dead_zone)
    c.set_dead_zone(np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]))
    c.set_target(pos=env.plant.ee_pos + np.array([0.001, 0.0, 0.03]))  # x: Kp * 1 mm < 5 N; z: Kp * 3 cm > 5 N
    c.torque(env.plant)
    assert c._F[0] == 0.0
    assert c._F[2] == pytest.approx(c.kp[2] * 0.03 - 5.0, abs=0.05)  # at rest: Kp e - dz
    seen = {tuple(st.dead_zone)}
    for _ in range(35):  # > 2 s at 15 Hz
        env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        seen.add(tuple(st.dead_zone))
    assert len(seen) >= 2


def test_part_and_grasp_randomization() -> None:
    """Friction, the peg's offset in the hand and the gripper yaw vary per episode; the reset never starts
    in contact; the critic sees friction and offset, the actor does not."""
    env = make_env("forge_peg", {"obs_mode": "asymmetric"})
    st, m, d = env.ctx.state["forge"], env.ctx.model, env.plant.data
    tcp = env.ctx.handles.tcp_site_id
    mus, offs = [], []
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert d.ncon == 0 and st.ik_err_max == 0.0
        assert 0.5 <= st.friction <= 1.0
        np.testing.assert_allclose(m.geom_friction[st.part_geoms, 0], st.friction)
        assert np.all(np.abs(st.held_offset) <= 0.003) and st.held_offset[1] == 0.0
        # tip in the hand (TCP) frame = nominal tip + offset
        R = d.site_xmat[tcp].reshape(3, 3)
        tip_in_tcp = R.T @ (d.site_xpos[st.peg_tip] - d.site_xpos[tcp])
        np.testing.assert_allclose(tip_in_tcp, [0.0, 0.0, st.tcp_to_tip] + st.held_offset, atol=1e-6)
        assert st.tcp_to_tip_true == pytest.approx(st.tcp_to_tip + st.held_offset[2])
        lay = dict(env.obs_mgr.layout["critic"])
        np.testing.assert_allclose(obs[env.policy_obs_dim:][lay["part_params_gt"]],
                                   np.r_[st.friction, st.held_offset], rtol=1e-5, atol=1e-7)
        mus.append(st.friction)
        offs.append(st.held_offset[0])
    assert np.ptp(mus) > 0.1 and np.ptp(offs) > 0.002


def test_observation_noise_policy_only() -> None:
    """Isaac observation noise reaches the actor only; per-entry std (quaternion w has none)."""
    env = make_env("forge_peg", {"obs_mode": "asymmetric"})
    obs, _ = env.reset(seed=12)
    pol, crit = obs[:env.policy_obs_dim], obs[env.policy_obs_dim:]
    lp, lc = dict(env.obs_mgr.layout["policy"]), dict(env.obs_mgr.layout["critic"])
    dq = pol[lp["ee_quat_rel_nominal"]] - crit[lc["ee_quat_rel_nominal"]]
    assert dq[0] == 0.0 and np.all(np.abs(dq[1:]) > 0.0) and np.all(np.abs(dq[1:]) < 0.005)
    dv = pol[lp["ee_vel"]] - crit[lc["ee_vel"]]
    assert np.all(np.abs(dv) > 0.0) and np.all(np.abs(dv) < 1.0)
    df = pol[lp["contact_force"]] - crit[lc["contact_force"]]
    assert np.all(np.abs(df) > 0.0) and np.all(np.abs(df) < 6.0)
    with pytest.raises(ValueError):
        make_env("forge_peg", {"task.obs.term_cfg.ee_vel.noise_std": "(0.1,0.1)"})  # wrong length


def test_policy_rate_from_task_env_defaults() -> None:
    """The task sets 15 Hz / 150 steps through env_defaults; an explicit CLI decimation still wins."""
    env = make()
    assert env.cfg.decimation == 33 and env.ctx.max_episode_steps == 150
    assert env.cfg.policy_hz == pytest.approx(15.15, abs=0.01)
    assert make({"decimation": "25"}).cfg.decimation == 25
