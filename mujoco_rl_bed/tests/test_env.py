"""Environment tests: spaces, reset/step, determinism, config overrides and lazy viewer import."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import mujoco_rl_bed.tasks  # noqa: F401  (registers tasks)
from mujoco_rl_bed.tasks.registry import list_tasks, make_env

TASKS = list_tasks()


@pytest.mark.parametrize("task", TASKS)
def test_spaces_and_shapes(task: str) -> None:
    """Observation/action shapes and dtypes match the spaces; critic obs is in info."""
    env = make_env(task)
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (env.obs_mgr.dim("policy"),)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)
    if env.obs_mgr.has_group("critic"):
        assert info["critic_obs"].shape == (env.obs_mgr.dim("critic"),)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()


@pytest.mark.parametrize("task", TASKS)
def test_full_episode_runs_and_truncates(task: str) -> None:
    """An episode of random actions ends by timeout with per-term reward logs."""
    env = make_env(task)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for k in range(env.ctx.max_episode_steps):
        _, _, terminated, truncated, info = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
        if terminated or truncated:
            break
    assert terminated or truncated
    assert set(info["reward_terms"]) == set(env.reward_mgr.names)
    assert "is_success" in info and "termination" in info
    assert np.all(np.isfinite(env.plant.data.qpos))
    env.close()


@pytest.mark.parametrize("task", TASKS)
def test_determinism_same_seed(task: str) -> None:
    """Two envs with the same seed and actions produce identical rollouts."""
    def rollout(seed: int) -> np.ndarray:
        env = make_env(task)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(123)
        traj = [obs]
        for _ in range(20):
            obs, r, *_ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
            traj.append(np.append(obs, r))
        env.close()
        return np.concatenate(traj)

    a, b, c = rollout(7), rollout(7), rollout(8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_returned_obs_is_not_shared_buffer() -> None:
    """step/reset return copies, so stored observations are never overwritten."""
    env = make_env("reach")
    o0, _ = env.reset(seed=0)
    keep = o0.copy()
    env.step(np.ones(env.action_space.shape, dtype=np.float32))
    np.testing.assert_array_equal(o0, keep)


def test_overrides_apply_and_type_check() -> None:
    """Dotted overrides reach nested dataclasses and reject bad keys/values."""
    env = make_env("reach", {"decimation": "10", "task.action.pos_scale": "0.01",
                             "task.rewards.action_rate_l2.weight": "-0.1"})
    assert env.cfg.decimation == 10 and env.cfg.task.action.pos_scale == 0.01
    assert env.reward_mgr._w[env.reward_mgr.names.index("action_rate_l2")] == -0.1
    with pytest.raises(KeyError):
        make_env("reach", {"task.does_not_exist": "1"})
    with pytest.raises(ValueError):
        make_env("reach", {"decimation": "abc"})


def test_wrist_wrench_accumulation_and_history() -> None:
    """Accumulated terms and history stacking produce the expected sizes and finite values."""
    from mujoco_rl_bed.env.cfg import ObsTermCfg

    env = make_env("reach")
    env.cfg.task.obs.groups["policy"] = ["ee_pos", "wrist_wrench"]
    env.cfg.task.obs.term_cfg["ee_pos"] = ObsTermCfg(history=3)
    from mujoco_rl_bed.env.torque_env import TorqueEnv

    env = TorqueEnv(env.cfg)
    assert env.obs_mgr.needs_accumulation
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 * 3 + 6,)
    np.testing.assert_array_equal(obs[0:3], obs[6:9])  # history filled with the reset value
    obs, *_ = env.step(np.zeros(3, dtype=np.float32))
    assert np.all(np.isfinite(obs))


def test_render_false_does_not_import_viewer() -> None:
    """With render=False, `mujoco.viewer` is never imported (checked in a clean interpreter)."""
    code = (
        "import sys, numpy as np\n"
        "from mujoco_rl_bed.tasks import make_env\n"
        "env = make_env('reach'); env.reset(seed=0); env.step(np.zeros(3, dtype=np.float32))\n"
        "print('mujoco.viewer' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"
