"""Asymmetric actor-critic tests: env layout, actor blind to privileged inputs, critic not, save/load."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import mujoco_rl_bed.tasks  # noqa: F401  (registers tasks)
from mujoco_rl_bed.rl.asymmetric import AsymmetricActorCriticPolicy, AsymmetricRecurrentPolicy
from mujoco_rl_bed.tasks.registry import make_env


def asym_env():
    """`reach` in asymmetric observation mode.

    Returns:
        A TorqueEnv.
    """
    return make_env("reach", {"obs_mode": "asymmetric"})


def test_env_asymmetric_layout() -> None:
    """The observation is [policy | critic] and its first block equals the policy group."""
    env = asym_env()
    obs, info = env.reset(seed=0)
    p, c = env.obs_mgr.dim("policy"), env.obs_mgr.dim("critic")
    assert env.policy_obs_dim == p and obs.shape == (p + c,) == env.observation_space.shape
    assert "critic_obs" not in info
    sym = make_env("reach")
    sym_obs, _ = sym.reset(seed=0)
    np.testing.assert_array_equal(obs[:p], sym_obs)
    with pytest.raises(ValueError):
        make_env("reach", {"obs_mode": "bogus"})


def _models():
    """Build an asymmetric PPO and RecurrentPPO on the same env.

    Returns:
        (venv, actor_dim, [(name, model)]).
    """
    venv = DummyVecEnv([asym_env])
    actor_dim = venv.get_attr("policy_obs_dim")[0]
    pk = dict(actor_dim=actor_dim, net_arch=dict(pi=[32], vf=[32]))
    ppo = PPO(AsymmetricActorCriticPolicy, venv, policy_kwargs=pk, n_steps=64, batch_size=64, device="cpu", seed=0)
    rpk = dict(pk, lstm_hidden_size=16)
    rppo = RecurrentPPO(AsymmetricRecurrentPolicy, venv, policy_kwargs=rpk, n_steps=64, batch_size=64, device="cpu",
                        seed=0)
    return venv, actor_dim, [("ppo", ppo), ("recurrent_ppo", rppo)]


def _actor_and_value(name: str, model, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Action mean and value for a batch of first-step observations.

    Args:
        name: "ppo" or "recurrent_ppo".
        model: The SB3 model.
        obs: Observations, shape (B, n).

    Returns:
        (action mean, value).
    """
    pol = model.policy
    with th.no_grad():
        if name == "ppo":
            return pol.get_distribution(obs).distribution.mean, pol.predict_values(obs)
        b = obs.shape[0]
        h = th.zeros(pol.lstm_hidden_state_shape[0], b, pol.lstm_hidden_state_shape[2])
        starts = th.ones(b)
        dist, _ = pol.get_distribution(obs, (h, h), starts)
        return dist.distribution.mean, pol.predict_values(obs, (h, h), starts)


@pytest.mark.parametrize("which", [0, 1])
def test_actor_ignores_privileged_critic_uses_it(which: int) -> None:
    """Changing only the critic block leaves the actor output unchanged but changes the value."""
    venv, actor_dim, models = _models()
    name, model = models[which]
    obs = th.as_tensor(venv.reset(), dtype=th.float32).repeat(4, 1)
    perturbed = obs.clone()
    perturbed[:, actor_dim:] += th.randn_like(perturbed[:, actor_dim:]) * 5.0
    mean_a, val_a = _actor_and_value(name, model, obs)
    mean_b, val_b = _actor_and_value(name, model, perturbed)
    th.testing.assert_close(mean_a, mean_b)
    assert not th.allclose(val_a, val_b)
    # the critic (default) ignores the actor block: clean privileged state only
    actor_only = obs.clone()
    actor_only[:, :actor_dim] += th.randn_like(actor_only[:, :actor_dim]) * 5.0
    _, val_d = _actor_and_value(name, model, actor_only)
    th.testing.assert_close(val_a, val_d)
    # and the actor does react to its own inputs
    moved = obs.clone()
    moved[:, :actor_dim] += 1.0
    mean_c, _ = _actor_and_value(name, model, moved)
    assert not th.allclose(mean_a, mean_c)


@pytest.mark.parametrize("which", [0, 1])
def test_learn_save_load_roundtrip(which: int, tmp_path) -> None:
    """A short learn() runs, and a saved model reloads with the same policy class and actions."""
    venv, actor_dim, models = _models()
    name, model = models[which]
    model.learn(total_timesteps=64)
    model.save(tmp_path / "m.zip")
    cls = PPO if name == "ppo" else RecurrentPPO
    loaded = cls.load(tmp_path / "m.zip", device="cpu")
    assert type(loaded.policy) is type(model.policy) and loaded.policy.actor_dim == actor_dim
    assert loaded.policy.critic_sees_actor_obs is False
    obs = venv.reset()
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2, atol=1e-6)
