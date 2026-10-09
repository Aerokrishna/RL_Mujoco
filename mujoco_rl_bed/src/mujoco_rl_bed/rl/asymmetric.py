"""Asymmetric actor-critic policies for stable-baselines3 / sb3-contrib (no library patching).

The env (with `EnvCfg.obs_mode="asymmetric"`) returns one flat vector

    obs = [ policy group (actor_dim) | critic group (privileged) ]

SB3 feeds the same observation tensor to the actor and the critic, so the split happens
in two parameter-free feature extractors (`share_features_extractor=False`):

- actor:  `ActorMaskExtractor` multiplies the observation by a fixed 0/1 mask that keeps only
          the first `actor_dim` entries. Privileged inputs are therefore exactly zero for the
          actor (zero input, zero gradient), so the policy cannot use them, at training or deployment.
- critic: `CriticMaskExtractor` keeps only the critic block (entries from `actor_dim` on), so the
          value function sees clean privileged state only, not the actor's noisy observations.
          With `critic_sees_actor_obs=True` it gets the full vector instead.

Both extractors output the same size. sb3-contrib needs this: its critic LSTM's input
size is the actor extractor's `features_dim`. Neither extractor has parameters, so replacing the
critic extractor after construction leaves the optimizer untouched.

`AsymmetricRecurrentPolicy` also replaces sb3-contrib's LSTM sequence processing with an exact, faster
version (`fast_process_sequence`): sb3-contrib steps the LSTM one timestep at a time in Python whenever
a minibatch contains an episode start, which with 150-step episodes is nearly every minibatch.

Use via `TrainCfg.asymmetric=true` (see `mujoco_rl_bed.rl.train`); checkpoints store the
policy class and `actor_dim`, so `eval.py` loads them unchanged.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class ActorMaskExtractor(BaseFeaturesExtractor):
    """Keep only the first `actor_dim` observation entries (zero the privileged rest).

    Output size equals the full observation size, so the actor and critic paths have
    the same `features_dim` (required by sb3-contrib's critic LSTM).
    """

    keep_actor_part: bool = True

    def __init__(self, observation_space: gym.spaces.Box, actor_dim: int) -> None:
        """Build the fixed mask.

        Args:
            observation_space: Flat Box of size actor_dim + privileged_dim.
            actor_dim: Number of leading entries in the actor (policy) block.
        """
        n = int(np.prod(observation_space.shape))
        if not 0 < actor_dim < n:
            raise ValueError(f"actor_dim={actor_dim} must be in (0, {n})")
        super().__init__(observation_space, features_dim=n)
        mask = th.zeros(n)
        if self.keep_actor_part:
            mask[:actor_dim] = 1.0
        else:
            mask[actor_dim:] = 1.0
        self.register_buffer("mask", mask)  # saved with the model, moves with .to(device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Mask the observation.

        Args:
            observations: Batch of observations, shape (B, n).

        Returns:
            Masked features, shape (B, n).
        """
        return th.flatten(observations, start_dim=1) * self.mask


class CriticMaskExtractor(ActorMaskExtractor):
    """Keep only the critic block (entries `actor_dim:`), zeroing the actor's noisy observations."""

    keep_actor_part = False


def _critic_extractor(observation_space: gym.spaces.Box, actor_dim: int, sees_actor_obs: bool) -> BaseFeaturesExtractor:
    """Critic feature extractor.

    Args:
        observation_space: Flat [policy | critic] Box.
        actor_dim: Size of the policy block.
        sees_actor_obs: If True, the critic gets the full vector; else only the critic block.

    Returns:
        A parameter-free extractor.
    """
    return FlattenExtractor(observation_space) if sees_actor_obs else CriticMaskExtractor(observation_space, actor_dim)


def _asym_kwargs(kwargs: dict[str, Any], actor_dim: int) -> dict[str, Any]:
    """Force separate extractors with the actor mask.

    Args:
        kwargs: Policy kwargs from the algorithm.
        actor_dim: Actor observation size.

    Returns:
        Updated kwargs.
    """
    kwargs = dict(kwargs)
    kwargs["share_features_extractor"] = False
    kwargs["features_extractor_class"] = ActorMaskExtractor
    kwargs["features_extractor_kwargs"] = {"actor_dim": int(actor_dim)}
    return kwargs


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """PPO (MLP) policy whose actor sees only the policy group and critic sees everything."""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, lr_schedule,
                 actor_dim: int, critic_sees_actor_obs: bool = False, **kwargs: Any) -> None:
        """Build the policy, then give the critic an unmasked extractor.

        Args:
            observation_space: Flat [policy | critic] Box.
            action_space: Action space.
            lr_schedule: Learning-rate schedule.
            actor_dim: Size of the policy (actor) part.
            critic_sees_actor_obs: Also feed the actor's (possibly noisy) block to the critic.
            **kwargs: Remaining `ActorCriticPolicy` kwargs (net_arch, log_std_init, ...).
        """
        self.actor_dim = int(actor_dim)
        self.critic_sees_actor_obs = bool(critic_sees_actor_obs)
        super().__init__(observation_space, action_space, lr_schedule, **_asym_kwargs(kwargs, actor_dim))
        # parameter-free: replacing it leaves the optimizer unaffected
        self.vf_features_extractor = _critic_extractor(observation_space, self.actor_dim, self.critic_sees_actor_obs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Include `actor_dim` so a saved policy can be rebuilt.

        Returns:
            Constructor kwargs.
        """
        data = super()._get_constructor_parameters()
        data["actor_dim"] = self.actor_dim
        data["critic_sees_actor_obs"] = self.critic_sees_actor_obs
        return data


def fast_process_sequence(features: th.Tensor, lstm_states: tuple[th.Tensor, th.Tensor], episode_starts: th.Tensor,
                          lstm: th.nn.LSTM) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
    """LSTM forward over padded sequences, exact replacement of `RecurrentActorCriticPolicy._process_sequence`.

    The recurrent rollout buffer cuts sequences at episode starts, so during gradient updates a start
    can only be the first element of a sequence. Then resetting the initial state of those sequences
    and running cuDNN over the whole sequence in one call gives the same result as sb3-contrib's
    per-timestep Python loop. If a start appears later in a sequence, fall back to the original loop.

    Args:
        features: (n_seq * seq_len, input) batch, sequence-major as in sb3-contrib.
        lstm_states: (h, c), each (n_layers, n_seq, hidden).
        episode_starts: (n_seq * seq_len,) 1.0 where an episode starts.
        lstm: The LSTM.

    Returns:
        (output (n_seq * seq_len, hidden), (h, c)).
    """
    n_seq = lstm_states[0].shape[1]
    starts = episode_starts.reshape((n_seq, -1))
    if starts.shape[1] > 1 and th.any(starts[:, 1:] != 0.0):
        return RecurrentActorCriticPolicy._process_sequence(features, lstm_states, episode_starts, lstm)
    keep = (1.0 - starts[:, 0]).view(1, n_seq, 1)
    seq = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
    out, states = lstm(seq, (keep * lstm_states[0], keep * lstm_states[1]))
    return th.flatten(out.transpose(0, 1), start_dim=0, end_dim=1), states


class AsymmetricRecurrentPolicy(RecurrentActorCriticPolicy):
    """RecurrentPPO (LSTM) policy: actor LSTM gets the policy group, critic LSTM gets everything."""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, lr_schedule,
                 actor_dim: int, critic_sees_actor_obs: bool = False, **kwargs: Any) -> None:
        """Build the policy, then give the critic an unmasked extractor.

        Args:
            observation_space: Flat [policy | critic] Box.
            action_space: Action space.
            lr_schedule: Learning-rate schedule.
            actor_dim: Size of the policy (actor) part.
            critic_sees_actor_obs: Also feed the actor's (possibly noisy) block to the critic.
            **kwargs: Remaining `RecurrentActorCriticPolicy` kwargs (lstm_hidden_size, net_arch, ...).
        """
        if kwargs.get("shared_lstm", False):
            raise ValueError("Asymmetric recurrent policy needs separate actor/critic LSTMs (shared_lstm=False)")
        self.actor_dim = int(actor_dim)
        self.critic_sees_actor_obs = bool(critic_sees_actor_obs)
        super().__init__(observation_space, action_space, lr_schedule, **_asym_kwargs(kwargs, actor_dim))
        # parameter-free: replacing it leaves the optimizer unaffected
        self.vf_features_extractor = _critic_extractor(observation_space, self.actor_dim, self.critic_sees_actor_obs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Include `actor_dim` so a saved policy can be rebuilt.

        Returns:
            Constructor kwargs.
        """
        data = super()._get_constructor_parameters()
        data["actor_dim"] = self.actor_dim
        data["critic_sees_actor_obs"] = self.critic_sees_actor_obs
        return data

    @staticmethod
    def _process_sequence(features: th.Tensor, lstm_states: tuple[th.Tensor, th.Tensor], episode_starts: th.Tensor,
                          lstm: th.nn.LSTM) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Exact, faster LSTM sequence processing (see `fast_process_sequence`)."""
        return fast_process_sequence(features, lstm_states, episode_starts, lstm)
