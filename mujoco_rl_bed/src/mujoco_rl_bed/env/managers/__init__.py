"""Managers that compile task configs into per-step callables (obs, action, reward, termination, events)."""

from mujoco_rl_bed.env.managers.action import ACTION_TERMS, ActionManager, ActionTerm, action_term
from mujoco_rl_bed.env.managers.events import EVENT_TERMS, EventManager, event_term
from mujoco_rl_bed.env.managers.observation import OBS_TERMS, ObservationManager, obs_term
from mujoco_rl_bed.env.managers.reward import REWARD_TERMS, RewardManager, reward_term
from mujoco_rl_bed.env.managers.termination import TERMINATION_TERMS, TerminationManager, termination_term

__all__ = [
    "ACTION_TERMS", "ActionManager", "ActionTerm", "action_term",
    "EVENT_TERMS", "EventManager", "event_term",
    "OBS_TERMS", "ObservationManager", "obs_term",
    "REWARD_TERMS", "RewardManager", "reward_term",
    "TERMINATION_TERMS", "TerminationManager", "termination_term",
]
