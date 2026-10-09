"""Environment and manager configuration dataclasses.

A task is pure configuration (`TaskCfg`). It names its observation, reward, termination
and event terms, and the managers resolve those names against their registries once at
env construction. Reward, termination and event configs are dicts keyed by term name, so
CLI overrides can address them directly, e.g. `task.rewards.action_rate.weight=-0.02`.
For convenience a task may pass the reward list as `[(name, weight), ...]`, and
`TaskCfg.__post_init__` normalizes it into the dict form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mujoco_rl_bed.control.base import ControllerCfg
from mujoco_rl_bed.control.cartesian_impedance import CartesianImpedanceCfg
from mujoco_rl_bed.sim.scene import SceneCfg


@dataclass
class ObsTermCfg:
    """Per-term observation options.

    Attributes:
        noise_std: Std of additive Gaussian noise (applied only in `ObsCfg.noisy_groups`): one value
            for all entries of the term, or one per entry (length = term dim).
        history: Number of stacked past values (1 = current only), ordered oldest -> newest.
        params: Extra keyword arguments bound to the term function.
    """

    noise_std: float | tuple[float, ...] = 0.0
    history: int = 1
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObsCfg:
    """Observation groups.

    Attributes:
        groups: Group name -> list of term names. Conventional groups: `policy` (actor
            input, returned by `step`), `critic` (privileged, returned in
            `info["critic_obs"]`), `dataset` (recorded for imitation learning).
        term_cfg: Term name -> options (noise, history, params) shared by all groups.
        noisy_groups: Groups that receive noise (the critic usually sees clean values).
    """

    groups: dict[str, list[str]] = field(default_factory=lambda: {"policy": []})
    term_cfg: dict[str, ObsTermCfg] = field(default_factory=dict)
    noisy_groups: tuple[str, ...] = ("policy",)


@dataclass
class ActionCfg:
    """Action term selection and scaling.

    Attributes:
        term: Registered action term name (`delta_ee_pose`, `absolute_ee_pose`,
            `delta_ee_pose_with_stiffness`, `delta_joint_pos`, `anchor_relative_pos`).
        pos_scale: Translation per unit action per policy step [m] (delta terms).
        rot_scale: Rotation per unit action [rad] (delta terms: per step; absolute: max offset).
        rotation: Include the 3 rotation dims; if False, orientation is held at the reset pose.
        pos_lo: Lower workspace bound for the TCP target, world frame [m].
        pos_hi: Upper workspace bound for the TCP target, world frame [m].
        kp_pos_range: Translational stiffness range [N/m] for variable-impedance actions (log scale).
        kp_rot_range: Rotational stiffness range [Nm/rad] for variable-impedance actions (log scale).
        joint_scale: Joint delta per unit action per step [rad] (`delta_joint_pos`).
        anchor: `ctx.state` buffer holding the anchor position for `anchor_relative_pos`
            (e.g. the estimated hole tip) [m].
        anchor_bounds: Target range around the anchor per unit action, per axis [m].
        max_step: λ: per-axis clip of the target around the current TCP position [m] (initial value of
            the `action_max_step` state buffer, shape (3,), which events may randomize per axis).
        ema_factor: Action smoothing alpha in (0, 1]: the applied (position) action is
            alpha * a_t + (1 - alpha) * applied_{t-1}. 1.0 = off. Removes step-to-step chatter
            (e.g. a policy flipping between -1 and +1) before it reaches the controller target
            (supported by `anchor_relative_pos`).
        ema_prediction: Also smooth the success-prediction dim a_ET with the same EMA (Isaac Lab FORGE
            smooths all action dims, so its predicted success lags the raw output).
        success_prediction: Append one action dim a_ET in [-1, 1] -> p = (a_ET + 1) / 2 in [0, 1],
            the policy's predicted probability that the task is currently solved (FORGE Sec. III-C).
            Written to `ctx.state["pred_success"]`; rewarded/used by task terms (supported by
            `anchor_relative_pos`).
        params: Extra term-specific options.
    """

    term: str = "delta_ee_pose"
    pos_scale: float = 0.02
    rot_scale: float = 0.1
    rotation: bool = True
    pos_lo: tuple[float, float, float] = (0.2, -0.5, 0.02)
    pos_hi: tuple[float, float, float] = (0.85, 0.5, 0.8)
    kp_pos_range: tuple[float, float] = (100.0, 2000.0)
    kp_rot_range: tuple[float, float] = (10.0, 200.0)
    joint_scale: float = 0.05
    anchor: str = "fixed_anchor"
    anchor_bounds: tuple[float, float, float] = (0.05, 0.05, 0.05)
    max_step: float = 0.02
    ema_factor: float = 1.0
    ema_prediction: bool = False
    success_prediction: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardTermCfg:
    """One reward term.

    Attributes:
        weight: Multiplier on the term value (0 disables the term).
        func: Registered term name; empty = use the dict key.
        params: Keyword arguments bound to the term function.
    """

    weight: float = 1.0
    func: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationTermCfg:
    """One termination term (term functions return a bool).

    Attributes:
        time_out: If True, firing counts as truncation (bootstrapped), not termination.
        success: If True, firing marks the episode as successful (`info["is_success"]`).
        ends_episode: If False, the term never ends the episode; with `success=True` it
            reports whether the condition holds at the final step (e.g. "near target at timeout").
        func: Registered term name; empty = use the dict key.
        params: Keyword arguments bound to the term function.
    """

    time_out: bool = False
    success: bool = False
    ends_episode: bool = True
    func: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventTermCfg:
    """One event (randomization / reset logic).

    Attributes:
        mode: "startup" (once at env creation), "reset" (every reset), or "interval".
        interval_s: Period for interval events [s] (rounded to whole policy steps, >= 1 step).
        func: Registered term name; empty = use the dict key.
        params: Keyword arguments bound to the event function.
    """

    mode: str = "reset"
    interval_s: float = 0.0
    func: str = ""
    params: dict[str, Any] = field(default_factory=dict)


def _normalize_terms(items: Any, cls: type, what: str) -> dict[str, Any]:
    """Convert list/tuple shorthands into a `{name: cfg}` dict.

    Accepted forms: dict name -> cfg | weight | params-dict; list of names; list of
    `(name, weight)` / `(name, weight, params)` (rewards) or `(name, params)`.

    Args:
        items: User-provided terms.
        cls: Config class to instantiate (`RewardTermCfg`, ...).
        what: Name used in error messages.

    Returns:
        Dict name -> cfg instance.
    """
    out: dict[str, Any] = {}
    if isinstance(items, dict):
        pairs = list(items.items())
    else:
        pairs = []
        for it in items:
            if isinstance(it, str):
                pairs.append((it, cls()))
            elif isinstance(it, (tuple, list)):
                pairs.append((it[0], tuple(it[1:])))
            else:
                raise TypeError(f"Bad {what} entry {it!r}")
    for name, v in pairs:
        if isinstance(v, cls):
            out[name] = v
        elif isinstance(v, (int, float)) and cls is RewardTermCfg:
            out[name] = cls(weight=float(v))
        elif isinstance(v, dict):
            out[name] = cls(params=dict(v))
        elif isinstance(v, tuple):
            if cls is RewardTermCfg:
                out[name] = cls(weight=float(v[0]), params=dict(v[1]) if len(v) > 1 else {})
            elif len(v) == 1 and isinstance(v[0], cls):
                out[name] = v[0]
            else:
                out[name] = cls(params=dict(v[0]) if v else {})
        else:
            raise TypeError(f"Bad {what} entry {name!r}: {v!r}")
    return out


@dataclass
class TaskCfg:
    """Everything that defines a task.

    Attributes:
        scene: Scene composition (robot + task assets).
        controller: Low-level controller config (its type selects the controller).
        action: Action term config.
        obs: Observation groups.
        rewards: Reward terms, name -> `RewardTermCfg`.
        terminations: Termination terms, name -> `TerminationTermCfg` (a timeout from
            `episode_length_s` is always added).
        events: Event terms, name -> `EventTermCfg`.
        episode_length_s: Episode length [s] (truncation).
        params: Free-form task parameters (read by task-specific terms).
        env_defaults: `EnvCfg`-level defaults this task needs (e.g. {"decimation": 33} for a 15 Hz policy),
            applied by `make_env_cfg` before the user's overrides, so the CLI still wins.
    """

    scene: SceneCfg = field(default_factory=SceneCfg)
    controller: ControllerCfg = field(default_factory=CartesianImpedanceCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    events: dict[str, EventTermCfg] = field(default_factory=dict)
    episode_length_s: float = 5.0
    params: dict[str, Any] = field(default_factory=dict)
    env_defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize shorthand term lists into dicts."""
        self.rewards = _normalize_terms(self.rewards, RewardTermCfg, "reward")
        self.terminations = _normalize_terms(self.terminations, TerminationTermCfg, "termination")
        self.events = _normalize_terms(self.events, EventTermCfg, "event")


@dataclass
class EnvCfg:
    """Top-level environment configuration.

    Attributes:
        task: The task config.
        task_name: Registered task name (informational, written to config.json).
        sim_dt: Physics timestep [s].
        decimation: Physics ticks per policy step (policy_hz = 1 / (sim_dt * decimation)).
        render: Open a passive viewer (forced off for multi-env training).
        realtime: Pace the viewer to wall time.
        seed: Base seed for `np.random.default_rng`.
        critic_in_info: Put the `critic` group into `info["critic_obs"]` every step (if defined;
            ignored in asymmetric mode, where the critic group is part of the observation).
        obs_mode: "policy": the observation is the `policy` group. "asymmetric": it is
            [policy group | critic group] (for asymmetric actor-critic: the actor reads only the first
            `TorqueEnv.policy_obs_dim` entries, see `mujoco_rl_bed.rl.asymmetric`).
    """

    task: TaskCfg = field(default_factory=TaskCfg)
    task_name: str = ""
    sim_dt: float = 0.002
    decimation: int = 25
    render: bool = False
    realtime: bool = False
    seed: int = 0
    critic_in_info: bool = True
    obs_mode: str = "policy"

    @property
    def policy_dt(self) -> float:
        """Policy period [s]."""
        return self.sim_dt * self.decimation

    @property
    def policy_hz(self) -> float:
        """Policy rate [Hz]."""
        return 1.0 / self.policy_dt
