"""EventManager and the `@event_term` registry.

Term contract: `fn(ctx, **params) -> None`. Modes:
- `startup`: once, right after the env is built (e.g. persistent model randomization),
- `reset`: on every reset, after `plant.reset` and before `mj_forward` and the controller
  reset (e.g. initial joints, object poses, targets, friction, controller gains),
- `interval`: every `round(interval_s / policy_dt)` policy steps, before the action is
  applied (e.g. torque noise held for one policy step, pushes),
- `step`: after the physics of every policy step and at the end of reset, before
  observations/rewards/terminations (task bookkeeping computed once and shared by terms).

Model randomization writes into `ctx.model`, which each env (and each subprocess) owns.
Nominal values should be cached in `ctx.state` on the first call, so repeated resets
don't compound.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from mujoco_rl_bed.env.cfg import EventTermCfg

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context

EventFn = Callable[..., None]
EVENT_TERMS: dict[str, EventFn] = {}
_MODES = ("startup", "reset", "interval", "step")


def event_term(name: str) -> Callable[[EventFn], EventFn]:
    """Decorator registering an event term.

    Args:
        name: Unique term name.

    Returns:
        The decorator (returns the function unchanged).
    """
    def deco(fn: EventFn) -> EventFn:
        if name in EVENT_TERMS:
            raise KeyError(f"Event term '{name}' already registered")
        EVENT_TERMS[name] = fn
        return fn
    return deco


class EventManager:
    """Runs startup/reset/interval events."""

    def __init__(self, cfg: dict[str, EventTermCfg], ctx: "Context") -> None:
        """Resolve and bind events by mode.

        Args:
            cfg: Name -> event config.
            ctx: Shared context (provides `policy_dt`).
        """
        self.ctx = ctx
        self._startup: list[Callable] = []
        self._reset: list[Callable] = []
        self._interval: list[tuple[Callable, int]] = []
        self._post_step: list[Callable] = []
        for name, ec in cfg.items():
            if ec.mode not in _MODES:
                raise ValueError(f"Event '{name}': mode must be one of {_MODES}, got '{ec.mode}'")
            key = ec.func or name
            if key not in EVENT_TERMS:
                raise KeyError(f"Unknown event term '{key}'. Registered: {sorted(EVENT_TERMS)}")
            fn = EVENT_TERMS[key]
            bound = partial(fn, **ec.params) if ec.params else fn
            if ec.mode == "startup":
                self._startup.append(bound)
            elif ec.mode == "reset":
                self._reset.append(bound)
            elif ec.mode == "step":
                self._post_step.append(bound)
            else:
                every = max(1, round(ec.interval_s / ctx.policy_dt))
                self._interval.append((bound, every))
        self.has_interval = bool(self._interval)
        self.has_post_step = bool(self._post_step)

    def startup(self) -> None:
        """Run startup events (once)."""
        for fn in self._startup:
            fn(self.ctx)

    def reset(self) -> None:
        """Run reset events."""
        for fn in self._reset:
            fn(self.ctx)

    def post_step(self) -> None:
        """Run `step` events (after physics, before obs/reward/termination)."""
        for fn in self._post_step:
            fn(self.ctx)

    def step(self) -> None:
        """Run interval events that are due at the current policy step."""
        ctx = self.ctx
        k = ctx.episode_step
        for fn, every in self._interval:
            if k % every == 0:
                fn(ctx)
