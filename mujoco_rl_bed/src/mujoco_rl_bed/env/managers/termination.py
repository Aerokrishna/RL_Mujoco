"""TerminationManager and the `@termination_term` registry.

Term contract: `fn(ctx, **params) -> bool`. Each term's config says whether firing
means termination (MDP end, no bootstrap) or truncation (`time_out=True`, bootstrapped),
and whether it marks success. `compute()` returns `(terminated, truncated)`. A
timeout after `max_episode_steps` policy steps is always included as truncation.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from mujoco_rl_bed.env.cfg import TerminationTermCfg

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context

TermFn = Callable[..., bool]
TERMINATION_TERMS: dict[str, TermFn] = {}


def termination_term(name: str) -> Callable[[TermFn], TermFn]:
    """Decorator registering a termination term.

    Args:
        name: Unique term name.

    Returns:
        The decorator (returns the function unchanged).
    """
    def deco(fn: TermFn) -> TermFn:
        if name in TERMINATION_TERMS:
            raise KeyError(f"Termination term '{name}' already registered")
        TERMINATION_TERMS[name] = fn
        return fn
    return deco


class TerminationManager:
    """Evaluates termination terms once per policy step.

    Attributes:
        fired: Name of the first term that fired this step (or "time_out"), else "".
        success: Whether a success-marking term fired during this episode.
    """

    def __init__(self, cfg: dict[str, TerminationTermCfg], ctx: "Context") -> None:
        """Resolve and bind terms.

        Args:
            cfg: Name -> termination term config.
            ctx: Shared context (provides `max_episode_steps`).
        """
        self.ctx = ctx
        self._terms: list[tuple[str, Callable[["Context"], bool], bool, bool]] = []
        self._monitors: list[Callable[["Context"], bool]] = []  # success-only terms that never end episodes
        for name, tc in cfg.items():
            key = tc.func or name
            if key not in TERMINATION_TERMS:
                raise KeyError(f"Unknown termination term '{key}'. Registered: {sorted(TERMINATION_TERMS)}")
            fn = TERMINATION_TERMS[key]
            if not tc.ends_episode:
                if tc.success:
                    self._monitors.append(partial(fn, **tc.params) if tc.params else fn)
                continue
            self._terms.append((name, partial(fn, **tc.params) if tc.params else fn, tc.time_out, tc.success))
        self.fired = ""
        self.success = False
        self._final_success = False

    def compute(self) -> tuple[bool, bool]:
        """Evaluate all terms.

        Returns:
            (terminated, truncated).
        """
        ctx = self.ctx
        terminated = truncated = False
        self.fired = ""
        for name, fn, time_out, success in self._terms:
            if fn(ctx):
                if time_out:
                    truncated = True
                else:
                    terminated = True
                if success:
                    self.success = True
                if not self.fired:
                    self.fired = name
        if ctx.episode_step >= ctx.max_episode_steps:
            truncated = True
            if not self.fired:
                self.fired = "time_out"
        if self._monitors and (terminated or truncated):
            self._final_success = any(fn(ctx) for fn in self._monitors)  # evaluated at the final step only
        return terminated, truncated

    @property
    def episode_success(self) -> bool:
        """Success flag for the episode: a success term ended it, or a monitor holds at the last step."""
        return self.success or self._final_success

    def reset(self) -> None:
        """Clear per-episode flags."""
        self.fired = ""
        self.success = False
        self._final_success = False
