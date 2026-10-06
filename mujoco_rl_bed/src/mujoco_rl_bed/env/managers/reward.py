"""RewardManager and the `@reward_term` registry.

Term contract: `fn(ctx, **params) -> float`. The step reward is sum_i w_i * fn_i(ctx),
evaluated once per policy step. Per-term episode sums are accumulated in a numpy
array and exposed through `episode_sums()` only at episode end, so the step path
never builds a dict.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np

from mujoco_rl_bed.env.cfg import RewardTermCfg

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context

RewardFn = Callable[..., float]
REWARD_TERMS: dict[str, RewardFn] = {}


def reward_term(name: str) -> Callable[[RewardFn], RewardFn]:
    """Decorator registering a reward term.

    Args:
        name: Unique term name.

    Returns:
        The decorator (returns the function unchanged).
    """
    def deco(fn: RewardFn) -> RewardFn:
        if name in REWARD_TERMS:
            raise KeyError(f"Reward term '{name}' already registered")
        REWARD_TERMS[name] = fn
        return fn
    return deco


class RewardManager:
    """Weighted sum of reward terms with per-term episode bookkeeping."""

    def __init__(self, cfg: dict[str, RewardTermCfg], ctx: "Context") -> None:
        """Resolve and bind terms; terms with weight 0 are dropped.

        Args:
            cfg: Name -> reward term config.
            ctx: Shared context.
        """
        self.ctx = ctx
        self.names: list[str] = []
        self._fns: list[Callable[["Context"], float]] = []
        weights = []
        for name, tc in cfg.items():
            if tc.weight == 0.0:
                continue
            key = tc.func or name
            if key not in REWARD_TERMS:
                raise KeyError(f"Unknown reward term '{key}'. Registered: {sorted(REWARD_TERMS)}")
            fn = REWARD_TERMS[key]
            self._fns.append(partial(fn, **tc.params) if tc.params else fn)
            self.names.append(name)
            weights.append(tc.weight)
        self._w = weights
        self._n = len(weights)
        self._sums = np.zeros(self._n)

    def compute(self) -> float:
        """Evaluate the weighted reward for the current policy step.

        Returns:
            Scalar reward.
        """
        ctx = self.ctx
        total = 0.0
        sums = self._sums
        for i in range(self._n):
            v = self._w[i] * float(self._fns[i](ctx))
            sums[i] += v
            total += v
        return total

    def episode_sums(self) -> dict[str, float]:
        """Per-term weighted sums over the current episode (call at episode end).

        Returns:
            Dict term name -> weighted episode sum.
        """
        return {n: float(v) for n, v in zip(self.names, self._sums)}

    def reset(self) -> None:
        """Clear episode sums."""
        self._sums.fill(0.0)
