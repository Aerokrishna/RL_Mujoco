"""ObservationManager and the `@obs_term` registry.

Term contract: `fn(ctx, out, **params) -> None` writes the term value into `out`, a
float64 scratch buffer of shape (dim,), with no allocation. The manager casts it into
the group's flat float32 buffer, adds noise if the group is noisy, and keeps history.

Accumulated terms (`needs_accumulation=True`, e.g. wrist wrench) are sampled every
physics tick via `accumulate()` and averaged over the policy step. When no active
term needs it, the env skips `accumulate()` entirely.

Buffer reuse: `compute(group)` fills a buffer the manager owns and returns it. The
next call overwrites it, so call `.copy()` if you need to keep the values. `TorqueEnv`
returns copies to Gymnasium/SB3 so terminal observations are never overwritten.
`compute` also advances that group's history by one slot; call it once per policy step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable

import gymnasium as gym
import numpy as np

from mujoco_rl_bed.env.cfg import ObsCfg, ObsTermCfg

if TYPE_CHECKING:
    from mujoco_rl_bed.env.context import Context

ObsFn = Callable[..., None]


@dataclass(frozen=True)
class ObsTermSpec:
    """Registered observation term.

    Attributes:
        name: Term name.
        fn: `fn(ctx, out, **params)`.
        dim: Output size, or a callable `dim(ctx) -> int` for context-dependent sizes.
        needs_accumulation: Average the value over physics ticks within a policy step.
    """

    name: str
    fn: ObsFn
    dim: int | Callable[["Context"], int]
    needs_accumulation: bool = False


OBS_TERMS: dict[str, ObsTermSpec] = {}


def obs_term(name: str, dim: int | Callable[["Context"], int], needs_accumulation: bool = False) -> Callable[[ObsFn], ObsFn]:
    """Decorator registering an observation term.

    Args:
        name: Unique term name used in `ObsCfg.groups`.
        dim: Output size or `dim(ctx)` callable.
        needs_accumulation: Average over physics ticks (the function is then a per-tick sampler).

    Returns:
        The decorator (returns the function unchanged).
    """
    def deco(fn: ObsFn) -> ObsFn:
        if name in OBS_TERMS:
            raise KeyError(f"Observation term '{name}' already registered")
        OBS_TERMS[name] = ObsTermSpec(name, fn, dim, needs_accumulation)
        return fn
    return deco


class _Entry:
    """Compiled per-group term: function, slices, scratch and noise buffers."""

    __slots__ = ("name", "fn", "dim", "history", "noise_std", "acc", "scratch", "hist", "newest", "noise")

    def __init__(self, name: str, fn: Callable, dim: int, history: int, noise_std: float, acc: bool,
                 group_buf: np.ndarray, offset: int) -> None:
        """Bind views into the group buffer.

        Args:
            name: Term name.
            fn: Bound term function `fn(ctx, out)`.
            dim: Term size.
            history: Number of stacked values.
            noise_std: Noise std (0 = none).
            acc: Whether the value comes from the accumulator.
            group_buf: The group's flat float32 buffer.
            offset: Start index of this term in `group_buf`.
        """
        self.name, self.fn, self.dim, self.history = name, fn, dim, history
        self.noise_std, self.acc = noise_std, acc
        self.scratch = np.zeros(dim)
        self.hist = group_buf[offset: offset + dim * history]        # view, oldest -> newest
        self.newest = self.hist[dim * (history - 1):]                 # view of the last slot
        self.noise = np.zeros(dim, dtype=np.float32)


class ObservationManager:
    """Compiles observation groups once and fills flat float32 buffers per policy step."""

    def __init__(self, cfg: ObsCfg, ctx: "Context") -> None:
        """Resolve terms, allocate buffers and slices.

        Args:
            cfg: Observation configuration.
            ctx: Shared context (actions must already be allocated for `last_action`).
        """
        self.ctx = ctx
        self.cfg = cfg
        self.buffers: dict[str, np.ndarray] = {}
        self._entries: dict[str, list[_Entry]] = {}
        self.layout: dict[str, list[tuple[str, slice]]] = {}

        # Accumulators are shared across groups (keyed by term name).
        self._acc_fns: list[Callable] = []
        self._acc_sum: list[np.ndarray] = []
        self._acc_sample: list[np.ndarray] = []
        self._acc_avg: dict[str, np.ndarray] = {}
        self._acc_names: list[str] = []
        self._acc_count = 0

        for group, names in cfg.groups.items():
            specs = []
            size = 0
            for name in names:
                if name not in OBS_TERMS:
                    raise KeyError(f"Unknown observation term '{name}' in group '{group}'. "
                                   f"Registered: {sorted(OBS_TERMS)}")
                spec = OBS_TERMS[name]
                tcfg = cfg.term_cfg.get(name, ObsTermCfg())
                dim = spec.dim(ctx) if callable(spec.dim) else int(spec.dim)
                if tcfg.history < 1:
                    raise ValueError(f"history for '{name}' must be >= 1")
                specs.append((spec, tcfg, dim))
                size += dim * tcfg.history
            buf = np.zeros(size, dtype=np.float32)
            entries, layout, off = [], [], 0
            for spec, tcfg, dim in specs:
                fn = partial(spec.fn, **tcfg.params) if tcfg.params else spec.fn
                noise = tcfg.noise_std if group in cfg.noisy_groups else 0.0
                entries.append(_Entry(spec.name, fn, dim, tcfg.history, noise, spec.needs_accumulation, buf, off))
                layout.append((spec.name, slice(off, off + dim * tcfg.history)))
                off += dim * tcfg.history
                if spec.needs_accumulation and spec.name not in self._acc_avg:
                    self._acc_names.append(spec.name)
                    self._acc_fns.append(fn)
                    self._acc_sum.append(np.zeros(dim))
                    self._acc_sample.append(np.zeros(dim))
                    self._acc_avg[spec.name] = np.zeros(dim)
            self.buffers[group] = buf
            self._entries[group] = entries
            self.layout[group] = layout

        self.needs_accumulation: bool = bool(self._acc_fns)
        self._n_acc = len(self._acc_fns)

    # ------------------------------------------------------------------ queries
    def has_group(self, group: str) -> bool:
        """Return whether `group` is configured.

        Args:
            group: Group name.

        Returns:
            True if the group exists.
        """
        return group in self.buffers

    def dim(self, group: str) -> int:
        """Flat size of a group.

        Args:
            group: Group name.

        Returns:
            Number of float32 entries.
        """
        return int(self.buffers[group].size)

    def space(self, group: str) -> gym.spaces.Box:
        """Unbounded float32 Box for a group.

        Args:
            group: Group name.

        Returns:
            `Box(-inf, inf, (dim,), float32)`.
        """
        return gym.spaces.Box(-np.inf, np.inf, shape=(self.dim(group),), dtype=np.float32)

    # ------------------------------------------------------------------ per tick
    def accumulate(self) -> None:
        """Sample accumulated terms once (called every physics tick when needed)."""
        ctx = self.ctx
        for i in range(self._n_acc):
            self._acc_fns[i](ctx, self._acc_sample[i])
            np.add(self._acc_sum[i], self._acc_sample[i], out=self._acc_sum[i])
        self._acc_count += 1

    def _finalize_accumulators(self) -> None:
        """Turn running sums into averages (no-op if nothing was accumulated since the last call)."""
        if self._acc_count == 0:
            return
        inv = 1.0 / self._acc_count
        for i, name in enumerate(self._acc_names):
            np.multiply(self._acc_sum[i], inv, out=self._acc_avg[name])
            self._acc_sum[i].fill(0.0)
        self._acc_count = 0

    def accumulated(self, name: str) -> np.ndarray:
        """Step-averaged value of an accumulated term (for rewards/metrics that need the same signal).

        Finalizes the running sums if new ticks were accumulated since the last call. Before
        any tick of an episode it holds the instantaneous sample taken by `reset()`.

        Args:
            name: Name of an active term registered with `needs_accumulation=True`.

        Returns:
            Manager-owned float64 buffer of shape (dim,), overwritten next step.
        """
        if name not in self._acc_avg:
            raise KeyError(f"'{name}' is not an active accumulated observation term "
                           f"(active: {self._acc_names}); add it to an observation group")
        self._finalize_accumulators()
        return self._acc_avg[name]

    # ------------------------------------------------------------------ per policy step
    def compute(self, group: str) -> np.ndarray:
        """Fill and return the group's flat buffer (owned; overwritten next call).

        Args:
            group: Group name.

        Returns:
            float32 buffer of shape (dim(group),).
        """
        if self._n_acc:
            self._finalize_accumulators()
        ctx = self.ctx
        for e in self._entries[group]:
            if e.acc:
                src = self._acc_avg[e.name]
            else:
                e.fn(ctx, e.scratch)
                src = e.scratch
            if e.history > 1:
                e.hist[:-e.dim] = e.hist[e.dim:]  # shift history left by one slot
            np.copyto(e.newest, src, casting="same_kind")  # float64 -> float32
            if e.noise_std > 0.0:
                ctx.rng.standard_normal(dtype=np.float32, out=e.noise)
                e.noise *= e.noise_std
                e.newest += e.noise
        return self.buffers[group]

    def reset(self) -> None:
        """Reset accumulators and fill every history slot with the current value."""
        ctx = self.ctx
        for i, name in enumerate(self._acc_names):
            self._acc_sum[i].fill(0.0)
            self._acc_fns[i](ctx, self._acc_avg[name])  # instantaneous sample as the initial value
        self._acc_count = 0
        for group, entries in self._entries.items():
            self.compute(group)
            for e in entries:
                if e.history > 1:
                    for k in range(e.history - 1):
                        e.hist[k * e.dim:(k + 1) * e.dim] = e.newest
