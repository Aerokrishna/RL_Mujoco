"""Stable-Baselines3 integration: vectorized env construction and episode-metric logging.

`make_vec_env` builds `n_envs` `TorqueEnv`s (one per subprocess with `SubprocVecEnv`),
seeded deterministically, wrapped in `VecMonitor`. The env observation is already the
flat float32 `policy` group, so no extra flattening wrapper is needed.

`EpisodeInfoCallback` logs every numeric/bool key that `TorqueEnv` puts into `info` at
episode end (is_success, reward_terms/*, task metrics such as contact_force_mean) to
TensorBoard as `episode/<key>` means over the rollout.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

log = logging.getLogger(__name__)

# info keys that are not episode metrics
_SKIP_KEYS = {"critic_obs", "terminal_observation", "TimeLimit.truncated", "episode", "termination"}


def _make_one(task: str, overrides: dict[str, str], seed: int, task_modules: tuple[str, ...] = ()):
    """Construct one env (module-level so it pickles for subprocesses).

    Args:
        task: Registered task name.
        overrides: Dotted EnvCfg overrides.
        seed: Env seed.
        task_modules: Modules to import first so project tasks register inside the subprocess.

    Returns:
        A `TorqueEnv`.
    """
    import importlib

    import torch

    torch.set_num_threads(1)  # workers only step the sim; avoid idle thread pools competing for cores

    import mujoco_rl_bed.tasks  # noqa: F401  (registers core tasks inside the subprocess)
    from mujoco_rl_bed.tasks.registry import make_env

    for name in task_modules:
        importlib.import_module(name)

    ov = dict(overrides)
    ov["seed"] = str(seed)
    return make_env(task, ov)


def make_vec_env(task: str, n_envs: int, seed: int, overrides: dict[str, str] | None = None,
                 vec: str = "subproc", start_method: str | None = None,
                 task_modules: tuple[str, ...] = (), worker_threads: int = 1) -> VecEnv:
    """Create a seeded, monitored vectorized env.

    Rendering is forced off when `n_envs > 1` (one viewer per subprocess is not useful).

    Args:
        task: Registered task name.
        n_envs: Number of parallel envs.
        seed: Base seed; env i is reset with seed + i.
        overrides: Dotted `EnvCfg` overrides (e.g. {"task.action.max_step": "0.015"}).
        vec: "subproc" (one process per env) or "dummy" (in-process, for debugging/eval).
        start_method: Multiprocessing start method for `SubprocVecEnv` (None = SB3 default).
        task_modules: Modules each worker imports to register project tasks (e.g. ("forge",)).
        worker_threads: BLAS/OpenMP threads per worker process (exported as OMP/OPENBLAS/MKL_NUM_THREADS
            before the workers start; the env math is small-matrix, so 1 avoids oversubscription).

    Returns:
        A `VecMonitor`-wrapped `VecEnv`.
    """
    ov = dict(overrides or {})
    render = str(ov.get("render", "false")).lower() in ("1", "true", "yes", "on")
    if render and n_envs > 1:
        log.warning("render=true requested with n_envs=%d; forcing render=false", n_envs)
        ov["render"] = "false"
    fns = [partial(_make_one, task, ov, seed + i, tuple(task_modules)) for i in range(n_envs)]
    if vec == "subproc" and n_envs > 1:
        # Inherited by the worker processes, which load numpy/BLAS fresh. No effect on this
        # process's already-loaded libraries.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(worker_threads)
        venv: VecEnv = SubprocVecEnv(fns, start_method=start_method)
    else:
        venv = DummyVecEnv(fns)
    venv.seed(seed)  # env i -> reset(seed=seed + i) on the first reset
    return VecMonitor(venv)


class EpisodeInfoCallback(BaseCallback):
    """Log episode-end info entries (success, reward terms, task metrics) to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        """Initialize buffers.

        Args:
            verbose: SB3 verbosity.
        """
        super().__init__(verbose)
        self._buf: dict[str, list[float]] = defaultdict(list)

    def _collect(self, prefix: str, d: dict[str, Any]) -> None:
        """Flatten numeric entries of an info dict into the buffer.

        Args:
            prefix: Key prefix.
            d: Info (sub)dict.
        """
        for k, v in d.items():
            if k in _SKIP_KEYS:
                continue
            if isinstance(v, dict):
                self._collect(f"{prefix}{k}/", v)
            elif isinstance(v, (bool, int, float, np.floating, np.integer, np.bool_)):
                self._buf[prefix + k].append(float(v))

    def _on_step(self) -> bool:
        """Collect infos of envs whose episode just ended.

        Returns:
            True (never stops training).
        """
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self._collect("", info)
        return True

    def _on_rollout_end(self) -> None:
        """Write rollout means and clear the buffer."""
        for k, vals in self._buf.items():
            if k == "time_to_success":
                vals = [v for v in vals if v >= 0.0]  # mean over successful episodes only
                if not vals:
                    continue
            self.logger.record(f"episode/{k}", float(np.mean(vals)))
        self._buf.clear()
