"""Seeding helpers.

Environment randomness always flows through `np.random.default_rng(seed)` held by the
env context. These helpers seed the remaining global sources (Python `random`, legacy
NumPy, torch) for training scripts and derive per-worker seeds for vectorized envs.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, legacy NumPy and torch (if installed) global RNGs.

    Args:
        seed: Base seed.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for pure-sim use
        pass


def worker_seed(base_seed: int, rank: int) -> int:
    """Deterministic, non-overlapping seed for vectorized env worker `rank`.

    Uses `np.random.SeedSequence.spawn` so worker streams are statistically independent.

    Args:
        base_seed: Run seed.
        rank: Worker index (0-based).

    Returns:
        A 32-bit seed.
    """
    child = np.random.SeedSequence(base_seed).spawn(rank + 1)[rank]
    return int(child.generate_state(1)[0])
