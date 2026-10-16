"""Hole trigger: a classifier that tells from the sensed contact state when the peg tip is within ~1 mm of the hole.

Used by `compliant_peg_s3` to end the spiral search and start the local x, y, yaw phase (`spiral_local`). It reads
the env's `trigger` observation group (`compliant_peg_s2f` actor layout, with sensor noise; 21 values): contact
wrench (6), spring deflection (2), TCP lateral velocity (2), roll/pitch (2), spiral offset (2), depth (1), phase (1),
try state (2), previous action (3). Try state and previous action are zeroed (they were zero in the training data).

Model (`models/hole_trigger_classifier.pkl`, scikit-learn 1.7.0 `HistGradientBoostingClassifier`): inputs are the
last 4 search-phase observations as [x_t, mean, std, x_t - x_(t-3)] (84 values; padded with the first observation at
the start); label tip within 1 mm of the hole axis vs farther than 1.5 mm (peg not yet dropping). Fires when the
probability exceeds 0.9 for 1 step. Trained on 782 plain-spiral episodes (312k steps): cylinder / cuboid / hexagon /
octagon, 8 mm, radial clearance N(0.3, 0.1) mm clipped to [0.1, 0.5], friction U(0.3, 1.0), hole yaw error up to
half the shape's symmetry angle, hole-estimate error U(0, 4 mm), no tilt. Held-out conditions (shape + clearance):
ROC AUC 0.81; fires within 1.5 mm of the hole in 59 % of episodes at 0.11 false firings per episode (catch rule
34 %): cylinder 94 %, octagon 74 %, hexagon 45 %, cuboid 24 %; strong for small yaw errors (73 % below a third of
the maximum) and wide clearances (81 % at 0.4-0.5 mm), weak for large yaw errors and 0.1-0.2 mm.
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
MODEL = MODELS / "hole_trigger_classifier.pkl"
ZEROED = slice(16, 21)                   # try state (2) + previous action (3) in the `trigger` group


def model_path(name: str | Path = "") -> Path:
    """A trigger model: "" = the default `MODEL`, an existing path, or a file name in `models/`."""
    if not str(name):
        return MODEL
    p = Path(name)
    return p if p.is_absolute() or p.exists() else MODELS / p


class HoleTrigger:
    """Fires (returns True) when the classifier's probability exceeds `th` for `k` consecutive search steps.

    Call `reset()` at every episode start and `__call__(obs)` on every search-phase step with
    `env.compute_obs("trigger")`; steps outside the search phase should not be passed (the window then continues
    with the next search step).
    """

    def __init__(self, path: str | Path = MODEL, th: float | None = None, k: int | None = None) -> None:
        """Load the model.

        Args:
            path: Pickled model dict (`clf`, `W`, `TH`, `K`).
            th: Probability threshold (default: the model's `TH`).
            k: Consecutive steps above it (default: the model's `K`).
        """
        with open(model_path(path), "rb") as f:
            d = pickle.load(f)
        self.clf, self.w = d["clf"], int(d["W"])
        self.th = float(d["TH"] if th is None else th)
        self.k = int(d["K"] if k is None else k)
        self.reset()

    def reset(self) -> None:
        """Clear the observation window and the run counter."""
        self.win: deque = deque(maxlen=self.w)
        self.run = 0
        self.p = 0.0

    def prob(self, obs: np.ndarray) -> float:
        """Append one search-phase observation and return the probability that the tip is within 1 mm.

        Args:
            obs: `trigger` group observation, shape (21,).

        Returns:
            Probability.
        """
        o = np.array(obs, dtype=np.float64)
        o[ZEROED] = 0.0
        self.win.append(o)
        w = list(self.win)
        stack = np.stack([w[0]] * (self.w - len(w)) + w)       # oldest first
        x = np.concatenate([stack[-1], stack.mean(0), stack.std(0), stack[-1] - stack[0]])
        self.p = float(self.clf.predict_proba(x[None])[0, 1])
        return self.p

    def __call__(self, obs: np.ndarray) -> bool:
        """Update with one search-phase observation; True when the trigger fires (the run counter then restarts).

        Args:
            obs: `trigger` group observation, shape (21,).

        Returns:
            Whether to start the local phase.
        """
        self.run = self.run + 1 if self.prob(obs) > self.th else 0
        if self.run >= self.k:
            self.run = 0
            return True
        return False
