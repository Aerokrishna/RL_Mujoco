"""Open the MuJoCo viewer on the FORGE scene and run a trained or simple policy.

Usage (`conda activate dqn` first; works from any directory):
    python projects/forge/view.py                                  # scripted expert (default)
    python projects/forge/view.py mode=idle                        # hold the reset pose (inspect the scene)
    python projects/forge/view.py mode=random                      # uniform random actions
    python projects/forge/view.py run=<run_folder> episodes=5      # trained checkpoint
    python projects/forge/view.py run=<run_folder> checkpoint=checkpoints/model_500000_steps.zip
    python projects/forge/view.py mode=scripted realtime=false     # as fast as possible

Close the window (or Ctrl+C) to stop. Each episode also prints the eval metrics.
All `eval.py` keys work here (episodes, seed, deterministic, env overrides such as
`task.events.reset_fixed.params.pos_noise_std=0.0025`). `render=true` is the default.
Viewer tips: double-click a body to select it; Ctrl+right-drag applies a force to it;
press Tab / Shift+Tab to toggle the side panels; site group 4 (tcp, peg tip, hole
tip/floor) is shown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mujoco_rl_bed.rl.evaluate import main as eval_main
from mujoco_rl_bed.utils.config import parse_cli

HERE = Path(__file__).resolve().parent
MODES = ("scripted", "idle", "random")


class IdlePolicy:
    """Hold the reset TCP position (action = current TCP offset from the anchor / bounds)."""

    def __init__(self, env) -> None:
        """Cache the env.

        Args:
            env: A `forge_peg` TorqueEnv.
        """
        self.env = env
        self.bounds = np.asarray(env.cfg.task.action.anchor_bounds, dtype=np.float64)
        self.hold = np.zeros(3)
        self.act_dim = env.action_space.shape[0]

    def reset(self) -> None:
        """Remember the TCP position after reset."""
        self.hold = self.env.plant.ee_pos - self.env.ctx.state["fixed_anchor"]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return the hold action.

        Args:
            obs: Unused.

        Returns:
            Action, shape (act_dim,), float32 (success prediction, if present, says "not solved").
        """
        a = -np.ones(self.act_dim, dtype=np.float32)
        a[:3] = np.clip(self.hold / self.bounds, -1.0, 1.0)
        return a


class RandomPolicy:
    """Uniform random actions in [-1, 1]."""

    def __init__(self, env) -> None:
        """Seeded RNG.

        Args:
            env: A TorqueEnv.
        """
        self.rng = np.random.default_rng(0)
        self.dim = env.action_space.shape[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return a random action.

        Args:
            obs: Unused.

        Returns:
            Action, shape (dim,), float32.
        """
        return self.rng.uniform(-1.0, 1.0, self.dim).astype(np.float32)


def main(argv: list[str]) -> None:
    """Parse `mode`, then delegate to the generic evaluator with rendering on.

    Args:
        argv: key=value arguments.
    """
    raw = parse_cli(argv)
    mode = raw.pop("mode", "scripted")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    defaults = {"task": "forge_peg", "run_root": str(HERE / "runs"), "render": "true", "realtime": "true",
                "episodes": "3"}
    if "run" in raw:
        factory = None  # trained checkpoint
    else:
        defaults["policy"] = "scripted"
        if mode == "scripted":
            from forge.scripted import ScriptedPegInsert as factory
        elif mode == "idle":
            factory = IdlePolicy
        else:
            factory = RandomPolicy
    args = [f"{k}={v}" for k, v in raw.items()]
    eval_main(args, task_modules=("forge",), defaults=defaults, scripted_factory=factory)


if __name__ == "__main__":
    main(sys.argv[1:])
