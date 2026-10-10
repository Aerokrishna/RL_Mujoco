"""Train the residual policy (default task `compliant_peg`). Runs go to `projects/compliant_RL/runs/`.

Usage (from anywhere, `dqn` env active):
    python projects/compliant_RL/train.py algo=ppo asymmetric=true n_envs=10 total_timesteps=5000000

Keys that are `TrainCfg` fields configure training (see `mujoco_rl_bed.rl.train.TrainCfg`);
every other key is an `EnvCfg` override (e.g. `task.events.randomize_grasp.params.tilt_max_deg=5`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from mujoco_rl_bed.rl.train import main

HERE = Path(__file__).resolve().parent
DEFAULTS = {"task": "compliant_peg", "run_root": str(HERE / "runs")}

if __name__ == "__main__":
    main(sys.argv[1:], task_modules=("compliant_RL",), defaults=DEFAULTS)
