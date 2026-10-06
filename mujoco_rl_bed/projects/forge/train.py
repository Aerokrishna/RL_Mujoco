"""Train a FORGE policy (default task `forge_peg`). Runs go to `projects/forge/runs/`.

Usage (from anywhere, `dqn` env active):
    python projects/forge/train.py algo=ppo n_envs=10 n_steps=256 batch_size=640 total_timesteps=3000000
    python projects/forge/train.py algo=recurrent_ppo n_envs=10 n_steps=256 batch_size=640 total_timesteps=3000000
    python projects/forge/train.py task.action.max_step=0.015 task.episode_length_s=10

Keys that are `TrainCfg` fields configure training (see `mujoco_rl_bed.rl.train.TrainCfg`);
every other key is an `EnvCfg` override (e.g. `task.rewards.contact_penalty.weight=-0.1`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from mujoco_rl_bed.rl.train import main

HERE = Path(__file__).resolve().parent
DEFAULTS = {"task": "forge_peg", "run_root": str(HERE / "runs")}

if __name__ == "__main__":
    main(sys.argv[1:], task_modules=("forge",), defaults=DEFAULTS)
