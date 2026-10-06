"""Train PPO / RecurrentPPO on a core task (e.g. `reach`). Project tasks have their own entry point.

Usage:
    python scripts/train_ppo.py task=reach seed=0 n_envs=8 total_timesteps=50000
    python scripts/train_ppo.py task=reach algo=recurrent_ppo render=false device=auto

See `mujoco_rl_bed.rl.train` for all options. Runs go to `runs/` in the repository root.
"""

from __future__ import annotations

import sys

from mujoco_rl_bed.rl.train import main

if __name__ == "__main__":
    main(sys.argv[1:])
