"""Evaluate a `compliant_peg` checkpoint.

Usage (from anywhere, `dqn` env active):
    python projects/compliant_RL/eval.py run=<run_dir_name_or_path> episodes=20
    python projects/compliant_RL/eval.py run=<run_dir> render=true episodes=5
"""

from __future__ import annotations

import sys
from pathlib import Path

from mujoco_rl_bed.rl.evaluate import main

HERE = Path(__file__).resolve().parent
DEFAULTS = {"task": "compliant_peg", "run_root": str(HERE / "runs")}

if __name__ == "__main__":
    main(sys.argv[1:], task_modules=("compliant_RL",), defaults=DEFAULTS)
