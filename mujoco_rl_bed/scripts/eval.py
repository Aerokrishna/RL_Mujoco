"""Evaluate a checkpoint of a core task.

Usage:
    python scripts/eval.py run=runs/<run_dir> episodes=20 render=true

Runs from projects can also be evaluated here: their config.json lists the modules to import.
See `mujoco_rl_bed.rl.evaluate` for all options.
"""

from __future__ import annotations

import sys

from mujoco_rl_bed.rl.evaluate import main

if __name__ == "__main__":
    main(sys.argv[1:])
