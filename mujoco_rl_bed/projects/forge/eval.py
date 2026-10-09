"""Evaluate a FORGE checkpoint or the scripted (privileged) expert.

Usage (from anywhere, `dqn` env active):
    python projects/forge/eval.py run=<run_dir_name_or_path> episodes=20
    python projects/forge/eval.py run=<run_dir> checkpoint=checkpoints/model_500000_steps.zip render=true
    python projects/forge/eval.py policy=scripted episodes=10 render=true

`run` may be a folder name inside `projects/forge/runs/`, or any path. `policy=scripted` runs with the
dynamics randomization and observation noise off (`forge.task.NO_DR`).
Reports success (final step / any step), time to success, mean/max contact force.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mujoco_rl_bed.rl.evaluate import main

HERE = Path(__file__).resolve().parent
DEFAULTS = {"task": "forge_peg", "run_root": str(HERE / "runs")}


def scripted_factory(env):
    """Build the privileged scripted expert for `policy=scripted`.

    Args:
        env: A `forge_peg` TorqueEnv.

    Returns:
        A callable policy with `reset()`.
    """
    from forge.scripted import ScriptedPegInsert

    return ScriptedPegInsert(env)


if __name__ == "__main__":
    defaults = dict(DEFAULTS)
    if "policy=scripted" in sys.argv[1:]:
        from forge.task import NO_DR

        defaults.update(NO_DR)  # the expert cannot compensate the dead zone (CLI overrides still win)
    main(sys.argv[1:], task_modules=("forge",), defaults=defaults, scripted_factory=scripted_factory)
