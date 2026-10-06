"""Reusable term library. Importing this package registers all built-in terms.

Paper/task-specific terms live in their project package (e.g. `projects/forge/terms.py`)
and register themselves when that package is imported.
"""

from mujoco_rl_bed.terms import events, obs, rewards, terminations  # noqa: F401  (registration side effects)
