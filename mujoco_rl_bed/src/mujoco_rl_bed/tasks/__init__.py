"""Task configs. Importing this package registers all built-in tasks."""

from mujoco_rl_bed.tasks import reach  # noqa: F401  (registration side effect)
from mujoco_rl_bed.tasks.registry import TASKS, list_tasks, make_env, make_env_cfg, make_task_cfg, register_task

__all__ = ["TASKS", "list_tasks", "make_env", "make_env_cfg", "make_task_cfg", "register_task"]
