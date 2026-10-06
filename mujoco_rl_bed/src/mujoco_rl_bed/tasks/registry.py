"""Task registry: tasks are functions returning a `TaskCfg`, registered by name.

    @register_task("reach")
    def reach() -> TaskCfg: ...

    env = make_env("reach", overrides={"task.episode_length_s": "3", "render": "true"})

Overrides use the dotted `key=value` syntax of `mujoco_rl_bed.utils.config`, rooted at `EnvCfg`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from mujoco_rl_bed.env.cfg import EnvCfg, TaskCfg
from mujoco_rl_bed.utils.config import apply_overrides

if TYPE_CHECKING:
    from mujoco_rl_bed.env.torque_env import TorqueEnv

TASKS: dict[str, Callable[[], TaskCfg]] = {}


def register_task(name: str) -> Callable[[Callable[[], TaskCfg]], Callable[[], TaskCfg]]:
    """Decorator registering a task config factory.

    Args:
        name: Unique task name.

    Returns:
        The decorator (returns the factory unchanged).
    """
    def deco(fn: Callable[[], TaskCfg]) -> Callable[[], TaskCfg]:
        if name in TASKS:
            raise KeyError(f"Task '{name}' already registered")
        TASKS[name] = fn
        return fn
    return deco


def list_tasks() -> list[str]:
    """Return the registered task names.

    Returns:
        Sorted task names.
    """
    return sorted(TASKS)


def make_task_cfg(name: str) -> TaskCfg:
    """Instantiate a fresh `TaskCfg` for a registered task.

    Args:
        name: Task name.

    Returns:
        A new `TaskCfg` (safe to mutate).
    """
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Registered: {list_tasks()}")
    return TASKS[name]()


def make_env_cfg(task: str, overrides: dict[str, str] | None = None) -> EnvCfg:
    """Build an `EnvCfg` for a task and apply dotted overrides.

    Args:
        task: Task name.
        overrides: Mapping such as {"sim_dt": "0.001", "task.action.pos_scale": "0.01"}.

    Returns:
        The resolved `EnvCfg`.
    """
    cfg = EnvCfg(task=make_task_cfg(task), task_name=task)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def make_env(task: str, overrides: dict[str, str] | None = None) -> "TorqueEnv":
    """Create a `TorqueEnv` for a registered task.

    Args:
        task: Task name.
        overrides: Dotted overrides rooted at `EnvCfg`.

    Returns:
        A constructed (not yet reset) `TorqueEnv`.
    """
    from mujoco_rl_bed.env.torque_env import TorqueEnv

    return TorqueEnv(make_env_cfg(task, overrides))
