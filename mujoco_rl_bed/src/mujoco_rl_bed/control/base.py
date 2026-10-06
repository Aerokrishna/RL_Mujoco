"""Controller protocol and config base class.

A controller turns a target (set once per policy step) into joint torques (computed
every physics tick). `torque(plant)` is in the hot loop: it must not allocate, log,
or look up names. It returns a controller-owned buffer that the plant copies into `data.ctrl`.

Controller configs are dataclasses with a `build(plant)` factory. Tasks therefore pick
a controller by choosing a config type, and adding a controller needs no registry edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from mujoco_rl_bed.sim.plant import FrankaPlant


@runtime_checkable
class Controller(Protocol):
    """Interface shared by all torque controllers."""

    def set_target(self, **kw: Any) -> None:
        """Update the setpoint and/or gains (called once per policy step).

        Args:
            **kw: Controller-specific fields (e.g. pos, quat, kp, kd).
        """
        ...

    def torque(self, plant: "FrankaPlant") -> np.ndarray:
        """Compute joint torques for the current state (called every physics tick).

        Args:
            plant: The plant, after `mj_step1`.

        Returns:
            Controller-owned buffer of joint torques [Nm], shape (n,), float64.
        """
        ...

    def reset(self, plant: "FrankaPlant") -> None:
        """Reset internal state and set the target to the current pose.

        Args:
            plant: The plant, after `reset` + `mj_forward`.
        """
        ...

    def setpoint(self, out: np.ndarray) -> np.ndarray:
        """Write the current setpoint (for logging/datasets) into `out`.

        Args:
            out: Buffer of shape (setpoint_dim,).

        Returns:
            `out`.
        """
        ...


class ControllerCfg:
    """Marker base class for controller configs (each defines `build`)."""

    def build(self, plant: "FrankaPlant") -> Controller:
        """Instantiate the controller for `plant`.

        Args:
            plant: Plant the controller will drive.

        Returns:
            A `Controller`.
        """
        raise NotImplementedError
