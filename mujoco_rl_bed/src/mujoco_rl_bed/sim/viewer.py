"""Optional passive viewer.

`make_viewer(...)` returns a `NullViewer` when rendering is disabled. In that case
`mujoco.viewer` is never imported (it pulls in GLFW/OpenGL), so a disabled viewer
costs nothing. When enabled, `PassiveViewer` wraps `mujoco.viewer.launch_passive`.
Call `sync()` once per policy step, not once per physics tick. With `realtime=True`,
`sync()` sleeps so that wall-clock time tracks simulation time.

Linux/Wayland: GLFW's Wayland backend hangs or crashes with MuJoCo's viewer here, so
`mujoco_rl_bed/__init__.py` selects the X11 backend (`PYGLFW_LIBRARY_VARIANT=x11`, via
XWayland) before `mujoco` (which loads GLFW) is imported. `PassiveViewer` repeats the
check as a fallback, which only helps if GLFW has not been loaded yet. `close()` waits briefly after closing the window: exiting while the viewer's render
thread is still tearing down segfaults on exit.
"""

from __future__ import annotations

import os
import time
from typing import Any

import mujoco


class NullViewer:
    """No-op viewer used when rendering is disabled."""

    enabled: bool = False

    def sync(self) -> None:
        """Do nothing."""

    def reset_clock(self) -> None:
        """Do nothing."""

    def is_running(self) -> bool:
        """Always True, so loops of the form `while viewer.is_running()` still work headless."""
        return True

    def close(self) -> None:
        """Do nothing."""


class PassiveViewer:
    """Thin wrapper around `mujoco.viewer.launch_passive` with optional real-time pacing.

    Attributes:
        enabled: Always True.
        handle: The underlying `mujoco.viewer.Handle`.
    """

    enabled: bool = True

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, realtime: bool = False,
                 show_sites: bool = True, max_lag: float = 0.2) -> None:
        """Launch the passive viewer window.

        Args:
            model: Model to display (shared with the simulation, not copied).
            data: Data to display (shared with the simulation, not copied).
            realtime: Sleep in `sync()` so that wall time matches simulated time.
            show_sites: Enable site group 4 (where `tcp` and `ft_site` live).
            max_lag: If the sim falls behind wall time by more than this [s], re-anchor the
                clock instead of trying to catch up.
        """
        if os.environ.get("WAYLAND_DISPLAY") and "PYGLFW_LIBRARY_VARIANT" not in os.environ:
            os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"  # must be set before glfw is imported
        import mujoco.viewer  # lazy: only imported when rendering is requested

        self._data = data
        self.realtime = realtime
        self.max_lag = max_lag
        self.handle: Any = mujoco.viewer.launch_passive(model, data)
        if show_sites:
            with self.handle.lock():
                self.handle.opt.sitegroup[4] = 1
        self.reset_clock()

    def reset_clock(self) -> None:
        """Re-anchor real-time pacing (call after `reset`, since `data.time` jumps to 0)."""
        self._wall0 = time.perf_counter()
        self._sim0 = self._data.time

    def sync(self) -> None:
        """Push the current state to the viewer and optionally pace to real time."""
        if self.handle is None:
            return
        self.handle.sync()
        if self.realtime:
            target = self._wall0 + (self._data.time - self._sim0)
            lag = target - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            elif lag < -self.max_lag:
                self.reset_clock()  # too slow to be real time; don't accumulate debt

    def is_running(self) -> bool:
        """Return False once the user closes the window (or after `close()`)."""
        return self.handle is not None and self.handle.is_running()

    def close(self, teardown_wait: float = 1.0) -> None:
        """Close the viewer window and give its render thread time to shut down.

        Args:
            teardown_wait: Seconds to wait after closing (avoids a segfault at interpreter exit).
        """
        if self.handle is None:
            return
        self.handle.close()
        self.handle = None
        time.sleep(teardown_wait)


def make_viewer(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool,
                realtime: bool = False) -> NullViewer | PassiveViewer:
    """Create a viewer, or a zero-cost no-op when disabled.

    Args:
        model: Model to display.
        data: Data to display.
        enabled: Whether to open a window.
        realtime: Pace `sync()` to wall time (only meaningful when enabled).

    Returns:
        `PassiveViewer` if enabled, else `NullViewer`.
    """
    if not enabled:
        return NullViewer()
    return PassiveViewer(model, data, realtime=realtime)
