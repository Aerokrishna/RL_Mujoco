"""mujoco_rl_bed: modular torque-controlled Franka simulation framework for RL and diffusion policies."""

import os as _os

# `import mujoco` loads GLFW immediately. On Wayland desktops GLFW's Wayland backend hangs
# or crashes with MuJoCo's viewer, so prefer the X11 backend (via XWayland) unless the
# user has chosen a variant. This must run before mujoco is first imported; every module
# in this package imports `mujoco_rl_bed` first. Set PYGLFW_LIBRARY_VARIANT=wayland to opt out.
if _os.environ.get("WAYLAND_DISPLAY") and "PYGLFW_LIBRARY_VARIANT" not in _os.environ:
    _os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"

__version__ = "0.1.0"
