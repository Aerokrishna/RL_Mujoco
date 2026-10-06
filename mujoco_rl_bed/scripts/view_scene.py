"""Launch the torque-controlled Franka scene for visual sanity checks.

Modes:
    hold      Cartesian impedance holding the initial TCP pose (default gains).
    gravcomp  tau = qfrc_bias (gravity + Coriolis compensation only). The arm should
              stay still; any drift comes only from integration error and joint damping.
    zero      tau = 0. The arm should fall under gravity (checks the actuator swap).

Usage:
    python scripts/view_scene.py                       # hold, real time, viewer
    python scripts/view_scene.py mode=gravcomp
    python scripts/view_scene.py mode=zero
    python scripts/view_scene.py headless=true duration=5    # no window, prints drift
    python scripts/view_scene.py scene.joint_damping=0 sim_dt=0.001

Every `print_every` seconds of sim time it prints the max joint drift [mrad] and the
TCP displacement [mm] relative to the initial pose.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from mujoco_rl_bed.control.cartesian_impedance import CartesianImpedanceCfg
from mujoco_rl_bed.sim.plant import FrankaPlant
from mujoco_rl_bed.sim.scene import SceneBuilder, SceneCfg
from mujoco_rl_bed.sim.viewer import make_viewer
from mujoco_rl_bed.utils.config import apply_overrides, parse_cli


@dataclass
class ViewCfg:
    """Options for this script (all overridable as key=value).

    Attributes:
        mode: "hold", "gravcomp" or "zero".
        sim_dt: Physics timestep [s].
        realtime: Pace the viewer to wall time.
        headless: Run without a window (prints drift only).
        duration: Stop after this much sim time [s]; 0 = until the window is closed.
        sync_hz: Viewer refresh rate [Hz] (sync every round(1 / (sync_hz * sim_dt)) ticks).
        print_every: Drift report period [s] of sim time.
        scene: Scene configuration.
        controller: Cartesian impedance configuration used by mode=hold.
    """

    mode: str = "hold"
    sim_dt: float = 0.002
    realtime: bool = True
    headless: bool = False
    duration: float = 0.0
    sync_hz: float = 60.0
    print_every: float = 1.0
    scene: SceneCfg = field(default_factory=SceneCfg)
    controller: CartesianImpedanceCfg = field(default_factory=CartesianImpedanceCfg)


def main(argv: list[str]) -> None:
    """Build the scene and run the selected mode.

    Args:
        argv: `key=value` overrides.
    """
    cfg = apply_overrides(ViewCfg(), parse_cli(argv))
    model, handles = SceneBuilder(cfg.scene, sim_dt=cfg.sim_dt).build()
    plant = FrankaPlant(model, handles)
    plant.reset()

    print(f"arm joints   : {handles.arm_joint_names}")
    print(f"torque limits: {handles.torque_limits.tolist()} Nm")
    print(f"q_home       : {handles.q_home.tolist()}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} dt={model.opt.timestep} integrator={model.opt.integrator}")
    print(f"tcp at {plant.ee_pos.round(4).tolist()} m, mode={cfg.mode}")

    zero = np.zeros(plant.n)
    if cfg.mode == "hold":
        controller = cfg.controller.build(plant)
        controller.reset(plant)
        torque_fn = controller.torque
    elif cfg.mode == "gravcomp":
        def torque_fn(p: FrankaPlant) -> np.ndarray:
            """Gravity + Coriolis compensation: tau = qfrc_bias (view, no allocation)."""
            return p.qfrc_bias
    elif cfg.mode == "zero":
        def torque_fn(p: FrankaPlant) -> np.ndarray:
            """Zero torque: the arm falls under gravity."""
            return zero
    else:
        raise ValueError(f"unknown mode '{cfg.mode}' (hold | gravcomp | zero)")

    viewer = make_viewer(model, plant.data, enabled=not cfg.headless, realtime=cfg.realtime)
    q0 = plant.q.copy()
    x0 = plant.ee_pos.copy()
    sync_every = max(1, round(1.0 / (cfg.sync_hz * cfg.sim_dt)))
    print_every = max(1, round(cfg.print_every / cfg.sim_dt))
    max_ticks = round(cfg.duration / cfg.sim_dt) if cfg.duration > 0 else -1
    tick = 0
    try:
        while viewer.is_running() and tick != max_ticks:
            plant.control_step(torque_fn)
            tick += 1
            if tick % sync_every == 0:
                viewer.sync()
            if tick % print_every == 0:
                dq = np.abs(plant.q - q0).max() * 1e3
                dx = np.linalg.norm(plant.ee_pos - x0) * 1e3
                print(f"t={plant.time:7.2f}s  max|dq|={dq:8.3f} mrad  |dx_tcp|={dx:8.3f} mm")
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
