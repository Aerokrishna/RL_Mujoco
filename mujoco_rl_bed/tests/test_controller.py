"""Controller tests: hold drift, stiffness under an external force, torque limits, stability.

All tests run headless on the default scene with sim_dt = 0.002 s.
"""

from __future__ import annotations

import gc
import tracemalloc

import mujoco
import numpy as np
import pytest

from mujoco_rl_bed.control.cartesian_impedance import CartesianImpedance, CartesianImpedanceCfg
from mujoco_rl_bed.control.joint_impedance import JointImpedanceCfg
from mujoco_rl_bed.control.math_utils import DampedLeastSquares, GravityTorque, damped_pinv
from mujoco_rl_bed.sim.plant import FrankaPlant
from mujoco_rl_bed.sim.scene import SceneBuilder, SceneCfg

DT = 0.002


def make_plant() -> FrankaPlant:
    """Build a fresh plant at the home pose.

    Returns:
        A reset `FrankaPlant`.
    """
    model, handles = SceneBuilder(SceneCfg(), sim_dt=DT).build()
    plant = FrankaPlant(model, handles)
    plant.reset()
    return plant


def run(plant: FrankaPlant, ctrl, seconds: float, hook=None) -> None:
    """Simulate `seconds` of closed-loop control.

    Args:
        plant: Plant to step.
        ctrl: Controller providing `torque`.
        seconds: Duration [s].
        hook: Optional callable(plant) run before every tick (e.g. to set xfrc_applied).
    """
    for _ in range(round(seconds / DT)):
        if hook is not None:
            hook(plant)
        plant.control_step(ctrl.torque)


def test_hold_pose_drift_below_1mm() -> None:
    """(a) Holding the initial pose for 2 s drifts less than 1 mm."""
    plant = make_plant()
    ctrl = CartesianImpedanceCfg().build(plant)
    ctrl.reset(plant)
    x0 = plant.ee_pos.copy()
    run(plant, ctrl, 2.0)
    assert np.linalg.norm(plant.ee_pos - x0) < 1e-3


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_external_force_displacement_matches_stiffness(axis: int) -> None:
    """(b) A constant force F on the TCP gives a steady-state displacement of about F / Kp (within 10%)."""
    plant = make_plant()
    ctrl = CartesianImpedanceCfg().build(plant)
    ctrl.reset(plant)
    x0 = plant.ee_pos.copy()
    F = np.zeros(3)
    F[axis] = 10.0  # [N]
    bid = plant.h.ee_body_id
    d = plant.data

    def apply_force(p: FrankaPlant) -> None:
        """Apply F at the TCP: xfrc_applied acts at the body COM, so add the moment (p_tcp - com) x F."""
        d.xfrc_applied[bid, :3] = F
        d.xfrc_applied[bid, 3:] = np.cross(p.ee_pos - d.xipos[bid], F)

    run(plant, ctrl, 3.0, hook=apply_force)
    disp = (plant.ee_pos - x0)[axis]
    expected = F[axis] / ctrl.kp[axis]
    assert abs(disp - expected) < 0.1 * expected, f"disp={disp:.5f} expected={expected:.5f}"


def test_commanded_torques_within_limits() -> None:
    """(c) Even for a far target plus a disturbance, commanded torques never exceed the limits."""
    plant = make_plant()
    ctrl = CartesianImpedanceCfg(kp=(3000.0,) * 3 + (300.0,) * 3).build(plant)
    ctrl.reset(plant)
    ctrl.set_target(pos=plant.ee_pos + np.array([0.3, 0.3, -0.3]))
    lim = plant.torque_limits
    worst = 0.0
    for k in range(round(1.5 / DT)):
        plant.data.xfrc_applied[plant.h.ee_body_id, :3] = (50.0, 0.0, 0.0) if k < 300 else (0.0, 0.0, 0.0)
        plant.control_step(ctrl.torque)
        worst = max(worst, float(np.max(np.abs(ctrl.last_torque) - lim)))
    assert worst <= 1e-9


@pytest.mark.parametrize("compensate_coriolis", [True, False])
def test_stable_step_response_default_gains(compensate_coriolis: bool) -> None:
    """(d) At default gains with sim_dt = 0.002, a 10 cm step converges and settles."""
    plant = make_plant()
    ctrl = CartesianImpedanceCfg(compensate_coriolis=compensate_coriolis).build(plant)
    ctrl.reset(plant)
    goal = plant.ee_pos + np.array([0.1, -0.05, -0.05])
    ctrl.set_target(pos=goal)
    run(plant, ctrl, 3.0)
    assert np.all(np.isfinite(plant.data.qpos))
    assert np.linalg.norm(plant.ee_pos - goal) < 2e-3
    assert np.linalg.norm(plant.qd) < 1e-2


def test_joint_impedance_holds_and_tracks() -> None:
    """Joint impedance holds home and tracks a small joint step."""
    plant = make_plant()
    ctrl = JointImpedanceCfg().build(plant)
    ctrl.reset(plant)
    q_goal = plant.q + 0.1
    ctrl.set_target(q=q_goal)
    run(plant, ctrl, 3.0)
    assert np.max(np.abs(plant.q - q_goal)) < 5e-3


def test_damped_least_squares_matches_reference() -> None:
    """The in-place null-space projector matches I - J^T J#^T computed with numpy."""
    rng = np.random.default_rng(0)
    J = rng.standard_normal((6, 7))
    tau0 = rng.standard_normal(7)
    dls = DampedLeastSquares(6, 7, 1e-2)
    dls.factor(J)
    out = np.zeros(7)
    dls.project_nullspace(J, tau0, out)
    ref = (np.eye(7) - J.T @ damped_pinv(J, 1e-2).T) @ tau0
    np.testing.assert_allclose(out, ref, atol=1e-9)


def test_gravity_torque_matches_bias_at_rest() -> None:
    """At zero velocity the gravity-only torque equals qfrc_bias on the arm."""
    plant = make_plant()
    mujoco.mj_forward(plant.model, plant.data)
    jids = [plant.h.joint_ids[n] for n in plant.h.arm_joint_names]
    g = GravityTorque(plant.model, plant.data, jids).compute()
    np.testing.assert_allclose(g, plant.qfrc_bias, atol=1e-8)


def test_hot_loop_does_not_allocate_arrays() -> None:
    """control_step + CartesianImpedance.torque allocates no array buffers after warm-up.

    tracemalloc also sees short-lived Python objects (views, floats), so the check is
    on net growth and a small peak bound rather than exactly zero bytes.
    """
    plant = make_plant()
    ctrl: CartesianImpedance = CartesianImpedanceCfg(torque_rate_limit=5.0).build(plant)
    ctrl.reset(plant)
    run(plant, ctrl, 0.05)  # warm-up
    gc.collect()
    gc.disable()  # a collection mid-measurement shows up as unrelated traced memory (flaky)
    try:
        tracemalloc.start()
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        run(plant, ctrl, 0.2)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        gc.enable()
    assert cur - base < 1024, f"net growth {cur - base} B"
    assert peak - base < 4096, f"peak {peak - base} B"
