"""Scene composition: build a torque-controlled Franka model with MjSpec.

`SceneBuilder` loads the read-only Franka MJCF (via `scene.xml`, which includes the
robot and adds floor/lights), then:

1. discovers the arm joints from the MJCF (hinge joints driven by joint-transmission
   actuators) instead of assuming names,
2. reads each arm actuator's resolved `forcerange` from the compiled original model
   and replaces those position servos with `motor` actuators (gear 1,
   `ctrlrange` = torque limit),
3. keeps any remaining actuator (the gripper tendon actuator) untouched,
4. adds a `tcp` site (configurable offset) and an `ft_site` at the flange carrying
   `force` and `torque` sensors,
5. drops keyframes (their `ctrl` vectors no longer match the actuator layout),
6. lets task assets add bodies/geoms/sites through the `SceneAsset` protocol,
7. compiles and resolves every name the hot path needs into a `SceneHandles` record.

Nothing here runs per step; all work happens once at env construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import mujoco
import numpy as np

# Repository root: src/mujoco_rl_bed/sim/scene.py -> parents[3] is the project directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
GENERATED_DIR: Path = ASSETS_DIR / "generated"

# Home pose from assets/franka/franka.yaml (arms.left.q_home) [rad].
FRANKA_Q_HOME: tuple[float, ...] = (0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816)

TCP_SITE: str = "tcp"
FT_SITE: str = "ft_site"
FT_FORCE_SENSOR: str = "ft_force"
FT_TORQUE_SENSOR: str = "ft_torque"


@runtime_checkable
class SceneAsset(Protocol):
    """A task asset that adds elements to the scene spec before compilation.

    Implementations are plain config dataclasses (so they serialize into config.json)
    with a `build` method. Names they create should be prefixed with `name` to avoid
    collisions; the env resolves them to ids once via `SceneHandles.body_ids` etc.
    """

    name: str

    def build(self, spec: mujoco.MjSpec, ee_body: mujoco.MjsBody) -> None:
        """Add bodies/geoms/sites/sensors to `spec`.

        Args:
            spec: The scene spec being composed (world = `spec.worldbody`).
            ee_body: The end-effector body spec (`SceneCfg.ee_body`), for hand-mounted assets.
        """
        ...


@dataclass
class SceneCfg:
    """Configuration for `SceneBuilder`.

    Attributes:
        scene_xml: Path to the base scene MJCF, absolute or relative to `assets/`.
        ee_body: Body that carries the `tcp` and `ft_site` sites (the Franka hand).
        arm_joints: Arm joint names in kinematic order; None = auto-discover.
        tcp_offset: `tcp` site position in the `ee_body` frame [m] (default: fingertip center).
        ft_site_offset: `ft_site` position in the `ee_body` frame [m] (default: flange).
        q_home: Home joint configuration [rad], shape (n_arm,).
        joint_damping: Arm joint damping [Nms/rad]; scalar, per-joint tuple, or None = keep MJCF.
        joint_armature: Arm joint armature [kg m^2]; scalar, per-joint tuple, or None = keep MJCF.
        torque_limits: Per-joint torque limits [Nm]; None = read from the MJCF actuators.
        integrator: MuJoCo integrator name ("implicitfast", "euler", ...); None = keep MJCF.
        cone: Friction cone ("pyramidal" or "elliptic"); None = keep MJCF.
        impratio: Frictional-to-normal impedance ratio (>1 reduces slip with elliptic cones); None = keep.
        gripper_open: Initial gripper opening in [0, 1] (1 = fully open).
        assets: Task assets (objects implementing `SceneAsset`).
        dump_xml: If True, write the composed model to `assets/generated/<model_name>.xml`.
    """

    scene_xml: str = "franka/scene.xml"
    ee_body: str = "hand"
    arm_joints: tuple[str, ...] | None = None
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    ft_site_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    q_home: tuple[float, ...] = FRANKA_Q_HOME
    joint_damping: float | tuple[float, ...] | None = None
    joint_armature: float | tuple[float, ...] | None = None
    torque_limits: tuple[float, ...] | None = None
    integrator: str | None = None
    cone: str | None = None
    impratio: float | None = None
    gripper_open: float = 1.0
    assets: list = field(default_factory=list)
    dump_xml: bool = False


@dataclass(frozen=True)
class SceneHandles:
    """Ids and index ranges resolved once from the compiled model.

    Attributes:
        arm_joint_names: Arm joint names in kinematic order, length n_arm.
        n_arm: Number of arm joints (7 for the Franka).
        arm_qpos: Contiguous slice of `data.qpos` for the arm.
        arm_dof: Contiguous slice of `data.qvel` / `qfrc_*` / Jacobian columns for the arm.
        arm_ctrl: Contiguous slice of `data.ctrl` for the arm motors.
        torque_limits: Per-joint torque limits [Nm], shape (n_arm,), float64.
        q_home: Home configuration [rad], shape (n_arm,), float64.
        gripper_act_id: Gripper actuator id, or -1 if the robot has none.
        gripper_ctrlrange: Gripper ctrl range (lo, hi), shape (2,).
        gripper_open: Initial gripper opening in [0, 1].
        finger_qpos_adr: qpos addresses of the finger joints, shape (k,), int.
        finger_qpos_range: Finger joint ranges, shape (k, 2) [m].
        ee_body_id: Body id of the end-effector body.
        tcp_site_id: Site id of `tcp`.
        ft_site_id: Site id of `ft_site`.
        ft_force: Slice of `data.sensordata` with the flange force (site frame) [N].
        ft_torque: Slice of `data.sensordata` with the flange torque (site frame) [Nm].
        body_ids: Name -> id for every named body (init-time lookups only).
        site_ids: Name -> id for every named site.
        geom_ids: Name -> id for every named geom.
        joint_ids: Name -> id for every named joint.
    """

    arm_joint_names: tuple[str, ...]
    n_arm: int
    arm_qpos: slice
    arm_dof: slice
    arm_ctrl: slice
    torque_limits: np.ndarray
    q_home: np.ndarray
    gripper_act_id: int
    gripper_ctrlrange: np.ndarray
    gripper_open: float
    finger_qpos_adr: np.ndarray
    finger_qpos_range: np.ndarray
    ee_body_id: int
    tcp_site_id: int
    ft_site_id: int
    ft_force: slice
    ft_torque: slice
    body_ids: dict[str, int]
    site_ids: dict[str, int]
    geom_ids: dict[str, int]
    joint_ids: dict[str, int]


def _resolve_asset_path(path: str) -> Path:
    """Resolve an asset path.

    Args:
        path: Absolute path, or path relative to `assets/`.

    Returns:
        Absolute path to an existing file.
    """
    p = Path(path)
    if not p.is_absolute():
        p = ASSETS_DIR / p
    if not p.is_file():
        raise FileNotFoundError(f"Scene MJCF not found: {p}")
    return p


def _per_joint(value: float | tuple[float, ...] | None, n: int, what: str) -> np.ndarray | None:
    """Broadcast a scalar-or-sequence override to shape (n,).

    Args:
        value: Scalar, length-n sequence, or None.
        n: Number of arm joints.
        what: Name used in error messages.

    Returns:
        float64 array of shape (n,), or None if `value` is None.
    """
    if value is None:
        return None
    arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)) if np.ndim(value) == 0 else np.asarray(value, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{what} must be a scalar or length-{n} sequence, got shape {arr.shape}")
    return arr.copy()


def _contiguous_slice(idx: list[int], what: str) -> slice:
    """Convert a list of consecutive indices into a slice (views instead of fancy indexing).

    Args:
        idx: Indices that must be strictly consecutive.
        what: Name used in error messages.

    Returns:
        slice(idx[0], idx[-1] + 1).
    """
    if list(idx) != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError(f"{what} indices are not contiguous: {idx}")
    return slice(idx[0], idx[0] + len(idx))


def _name_map(model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int) -> dict[str, int]:
    """Map every named object of one type to its id.

    Args:
        model: Compiled model.
        objtype: MuJoCo object type.
        count: Number of objects of that type.

    Returns:
        Dict name -> id (unnamed objects are skipped).
    """
    out: dict[str, int] = {}
    for i in range(count):
        name = mujoco.mj_id2name(model, objtype, i)
        if name:
            out[name] = i
    return out


def _is_descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    """Return True if `body` is `ancestor` or lies in its subtree.

    Args:
        model: Compiled model.
        body: Body id to test.
        ancestor: Candidate ancestor body id.

    Returns:
        Whether `body` is in the subtree rooted at `ancestor`.
    """
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return ancestor == 0


class SceneBuilder:
    """Compose the torque-controlled Franka scene and resolve handles.

    Usage:
        model, handles = SceneBuilder(SceneCfg(), sim_dt=0.002).build()

    Each call to `build()` returns a fresh `MjModel`, so every env (and every
    subprocess) owns its model and may randomize parameters in place.
    """

    def __init__(self, cfg: SceneCfg, sim_dt: float = 0.002) -> None:
        """Store configuration.

        Args:
            cfg: Scene configuration.
            sim_dt: Physics timestep [s].
        """
        self.cfg = cfg
        self.sim_dt = float(sim_dt)
        self.spec: mujoco.MjSpec | None = None  # kept after build() for inspection/debugging

    # ------------------------------------------------------------------ discovery
    def _discover_arm(self, model: mujoco.MjModel) -> tuple[list[str], list[str], np.ndarray]:
        """Find arm joints, their actuators and torque limits in the original model.

        Arm joints are hinge joints driven by joint-transmission actuators (the Franka
        position servos); everything else (e.g. the gripper tendon actuator) is left alone.

        Args:
            model: Compiled original (unmodified) model.

        Returns:
            (joint_names, actuator_names, torque_limits[n_arm]) in joint-id order.
        """
        cfg = self.cfg
        pairs: list[tuple[int, int]] = []  # (joint id, actuator id)
        for a in range(model.nu):
            # Compare as plain ints: from MuJoCo 3.15, mjt* enums no longer compare equal to numpy ints
            # in all forms (e.g. `np.int32 in (enum, ...)` is False), which silently broke detection.
            if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            j = int(model.actuator_trnid[a, 0])
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                pairs.append((j, a))
        pairs.sort()
        jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j, _ in pairs]
        anames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for _, a in pairs]

        if cfg.arm_joints is not None:
            wanted = list(cfg.arm_joints)
            missing = [n for n in wanted if n not in jnames]
            if missing:
                raise ValueError(f"arm_joints {missing} have no joint actuator; found {jnames}")
            keep = [jnames.index(n) for n in wanted]
            pairs = [pairs[k] for k in keep]
            jnames = [jnames[k] for k in keep]
            anames = [anames[k] for k in keep]
        if not pairs:
            raise RuntimeError("No arm joints with joint-transmission actuators found in the MJCF.")

        limits = np.zeros(len(pairs))
        for k, (j, a) in enumerate(pairs):
            if model.actuator_forcelimited[a]:
                limits[k] = max(abs(model.actuator_forcerange[a, 0]), abs(model.actuator_forcerange[a, 1]))
            elif model.jnt_actfrclimited[j]:
                limits[k] = max(abs(model.jnt_actfrcrange[j, 0]), abs(model.jnt_actfrcrange[j, 1]))
        if cfg.torque_limits is not None:
            limits = _per_joint(cfg.torque_limits, len(pairs), "torque_limits")
        if np.any(limits <= 0):
            raise ValueError(f"Missing torque limit for arm joints {jnames}: {limits}. Set SceneCfg.torque_limits.")
        return jnames, anames, limits

    # ------------------------------------------------------------------ build
    def build(self) -> tuple[mujoco.MjModel, SceneHandles]:
        """Compose, compile and resolve the scene.

        Returns:
            (model, handles): the compiled model and the resolved id/slice record.
        """
        cfg = self.cfg
        xml_path = _resolve_asset_path(cfg.scene_xml)
        spec = mujoco.MjSpec.from_file(str(xml_path))
        orig = spec.compile()  # original model, used only to read resolved defaults (class forcerange etc.)
        jnames, anames, limits = self._discover_arm(orig)
        n_arm = len(jnames)

        # --- replace arm position servos with pure torque motors (gear 1, no gain/bias dynamics)
        for name in anames:
            spec.delete(spec.actuator(name))
        for jn, lim in zip(jnames, limits):
            act = spec.add_actuator(name=f"{jn}_motor", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_motor()  # gainprm=[1,0..], biasprm=0, dyntype none -> force = ctrl
            act.gear = [1.0, 0, 0, 0, 0, 0]
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = [-float(lim), float(lim)]  # MuJoCo clamps ctrl -> torque saturation for free

        # --- joint dynamics overrides
        damping = _per_joint(cfg.joint_damping, n_arm, "joint_damping")
        armature = _per_joint(cfg.joint_armature, n_arm, "joint_armature")
        for k, jn in enumerate(jnames):
            j = spec.joint(jn)
            if damping is not None:
                j.damping = float(damping[k])
            if armature is not None:
                j.armature = float(armature[k])

        # --- keyframes store ctrl in the old actuator order; drop them (plant.reset sets the state)
        for key in list(spec.keys):
            spec.delete(key)

        # --- EE / flange sites and wrist F/T sensors
        ee = spec.body(cfg.ee_body)
        if ee is None:
            raise ValueError(f"ee_body '{cfg.ee_body}' not found in {xml_path.name}")
        ee.add_site(name=TCP_SITE, pos=list(cfg.tcp_offset), size=[0.004, 0, 0], rgba=[0, 1, 0, 0.6], group=4)
        ee.add_site(name=FT_SITE, pos=list(cfg.ft_site_offset), size=[0.004, 0, 0], rgba=[0, 0, 1, 0.6], group=4)
        # force/torque sensors report the interaction wrench between ee_body's subtree and its parent,
        # expressed in the site frame (MuJoCo convention).
        spec.add_sensor(name=FT_FORCE_SENSOR, type=mujoco.mjtSensor.mjSENS_FORCE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=FT_SITE)
        spec.add_sensor(name=FT_TORQUE_SENSOR, type=mujoco.mjtSensor.mjSENS_TORQUE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=FT_SITE)

        # --- task assets
        for asset in cfg.assets:
            if not isinstance(asset, SceneAsset):
                raise TypeError(f"Asset {asset!r} does not implement SceneAsset.build(spec, ee_body)")
            asset.build(spec, ee)

        # --- physics options
        spec.option.timestep = self.sim_dt
        if cfg.integrator is not None:
            spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{cfg.integrator.upper()}")
        if cfg.cone is not None:
            spec.option.cone = getattr(mujoco.mjtCone, f"mjCONE_{cfg.cone.upper()}")
        if cfg.impratio is not None:
            spec.option.impratio = float(cfg.impratio)

        model = spec.compile()
        self.spec = spec
        if cfg.dump_xml:
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            (GENERATED_DIR / f"{spec.modelname or 'scene'}_torque.xml").write_text(spec.to_xml())

        return model, self._resolve(model, jnames, limits)

    # ------------------------------------------------------------------ handles
    def _resolve(self, model: mujoco.MjModel, jnames: list[str], limits: np.ndarray) -> SceneHandles:
        """Resolve all names into ids/slices on the final model.

        Args:
            model: Final compiled model.
            jnames: Arm joint names in kinematic order.
            limits: Torque limits [Nm], shape (n_arm,).

        Returns:
            The `SceneHandles` record.
        """
        cfg = self.cfg
        n_arm = len(jnames)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jnames]
        qadr = [int(model.jnt_qposadr[j]) for j in jids]
        dadr = [int(model.jnt_dofadr[j]) for j in jids]
        aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in jnames]

        q_home = np.asarray(cfg.q_home, dtype=np.float64)
        if q_home.shape != (n_arm,):
            raise ValueError(f"q_home must have {n_arm} entries, got {q_home.shape}")

        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.ee_body)

        # Gripper: first non-arm actuator (Franka: tendon actuator on the finger split tendon).
        arm_set = set(aids)
        grip = [a for a in range(model.nu) if a not in arm_set]
        gid = grip[0] if grip else -1
        grange = model.actuator_ctrlrange[gid].copy() if gid >= 0 else np.zeros(2)

        # Finger joints: non-arm joints in the EE subtree (used to set the initial opening on reset).
        arm_jset = set(jids)
        fingers = [j for j in range(model.njnt)
                   if j not in arm_jset and _is_descendant(model, int(model.jnt_bodyid[j]), ee_id)
                   and int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE))]

        def sensor_slice(name: str) -> slice:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = int(model.sensor_adr[sid])
            return slice(adr, adr + int(model.sensor_dim[sid]))

        return SceneHandles(
            arm_joint_names=tuple(jnames),
            n_arm=n_arm,
            arm_qpos=_contiguous_slice(qadr, "arm qpos"),
            arm_dof=_contiguous_slice(dadr, "arm dof"),
            arm_ctrl=_contiguous_slice(aids, "arm actuator"),
            torque_limits=np.asarray(limits, dtype=np.float64),
            q_home=q_home,
            gripper_act_id=gid,
            gripper_ctrlrange=grange,
            gripper_open=float(np.clip(cfg.gripper_open, 0.0, 1.0)),
            finger_qpos_adr=np.array([model.jnt_qposadr[j] for j in fingers], dtype=np.int64),
            finger_qpos_range=np.array([model.jnt_range[j] for j in fingers], dtype=np.float64).reshape(-1, 2),
            ee_body_id=ee_id,
            tcp_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE),
            ft_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FT_SITE),
            ft_force=sensor_slice(FT_FORCE_SENSOR),
            ft_torque=sensor_slice(FT_TORQUE_SENSOR),
            body_ids=_name_map(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
            site_ids=_name_map(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            geom_ids=_name_map(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
            joint_ids=_name_map(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        )
