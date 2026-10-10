"""Reusable task assets implementing the `SceneAsset` protocol.

Each asset is a config dataclass (so it serializes into config.json) with a `build`
method that adds elements to the scene `MjSpec`. Element names are prefixed with the
asset `name`, so several assets of the same kind can coexist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco


@dataclass
class TargetMarker:
    """Visual-only mocap sphere marking a target position (no collisions, no mass effects).

    Creates body `<name>` (mocap) with geom `<name>_geom` and site `<name>_site`.
    Events move it via `data.mocap_pos[model.body_mocapid[body_id]]`.

    Attributes:
        name: Body name.
        radius: Sphere radius [m].
        rgba: Color.
    """

    name: str = "target"
    radius: float = 0.015
    rgba: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)

    def build(self, spec: mujoco.MjSpec, ee_body: mujoco.MjsBody) -> None:
        """Add the mocap marker to the world.

        Args:
            spec: Scene spec.
            ee_body: End-effector body spec (unused).
        """
        body = spec.worldbody.add_body(name=self.name, mocap=True, pos=[0.5, 0.0, 0.3])
        body.add_geom(name=f"{self.name}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self.radius, 0, 0],
                      rgba=list(self.rgba), contype=0, conaffinity=0, group=2)
        body.add_site(name=f"{self.name}_site", size=[0.002, 0, 0])


# Collision bitmask shared by insertion parts: they collide only with each other
# (robot/floor geoms use contype = conaffinity = 1, so (1 & 2) == 0 in both directions).
PART_COLLISION_BIT: int = 2


@dataclass
class Peg:
    """Cylindrical peg rigidly attached to the end-effector body (no grasp slip).

    The peg axis is the `ee_body` z-axis (pointing out of the gripper). Creates:
      body `<name>`, geom `<name>_geom`,
      site `<name>_tip` (center of the bottom face),
      sites `<name>_kp0..kp{n-1}` evenly spaced from the tip (kp0) to the top face.

    Attributes:
        name: Body name.
        radius: Peg radius [m] (8 mm peg -> 0.004).
        length: Peg length [m].
        top_z: z of the peg top face in the `ee_body` frame [m] (Franka hand: pads span 0.098-0.112).
        n_keypoints: Number of keypoints along the axis (>= 2).
        friction: Sliding friction coefficient.
        solref: Contact solref (timeconst, dampratio); timeconst should be >= 2 * sim_dt.
        solimp: Contact solimp (dmin, dmax, width, midpoint, power).
        density: Material density [kg/m^3].
        rgba: Color.
    """

    name: str = "peg"
    radius: float = 0.004
    length: float = 0.05
    top_z: float = 0.0884
    n_keypoints: int = 4
    friction: float = 0.75
    solref: tuple[float, float] = (0.004, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.0005, 0.5, 2.0)
    density: float = 2700.0
    rgba: tuple[float, float, float, float] = (0.9, 0.6, 0.1, 1.0)

    def build(self, spec: mujoco.MjSpec, ee_body: mujoco.MjsBody) -> None:
        """Attach the peg body to the end-effector body.

        Args:
            spec: Scene spec (unused).
            ee_body: End-effector body spec (Franka `hand`).
        """
        half = 0.5 * self.length
        body = ee_body.add_body(name=self.name, pos=[0.0, 0.0, self.top_z + half])
        gtype, gsize = self._shape(half)
        body.add_geom(name=f"{self.name}_geom", type=gtype, size=gsize,
                      rgba=list(self.rgba), density=self.density, friction=[self.friction, 0.005, 0.0001],
                      solref=list(self.solref), solimp=list(self.solimp),
                      contype=PART_COLLISION_BIT, conaffinity=PART_COLLISION_BIT, condim=3)
        body.add_site(name=f"{self.name}_tip", pos=[0, 0, half], size=[0.0015, 0, 0], rgba=[1, 0, 0, 1], group=4)
        n = max(2, self.n_keypoints)
        for i in range(n):
            # kp0 at the tip (+half in body z, i.e. away from the hand), kp{n-1} at the top face.
            z = half - self.length * i / (n - 1)
            body.add_site(name=f"{self.name}_kp{i}", pos=[0, 0, z], size=[0.001, 0, 0], rgba=[1, 1, 0, 1], group=4)

    def _shape(self, half: float) -> tuple[int, list[float]]:
        """Geom type and size of the peg (cylinder of `radius`, half-length `half`)."""
        return mujoco.mjtGeom.mjGEOM_CYLINDER, [self.radius, half, 0.0]


@dataclass
class RoundHole:
    """Solid square socket block with a round hole, on a mocap body (moved by events at reset).

    MuJoCo cannot collide against concave meshes, so the hole wall is a ring of
    `n_walls` boxes whose inner faces are tangent to a circle of radius `inner_radius`
    (the polygon's inscribed radius equals the hole radius). Four slabs fill the rest
    of a solid block of half-size `block_half_size`, whose top is flush with the hole
    opening. They leave a square opening of half-width (inner_radius + wall_thickness) / sqrt(2),
    whose corners lie inside the ring wall, so there is no gap.

    A solid block matters for learning. With a thin tube on a plate, the peg can be lowered
    beside the socket, where its keypoints are closer to the inserted pose than when hovering
    above the opening, and PPO learns to park there. With a wide block, any pose below the
    opening's height lies at least `block_half_size` + peg radius off-axis.

    Creates: mocap body `<name>`, geoms `<name>_base`, `<name>_wall{i}`, `<name>_block{0..3}`,
    site `<name>_tip` (center of the hole opening = top of the block),
    site `<name>_floor` (center of the hole bottom). The body origin is the bottom of the base plate.

    Attributes:
        name: Body name.
        inner_radius: Hole radius [m] (8 mm peg + 0.5 mm diametrical clearance -> 0.00425).
        depth: Hole depth [m].
        n_walls: Number of wall boxes.
        wall_thickness: Radial wall thickness [m].
        base_thickness: Base plate thickness under the hole [m].
        block_half_size: Half extent of the solid block (and base plate) in x and y [m].
        friction: Sliding friction coefficient.
        solref: Contact solref.
        solimp: Contact solimp (dmin, dmax, width, midpoint, power).
        rgba: Color.
    """

    name: str = "hole"
    inner_radius: float = 0.00425
    depth: float = 0.025
    n_walls: int = 24
    wall_thickness: float = 0.006
    base_thickness: float = 0.005
    block_half_size: float = 0.03
    friction: float = 0.75
    solref: tuple[float, float] = (0.004, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.0005, 0.5, 2.0)
    rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.65, 1.0)

    def build(self, spec: mujoco.MjSpec, ee_body: mujoco.MjsBody) -> None:
        """Add the socket to the world as a mocap body.

        Args:
            spec: Scene spec.
            ee_body: End-effector body spec (unused).
        """
        common = dict(friction=[self.friction, 0.005, 0.0001], solref=list(self.solref), solimp=list(self.solimp),
                      contype=PART_COLLISION_BIT, conaffinity=PART_COLLISION_BIT, condim=3, rgba=list(self.rgba))
        body = spec.worldbody.add_body(name=self.name, mocap=True, pos=[0.6, 0.0, 0.05])
        bt = self.base_thickness
        H = self.block_half_size
        body.add_geom(name=f"{self.name}_base", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[H, H, 0.5 * bt], pos=[0, 0, 0.5 * bt], **common)
        n = self.n_walls
        r_mid = self.inner_radius + 0.5 * self.wall_thickness
        # Tangential half-width so neighbouring boxes overlap at the outer radius (no gaps).
        half_w = (self.inner_radius + self.wall_thickness) * math.tan(math.pi / n)
        z_mid = bt + 0.5 * self.depth
        for i in range(n):
            th = 2.0 * math.pi * i / n
            # Box local x = radial direction (rotation th about z), so its inner face sits at inner_radius.
            body.add_geom(name=f"{self.name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.5 * self.wall_thickness, half_w, 0.5 * self.depth],
                          pos=[r_mid * math.cos(th), r_mid * math.sin(th), z_mid],
                          quat=[math.cos(0.5 * th), 0.0, 0.0, math.sin(0.5 * th)], **common)
        # Block slabs around the ring: +-x slabs span the full y range; +-y slabs fill between them.
        a = (self.inner_radius + self.wall_thickness) / math.sqrt(2.0)  # square opening inside the ring
        if H <= a:
            raise ValueError("block_half_size must exceed (inner_radius + wall_thickness) / sqrt(2)")
        hx = 0.5 * (H - a)
        for k, (sx, sy, size) in enumerate([(1, 0, [hx, H]), (-1, 0, [hx, H]), (0, 1, [a, hx]), (0, -1, [a, hx])]):
            off = a + hx
            body.add_geom(name=f"{self.name}_block{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[size[0], size[1], 0.5 * self.depth], pos=[sx * off, sy * off, z_mid], **common)
        body.add_site(name=f"{self.name}_tip", pos=[0, 0, bt + self.depth], size=[0.0015, 0, 0],
                      rgba=[0, 1, 0, 1], group=4)
        body.add_site(name=f"{self.name}_floor", pos=[0, 0, bt], size=[0.0015, 0, 0], rgba=[0, 0, 1, 1], group=4)


@dataclass
class BoxPeg(Peg):
    """Cuboid peg (rectangular cross-section) rigidly attached to the end-effector body.

    Same frame, sites and keypoints as `Peg` (axis = `ee_body` z, `<name>_tip` at the bottom-face centre);
    the cross-section is `2 half_x` x `2 half_y` along the `ee_body` x and y axes. `radius` is unused.

    Attributes:
        half_x: Cross-section half-size along the `ee_body` x axis [m] (8 mm -> 0.004).
        half_y: Cross-section half-size along the `ee_body` y axis [m].
    """

    half_x: float = 0.004
    half_y: float = 0.004

    def _shape(self, half: float) -> tuple[int, list[float]]:
        """Geom type and size of the peg (box of half-sizes `half_x`, `half_y`, `half`)."""
        return mujoco.mjtGeom.mjGEOM_BOX, [self.half_x, self.half_y, half]


@dataclass
class RectHole:
    """Solid square socket block with a rectangular hole, on a mocap body (moved by events at reset).

    The hole (`2 inner_half_x` x `2 inner_half_y`, along the body x and y axes) is the gap between four slabs:
    the +-x slabs span the full block in y, the +-y slabs fill between them. Same sites and body layout as
    `RoundHole`: `<name>_tip` (centre of the opening = top of the block), `<name>_floor` (centre of the hole
    bottom); the body origin is the bottom of the base plate. Rotate the hole with the mocap orientation.

    Attributes:
        name: Body name.
        inner_half_x: Hole half-size along the body x axis [m] (8 mm peg + 0.5 mm clearance -> 0.00425).
        inner_half_y: Hole half-size along the body y axis [m].
        depth: Hole depth [m].
        base_thickness: Base plate thickness under the hole [m].
        block_half_size: Half extent of the block (and base plate) in x and y [m].
        friction: Sliding friction coefficient.
        solref: Contact solref.
        solimp: Contact solimp (dmin, dmax, width, midpoint, power).
        rgba: Color.
    """

    name: str = "hole"
    inner_half_x: float = 0.00425
    inner_half_y: float = 0.00425
    depth: float = 0.025
    base_thickness: float = 0.005
    block_half_size: float = 0.03
    friction: float = 0.75
    solref: tuple[float, float] = (0.004, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.0005, 0.5, 2.0)
    rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.65, 1.0)

    def build(self, spec: mujoco.MjSpec, ee_body: mujoco.MjsBody) -> None:
        """Add the socket to the world as a mocap body.

        Args:
            spec: Scene spec.
            ee_body: End-effector body spec (unused).
        """
        common = dict(friction=[self.friction, 0.005, 0.0001], solref=list(self.solref), solimp=list(self.solimp),
                      contype=PART_COLLISION_BIT, conaffinity=PART_COLLISION_BIT, condim=3, rgba=list(self.rgba))
        body = spec.worldbody.add_body(name=self.name, mocap=True, pos=[0.6, 0.0, 0.05])
        bt, H, ix, iy = self.base_thickness, self.block_half_size, self.inner_half_x, self.inner_half_y
        if H <= max(ix, iy):
            raise ValueError("block_half_size must exceed the hole half-sizes")
        body.add_geom(name=f"{self.name}_base", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[H, H, 0.5 * bt], pos=[0, 0, 0.5 * bt], **common)
        z_mid = bt + 0.5 * self.depth
        hx, hy = 0.5 * (H - ix), 0.5 * (H - iy)
        for k, (pos, size) in enumerate([((ix + hx, 0.0), (hx, H)), ((-(ix + hx), 0.0), (hx, H)),
                                         ((0.0, iy + hy), (ix, hy)), ((0.0, -(iy + hy)), (ix, hy))]):
            body.add_geom(name=f"{self.name}_block{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[size[0], size[1], 0.5 * self.depth], pos=[pos[0], pos[1], z_mid], **common)
        body.add_site(name=f"{self.name}_tip", pos=[0, 0, bt + self.depth], size=[0.0015, 0, 0],
                      rgba=[0, 1, 0, 1], group=4)
        body.add_site(name=f"{self.name}_floor", pos=[0, 0, bt], size=[0.0015, 0, 0], rgba=[0, 0, 1, 1], group=4)
