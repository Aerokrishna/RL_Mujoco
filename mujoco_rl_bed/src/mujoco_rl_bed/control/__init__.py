"""Torque controllers: Cartesian and joint impedance, plus shared math helpers."""

from mujoco_rl_bed.control.base import Controller, ControllerCfg
from mujoco_rl_bed.control.cartesian_impedance import CartesianImpedance, CartesianImpedanceCfg
from mujoco_rl_bed.control.joint_impedance import JointImpedance, JointImpedanceCfg
from mujoco_rl_bed.control.math_utils import critical_damping, damped_pinv

__all__ = ["Controller", "ControllerCfg", "CartesianImpedance", "CartesianImpedanceCfg",
           "JointImpedance", "JointImpedanceCfg", "critical_damping", "damped_pinv"]
