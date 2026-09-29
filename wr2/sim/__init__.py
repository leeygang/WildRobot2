"""Common interfaces shared by simulation and physical hardware."""

from wr2.sim.interface import (
    PositionActionContract,
    RobotBackend,
    RobotObservation,
    build_wr2_proprio_v1,
)
from wr2.sim.robot import MotorSpec, RobotDescription

__all__ = [
    "MotorSpec",
    "PositionActionContract",
    "RobotBackend",
    "RobotDescription",
    "RobotObservation",
    "build_wr2_proprio_v1",
]
