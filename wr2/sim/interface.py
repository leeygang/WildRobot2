"""Shared observation and command contract for MJX, MuJoCo, and hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from wr2.sensing.imu import normalize_quaternion_wxyz, rotate_vector_wxyz


FloatArray = npt.NDArray[np.float32]


def _vector(value: npt.ArrayLike, size: int, label: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


@dataclass(frozen=True)
class RobotObservation:
    """One backend-independent WR2 observation sample."""

    time_s: float
    joint_position_rad: FloatArray
    joint_velocity_rad_s: FloatArray
    torso_to_world_quat_wxyz: FloatArray
    angular_velocity_torso_rad_s: FloatArray
    foot_contact: npt.NDArray[np.bool_] | None = None

    def validate(self, actuator_count: int) -> None:
        if not np.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        _vector(self.joint_position_rad, actuator_count, "joint_position_rad")
        _vector(self.joint_velocity_rad_s, actuator_count, "joint_velocity_rad_s")
        normalize_quaternion_wxyz(self.torso_to_world_quat_wxyz)
        _vector(
            self.angular_velocity_torso_rad_s,
            3,
            "angular_velocity_torso_rad_s",
        )
        if self.foot_contact is not None:
            contact = np.asarray(self.foot_contact)
            if contact.shape != (2,):
                raise ValueError(f"foot_contact must have shape (2,), got {contact.shape}")

    @property
    def projected_gravity_torso(self) -> FloatArray:
        quat = normalize_quaternion_wxyz(self.torso_to_world_quat_wxyz)
        inverse = quat.copy()
        inverse[1:] *= -1.0
        return rotate_vector_wxyz(inverse, [0.0, 0.0, -1.0])


@dataclass(frozen=True)
class PositionActionContract:
    """Map normalized policy actions to bounded joint-position targets."""

    scale_rad: float = 0.25
    normalized_limit: float = 1.0
    joint_limit_margin_rad: float = 0.0

    def targets(
        self,
        normalized_action: npt.ArrayLike,
        home_position_rad: npt.ArrayLike,
        lower_limit_rad: npt.ArrayLike,
        upper_limit_rad: npt.ArrayLike,
    ) -> FloatArray:
        home = np.asarray(home_position_rad, dtype=np.float32)
        if home.ndim != 1:
            raise ValueError("home_position_rad must be one-dimensional")
        size = home.shape[0]
        action = _vector(normalized_action, size, "normalized_action")
        lower = _vector(lower_limit_rad, size, "lower_limit_rad")
        upper = _vector(upper_limit_rad, size, "upper_limit_rad")
        if np.any(lower + self.joint_limit_margin_rad >= upper - self.joint_limit_margin_rad):
            raise ValueError("Joint-limit margin leaves an empty command range")

        action = np.clip(action, -self.normalized_limit, self.normalized_limit)
        target = home + self.scale_rad * action
        return np.clip(
            target,
            lower + self.joint_limit_margin_rad,
            upper - self.joint_limit_margin_rad,
        ).astype(np.float32)


def build_wr2_proprio_v1(
    observation: RobotObservation,
    *,
    home_position_rad: npt.ArrayLike,
    previous_action: npt.ArrayLike,
    command_velocity: npt.ArrayLike,
) -> FloatArray:
    """Build the 60-value, hardware-reproducible WR2 actor observation."""
    home = np.asarray(home_position_rad, dtype=np.float32)
    if home.ndim != 1:
        raise ValueError("home_position_rad must be one-dimensional")
    actuator_count = home.shape[0]
    observation.validate(actuator_count)
    previous = _vector(previous_action, actuator_count, "previous_action")
    command = _vector(command_velocity, 3, "command_velocity")

    return np.concatenate(
        [
            command,
            observation.joint_position_rad - home,
            0.05 * observation.joint_velocity_rad_s,
            previous,
            observation.angular_velocity_torso_rad_s,
            observation.projected_gravity_torso,
        ],
        dtype=np.float32,
    )


class RobotBackend(Protocol):
    """Interface that both a simulator and physical WR2 backend must satisfy."""

    actuator_names: tuple[str, ...]
    control_period_s: float

    def read(self) -> RobotObservation:
        ...

    def write_joint_targets(self, target_position_rad: npt.ArrayLike) -> None:
        ...

    def close(self) -> None:
        ...
