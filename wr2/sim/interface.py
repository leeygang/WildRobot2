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
                raise ValueError(
                    f"foot_contact must have shape (2,), got {contact.shape}"
                )

    @property
    def projected_gravity_torso(self) -> FloatArray:
        quat = normalize_quaternion_wxyz(self.torso_to_world_quat_wxyz)
        inverse = quat.copy()
        inverse[1:] *= -1.0
        return rotate_vector_wxyz(inverse, [0.0, 0.0, -1.0])


@dataclass(frozen=True)
class PositionActionContract:
    """Map normalized policy actions over each joint's safe position range.

    ``-1`` and ``+1`` are the lower and upper command limits, respectively,
    and ``0`` is their midpoint.  The walking keyframe is therefore generally
    a nonzero action; callers can obtain it with :meth:`home_action`.
    """

    normalized_limit: float = 1.0
    joint_limit_margin_rad: float = 0.0

    def _safe_limits(
        self,
        lower_limit_rad: npt.ArrayLike,
        upper_limit_rad: npt.ArrayLike,
        size: int,
    ) -> tuple[FloatArray, FloatArray]:
        lower = _vector(lower_limit_rad, size, "lower_limit_rad")
        upper = _vector(upper_limit_rad, size, "upper_limit_rad")
        safe_lower = lower + self.joint_limit_margin_rad
        safe_upper = upper - self.joint_limit_margin_rad
        if np.any(safe_lower >= safe_upper):
            raise ValueError("Joint-limit margin leaves an empty command range")
        return safe_lower, safe_upper

    def home_action(
        self,
        home_position_rad: npt.ArrayLike,
        lower_limit_rad: npt.ArrayLike,
        upper_limit_rad: npt.ArrayLike,
    ) -> FloatArray:
        """Return the normalized action corresponding to a reference pose."""
        home = np.asarray(home_position_rad, dtype=np.float32)
        if home.ndim != 1:
            raise ValueError("home_position_rad must be one-dimensional")
        safe_lower, safe_upper = self._safe_limits(
            lower_limit_rad, upper_limit_rad, home.size
        )
        if np.any(home < safe_lower) or np.any(home > safe_upper):
            raise ValueError("Home position lies outside the safe command range")
        midpoint = 0.5 * (safe_lower + safe_upper)
        half_range = 0.5 * (safe_upper - safe_lower)
        return ((home - midpoint) / half_range).astype(np.float32)

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
        safe_lower, safe_upper = self._safe_limits(
            lower_limit_rad, upper_limit_rad, size
        )

        action = np.clip(action, -self.normalized_limit, self.normalized_limit)
        midpoint = 0.5 * (safe_lower + safe_upper)
        half_range = 0.5 * (safe_upper - safe_lower)
        return np.clip(midpoint + half_range * action, safe_lower, safe_upper).astype(
            np.float32
        )

    def active_targets(
        self,
        normalized_action: npt.ArrayLike,
        *,
        active_indices: npt.ArrayLike,
        home_position_rad: npt.ArrayLike,
        lower_limit_rad: npt.ArrayLike,
        upper_limit_rad: npt.ArrayLike,
    ) -> FloatArray:
        """Expand active-joint absolute actions; inactive joints stay at home."""
        home = np.asarray(home_position_rad, dtype=np.float32)
        if home.ndim != 1:
            raise ValueError("home_position_rad must be one-dimensional")
        size = home.shape[0]
        safe_lower, safe_upper = self._safe_limits(
            lower_limit_rad, upper_limit_rad, size
        )
        indices = np.asarray(active_indices, dtype=np.int32)
        if indices.ndim != 1 or len(np.unique(indices)) != indices.size:
            raise ValueError("active_indices must be a unique one-dimensional array")
        if np.any(indices < 0) or np.any(indices >= size):
            raise ValueError("active_indices contains an out-of-range index")
        action = _vector(normalized_action, indices.size, "normalized_action")
        action = np.clip(action, -self.normalized_limit, self.normalized_limit)

        active_lower = safe_lower[indices]
        active_upper = safe_upper[indices]
        midpoint = 0.5 * (active_lower + active_upper)
        half_range = 0.5 * (active_upper - active_lower)
        target = home.copy()
        target[indices] = np.clip(
            midpoint + half_range * action, active_lower, active_upper
        )
        return target.astype(np.float32)

    def active_residual_targets(
        self,
        policy_action: npt.ArrayLike,
        *,
        action_scale_rad: float,
        active_indices: npt.ArrayLike,
        home_position_rad: npt.ArrayLike,
        lower_limit_rad: npt.ArrayLike,
        upper_limit_rad: npt.ArrayLike,
    ) -> FloatArray:
        """Map an unbounded walking-policy residual to safe joint targets.

        This mirrors ToddlerBot's locomotion action path. The policy action is
        not clipped; only the resulting physical target is clipped to the safe
        joint limits. Inactive actuators remain at ``home_position_rad``.
        """
        if not np.isfinite(action_scale_rad) or action_scale_rad <= 0.0:
            raise ValueError("action_scale_rad must be finite and positive")
        home = np.asarray(home_position_rad, dtype=np.float32)
        if home.ndim != 1:
            raise ValueError("home_position_rad must be one-dimensional")
        size = home.shape[0]
        safe_lower, safe_upper = self._safe_limits(
            lower_limit_rad, upper_limit_rad, size
        )
        indices = np.asarray(active_indices, dtype=np.int32)
        if indices.ndim != 1 or len(np.unique(indices)) != indices.size:
            raise ValueError("active_indices must be a unique one-dimensional array")
        if np.any(indices < 0) or np.any(indices >= size):
            raise ValueError("active_indices contains an out-of-range index")
        action = _vector(policy_action, indices.size, "policy_action")

        target = home.copy()
        target[indices] = np.clip(
            home[indices] + action_scale_rad * action,
            safe_lower[indices],
            safe_upper[indices],
        )
        return target.astype(np.float32)


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


def build_wr2_proprio_v2(
    observation: RobotObservation,
    *,
    gait_phase_rad: float,
    home_position_rad: npt.ArrayLike,
    previous_action: npt.ArrayLike,
    command_velocity: npt.ArrayLike,
) -> FloatArray:
    """Build one 62-value walking frame, including a local gait clock."""
    if not np.isfinite(gait_phase_rad):
        raise ValueError("gait_phase_rad must be finite")
    phase = np.asarray(
        [np.sin(gait_phase_rad), np.cos(gait_phase_rad)], dtype=np.float32
    )
    proprio = build_wr2_proprio_v1(
        observation,
        home_position_rad=home_position_rad,
        previous_action=previous_action,
        command_velocity=command_velocity,
    )
    return np.concatenate([phase, proprio], dtype=np.float32)


def build_wr2_proprio_v3(
    observation: RobotObservation,
    *,
    gait_phase_rad: float,
    home_position_rad: npt.ArrayLike,
    previous_active_action: npt.ArrayLike,
    active_indices: npt.ArrayLike,
    command_velocity: npt.ArrayLike,
) -> FloatArray:
    """Build the 55-value walking frame used by the active-leg policy.

    Joint position and velocity retain all 17 actuators, as in ToddlerBot,
    while the previous-action field contains only the ten policy-controlled
    leg joints.
    """
    if not np.isfinite(gait_phase_rad):
        raise ValueError("gait_phase_rad must be finite")
    home = np.asarray(home_position_rad, dtype=np.float32)
    if home.ndim != 1:
        raise ValueError("home_position_rad must be one-dimensional")
    observation.validate(home.size)
    indices = np.asarray(active_indices, dtype=np.int32)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= home.size):
        raise ValueError("active_indices must contain valid actuator indices")
    previous = _vector(previous_active_action, indices.size, "previous_active_action")
    command = _vector(command_velocity, 3, "command_velocity")
    return np.concatenate(
        [
            np.asarray(
                [np.sin(gait_phase_rad), np.cos(gait_phase_rad)], dtype=np.float32
            ),
            command,
            observation.joint_position_rad - home,
            0.05 * observation.joint_velocity_rad_s,
            previous,
            observation.angular_velocity_torso_rad_s,
            observation.projected_gravity_torso,
        ],
        dtype=np.float32,
    )


def build_wr2_proprio_v4(
    observation: RobotObservation,
    *,
    reference_heading_rad: float,
    **kwargs,
) -> FloatArray:
    """Append IMU-relative heading error to the v3 walking frame.

    Hardware uses a six-axis/game rotation vector in a frame whose initial
    heading is zero, and integrates commanded yaw in the same frame.
    """
    if not np.isfinite(reference_heading_rad):
        raise ValueError("reference_heading_rad must be finite")
    frame = build_wr2_proprio_v3(observation, **kwargs)
    heading = rotate_vector_wxyz(observation.torso_to_world_quat_wxyz, [1, 0, 0])
    error = np.arctan2(heading[1], heading[0]) - reference_heading_rad
    return np.concatenate([frame, [np.sin(error), np.cos(error)]]).astype(np.float32)


def update_wr2_observation_history(
    observation_frame: npt.ArrayLike,
    *,
    history_frames: int,
    previous_history: npt.ArrayLike | None = None,
) -> FloatArray:
    """Insert one frame at the front of a deployment-compatible history."""
    frame = np.asarray(observation_frame, dtype=np.float32)
    if frame.ndim != 1:
        raise ValueError("observation_frame must be one-dimensional")
    if history_frames <= 0:
        raise ValueError("history_frames must be positive")
    history_size = history_frames * frame.size
    if previous_history is None:
        history = np.zeros(history_size, dtype=np.float32)
    else:
        history = _vector(previous_history, history_size, "previous_history").copy()
    history = np.roll(history, frame.size)
    history[: frame.size] = frame
    return history


class RobotBackend(Protocol):
    """Interface that both a simulator and physical WR2 backend must satisfy."""

    actuator_names: tuple[str, ...]
    control_period_s: float

    def read(self) -> RobotObservation: ...

    def write_joint_targets(self, target_position_rad: npt.ArrayLike) -> None: ...

    def close(self) -> None: ...
