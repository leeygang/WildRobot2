"""IMU frame conversion shared by simulation and physical hardware."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]


def normalize_quaternion_wxyz(quaternion: npt.ArrayLike) -> FloatArray:
    """Normalize a wxyz quaternion and select the non-negative-w hemisphere."""
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion must have a finite, non-zero norm")
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def quaternion_conjugate_wxyz(quaternion: npt.ArrayLike) -> FloatArray:
    quat = normalize_quaternion_wxyz(quaternion).copy()
    quat[1:] *= -1.0
    return quat


def quaternion_multiply_wxyz(left: npt.ArrayLike, right: npt.ArrayLike) -> FloatArray:
    """Compose wxyz quaternions; result applies right, then left."""
    w1, x1, y1, z1 = normalize_quaternion_wxyz(left).astype(np.float64)
    w2, x2, y2, z2 = normalize_quaternion_wxyz(right).astype(np.float64)
    return normalize_quaternion_wxyz(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
    )


def rotate_vector_wxyz(quaternion: npt.ArrayLike, vector: npt.ArrayLike) -> FloatArray:
    """Rotate a three-vector by a wxyz quaternion."""
    quat = normalize_quaternion_wxyz(quaternion).astype(np.float64)
    vec = np.asarray(vector, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"Expected vector shape (3,), got {vec.shape}")
    w = quat[0]
    xyz = quat[1:]
    rotated = (
        (2.0 * w * w - 1.0) * vec
        + 2.0 * np.dot(xyz, vec) * xyz
        + 2.0 * w * np.cross(xyz, vec)
    )
    return rotated.astype(np.float32)


@dataclass(frozen=True)
class CanonicalImuSample:
    """IMU signals expressed in WR2's torso frame."""

    torso_to_world_quat_wxyz: FloatArray
    angular_velocity_torso_rad_s: FloatArray

    @property
    def projected_gravity_torso(self) -> FloatArray:
        world_to_torso = quaternion_conjugate_wxyz(self.torso_to_world_quat_wxyz)
        return rotate_vector_wxyz(world_to_torso, [0.0, 0.0, -1.0])


def canonicalize_sensor_sample(
    *,
    sensor_to_world_quat_wxyz: npt.ArrayLike,
    angular_velocity_sensor_rad_s: npt.ArrayLike,
    sensor_to_torso_quat_wxyz: npt.ArrayLike,
) -> CanonicalImuSample:
    """Convert a mounted sensor sample into the canonical torso frame.

    Quaternion convention: ``q_AB`` rotates vectors from frame B into frame A.
    The hardware adapter must first convert the BNO085 output to q_WS
    (sensor-to-world). Given the fixed mount q_TS (sensor-to-torso), this
    function computes q_WT = q_WS * inverse(q_TS).
    """
    sensor_to_torso = normalize_quaternion_wxyz(sensor_to_torso_quat_wxyz)
    torso_to_world = quaternion_multiply_wxyz(
        sensor_to_world_quat_wxyz,
        quaternion_conjugate_wxyz(sensor_to_torso),
    )
    angular_velocity_torso = rotate_vector_wxyz(
        sensor_to_torso, angular_velocity_sensor_rad_s
    )
    return CanonicalImuSample(torso_to_world, angular_velocity_torso)
