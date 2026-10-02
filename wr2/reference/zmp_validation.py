"""Reference-only geometry validation for WR2's ZMP walking lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from wr2.locomotion.configs import (
    DEFAULT_TRAINING_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
)
from wr2.reference.walk_zmp import WR2ZMPReference, ZMPReferenceSample
from wr2.sim.robot import RobotDescription


@dataclass(frozen=True)
class ZMPReferenceContext:
    """MuJoCo model and lookup data shared by validators and the viewer."""

    model: mujoco.MjModel
    data: mujoco.MjData
    reference: WR2ZMPReference
    home_qpos: np.ndarray
    joint_qpos_indices: np.ndarray
    foot_geom_ids: np.ndarray
    training_config: TrainingConfig


@dataclass(frozen=True)
class ZMPGeometryResult:
    """Geometry and continuity measurements for one forward command."""

    command_forward_m_s: float
    samples: int
    failures: tuple[str, ...]
    worst_stance_bottom_z_m: float
    worst_swing_bottom_z_m: float
    minimum_zmp_support_margin_m: float
    maximum_lipm_residual_m: float
    maximum_joint_step_rad: float

    @property
    def passed(self) -> bool:
        return not self.failures


def create_reference_context(
    commands_m_s: tuple[float, ...],
    config_path: str | Path = DEFAULT_TRAINING_CONFIG_PATH,
) -> ZMPReferenceContext:
    """Load the canonical model and build an exact command lookup."""
    if not commands_m_s or any(command <= 0.0 for command in commands_m_s):
        raise ValueError("commands_m_s must contain positive forward speeds")
    commands = tuple(sorted(set(float(command) for command in commands_m_s)))
    if len(commands) > 2:
        spacing = np.diff(commands)
        if not np.allclose(spacing, spacing[0], atol=1e-9):
            raise ValueError("three or more validation commands must be evenly spaced")

    training_config = load_training_config(config_path)
    robot = RobotDescription.load(include_local_calibration=False)
    model = mujoco.MjModel.from_xml_path(str(robot.directory / "scene_mjx.xml"))
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "walk_home")
    if keyframe_id < 0:
        raise ValueError("WR2 model is missing walk_home")
    home_qpos = np.asarray(model.key_qpos[keyframe_id], dtype=np.float64).copy()

    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in robot.actuator_names
        ],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        raise ValueError("WR2 validation could not resolve every actuator joint")
    joint_qpos_indices = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
    foot_geom_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
            for side in ("left", "right")
        ],
        dtype=np.int32,
    )
    if np.any(foot_geom_ids < 0):
        raise ValueError("WR2 validation requires both foot collision boxes")

    zmp = training_config.environment.zmp_reference
    reference = WR2ZMPReference(
        model,
        keyframe_id=keyframe_id,
        actuator_names=robot.actuator_names,
        command_forward_range_m_s=(commands[0], commands[-1]),
        gait_cycle_s=training_config.environment.gait_cycle_s,
        swing_height_m=training_config.environment.swing_height_m,
        com_height_m=zmp.com_height_m,
        single_double_support_ratio=zmp.single_double_support_ratio,
        phase_samples=zmp.phase_samples,
        command_samples=len(commands),
        ik_damping=zmp.ik_damping,
        ik_max_iterations=zmp.ik_max_iterations,
        max_position_residual_m=zmp.max_position_residual_m,
        max_orientation_residual_rad=zmp.max_orientation_residual_rad,
    )
    return ZMPReferenceContext(
        model=model,
        data=data,
        reference=reference,
        home_qpos=home_qpos,
        joint_qpos_indices=joint_qpos_indices,
        foot_geom_ids=foot_geom_ids,
        training_config=training_config,
    )


def box_projected_corners_xy(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray:
    """Project all eight world-frame box corners onto the ground plane."""
    size = np.asarray(model.geom_size[geom_id])
    local = np.asarray(
        [
            [x * size[0], y * size[1], z * size[2]]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world = local @ rotation.T + data.geom_xpos[geom_id]
    return world[:, :2]


def box_minimum_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Return the lowest world-frame point of a box collision geometry."""
    position = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    return float(position[2] - np.sum(np.abs(rotation[2]) * size))


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise two-dimensional convex hull."""
    unique = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
    if len(unique) < 3:
        raise ValueError("support polygon requires at least three unique points")

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def point_support_margin(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
    """Return minimum signed edge distance; non-negative means inside."""
    point = np.asarray(point_xy, dtype=np.float64)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    margins = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge = end - start
        edge_length = np.linalg.norm(edge)
        if edge_length <= 1e-12:
            continue
        delta = point - start
        cross = edge[0] * delta[1] - edge[1] * delta[0]
        margins.append(cross / edge_length)
    if not margins:
        raise ValueError("support polygon contains no valid edge")
    return float(min(margins))


def apply_reference_frame(
    context: ZMPReferenceContext,
    *,
    phase_rad: float,
    command_forward_m_s: float,
    cycle_index: int = 0,
) -> tuple[ZMPReferenceSample, float]:
    """Apply one kinematic reference frame and return its sample and root x."""
    sample = context.reference.sample(phase_rad, command_forward_m_s)
    cycle_time = context.training_config.environment.gait_cycle_s
    phase_time = sample.phase_rad / (2.0 * np.pi) * cycle_time
    elapsed = cycle_index * cycle_time + phase_time
    root_x = sample.command_forward_m_s * elapsed
    qpos = context.home_qpos.copy()
    qpos[0] = root_x
    qpos[1] = sample.com_lateral_m
    qpos[2] = sample.root_height_m
    qpos[context.joint_qpos_indices] = sample.joint_position_rad
    context.data.qpos[:] = qpos
    context.data.qvel[:] = 0.0
    mujoco.mj_forward(context.model, context.data)
    return sample, root_x


def validate_zmp_reference(
    context: ZMPReferenceContext,
    command_forward_m_s: float,
    *,
    samples: int = 72,
    stance_height_tolerance_m: float = 0.003,
    swing_penetration_tolerance_m: float = -0.002,
    support_tolerance_m: float = 0.0005,
) -> ZMPGeometryResult:
    """Validate one complete reference cycle without running physics."""
    if samples < 8:
        raise ValueError("samples must be at least eight")
    failures: list[str] = []
    worst_stance = -np.inf
    worst_swing = np.inf
    minimum_support_margin = np.inf
    maximum_lipm_residual = 0.0
    joint_frames = []
    cycle_time = context.training_config.environment.gait_cycle_s
    com_height = context.training_config.environment.zmp_reference.com_height_m
    angular_frequency = 2.0 * np.pi / cycle_time

    for frame_index in range(samples):
        phase = frame_index * 2.0 * np.pi / samples
        sample, root_x = apply_reference_frame(
            context,
            phase_rad=phase,
            command_forward_m_s=command_forward_m_s,
        )
        joint_frames.append(sample.joint_position_rad)
        bottom_z = np.asarray(
            [
                box_minimum_z(context.model, context.data, int(geom_id))
                for geom_id in context.foot_geom_ids
            ]
        )
        for side_index, side in enumerate(("left", "right")):
            if sample.stance_mask[side_index]:
                worst_stance = max(worst_stance, float(bottom_z[side_index]))
                if bottom_z[side_index] > stance_height_tolerance_m:
                    failures.append(
                        f"frame {frame_index} {side} stance bottom "
                        f"{bottom_z[side_index]:+.4f} m exceeds "
                        f"{stance_height_tolerance_m:+.4f} m"
                    )
            else:
                worst_swing = min(worst_swing, float(bottom_z[side_index]))
                if bottom_z[side_index] < swing_penetration_tolerance_m:
                    failures.append(
                        f"frame {frame_index} {side} swing bottom "
                        f"{bottom_z[side_index]:+.4f} m is below "
                        f"{swing_penetration_tolerance_m:+.4f} m"
                    )

        support_points = np.concatenate(
            [
                box_projected_corners_xy(context.model, context.data, int(geom_id))
                for geom_id, stance in zip(
                    context.foot_geom_ids, sample.stance_mask, strict=True
                )
                if stance
            ]
        )
        support_polygon = convex_hull_xy(support_points)
        zmp_world = np.asarray([root_x, sample.zmp_lateral_m])
        support_margin = point_support_margin(zmp_world, support_polygon)
        minimum_support_margin = min(minimum_support_margin, support_margin)
        if support_margin < -support_tolerance_m:
            failures.append(
                f"frame {frame_index} ZMP is {-support_margin * 1000.0:.2f} mm "
                "outside the active support polygon"
            )

        com_acceleration_y = -(angular_frequency**2) * sample.com_lateral_m
        implied_zmp_y = sample.com_lateral_m - com_height / 9.81 * com_acceleration_y
        maximum_lipm_residual = max(
            maximum_lipm_residual, abs(implied_zmp_y - sample.zmp_lateral_m)
        )

    joint_frames_array = np.asarray(joint_frames)
    wrapped = np.concatenate([joint_frames_array, joint_frames_array[:1]], axis=0)
    maximum_joint_step = float(np.max(np.abs(np.diff(wrapped, axis=0))))
    if maximum_lipm_residual > 1e-6:
        failures.append(f"LIPM residual {maximum_lipm_residual:.3e} m exceeds 1e-6 m")

    return ZMPGeometryResult(
        command_forward_m_s=float(command_forward_m_s),
        samples=samples,
        failures=tuple(failures),
        worst_stance_bottom_z_m=float(worst_stance),
        worst_swing_bottom_z_m=float(worst_swing),
        minimum_zmp_support_margin_m=float(minimum_support_margin),
        maximum_lipm_residual_m=float(maximum_lipm_residual),
        maximum_joint_step_rad=maximum_joint_step,
    )
