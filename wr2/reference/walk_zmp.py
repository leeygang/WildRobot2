"""WR2 zero-moment-point walking reference for the privileged PPO critic.

The deployable actor does not consume this reference.  Like ToddlerBot's
``WalkZMPReference``, it gives the asymmetric critic a phase- and
command-conditioned bilateral leg trajectory. The six-DOF leg lookup is
generated from the canonical model, including full foot orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
import numpy as np


_GRAVITY_M_S2 = 9.81
_TWO_PI = 2.0 * np.pi
_LEG_JOINT_SUFFIXES = (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee_pitch",
    "ankle_pitch",
    "ankle_roll",
)


def reference_command_grid(
    forward_range: tuple[float, float], lateral_range: tuple[float, float],
    samples: int,
) -> tuple[tuple[float, float], ...]:
    """XY lattice including the envelope of TB's heading-corrected commands."""
    axes = []
    radius = float(np.linalg.norm([max(map(abs, forward_range)), max(map(abs, lateral_range))]))
    for low, high in (forward_range, lateral_range):
        axis = np.asarray([low]) if np.isclose(low, high) else np.linspace(low, high, samples)
        if lateral_range != (0.0, 0.0):
            axis = np.unique(np.concatenate([axis, [-radius, radius]]))
        axes.append(axis)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    return tuple(tuple(map(float, command)) for command in grid)


def periodic_lipm_lateral_reference(
    gait_phase: jax.Array,
    foot_half_width_m: float,
    com_height_m: float,
    gait_cycle_s: float,
    lateral_speed_m_s: float = 0.0,
    single_double_support_ratio: float = 2.0,
) -> jax.Array:
    """Return periodic lateral CoM and implied ZMP positions.

    The reduced-order CoM follows a sinusoid.  Its amplitude is selected so the
    Linear Inverted Pendulum Model relation
    ``zmp = com - h/g * com_acceleration`` places the ZMP beneath the
    alternating support-foot centers.  Phase zero is the middle of double
    support before left-foot swing; the ZMP is on the right at pi/2.
    """
    return _lipm_lateral_reference(
        jp, gait_phase, foot_half_width_m, com_height_m, gait_cycle_s,
        lateral_speed_m_s, single_double_support_ratio,
    )[:2]


def _lipm_lateral_reference(array_lib, phase, width, height, cycle, lateral_speed, support_ratio):
    """Invert the periodic LIPM for sway plus the translating stance centers.

    Stance centers advance twice per cycle. The second harmonic matches their
    offsets at single-support boundaries; each harmonic obeys z = c - h/g*c''.
    Zero lateral speed retains the existing forward reference exactly.
    """
    angular_frequency = _TWO_PI / cycle
    transition_half_width = np.pi / (2.0 * (support_ratio + 1.0))
    stance_length = np.pi + 2.0 * transition_half_width
    half_stride = lateral_speed * cycle * stance_length / (2.0 * _TWO_PI)
    boundary_offset = half_stride * (np.pi - 2.0 * transition_half_width) / stance_length
    zmp_advance = boundary_offset / np.sin(2.0 * transition_half_width)
    zmp_gain = 1.0 + height * angular_frequency**2 / _GRAVITY_M_S2
    advance_gain = 1.0 + height * (2.0 * angular_frequency)**2 / _GRAVITY_M_S2
    sway = -(width / zmp_gain) * array_lib.sin(phase)
    advance = zmp_advance / advance_gain * array_lib.sin(2.0 * phase)
    com_y = sway + advance
    com_acceleration_y = -(angular_frequency**2) * sway - (2.0 * angular_frequency)**2 * advance
    zmp_y = com_y - height / _GRAVITY_M_S2 * com_acceleration_y
    return array_lib.stack([com_y, zmp_y, com_acceleration_y])


def _periodic_foot_reference(
    array_lib,
    gait_phase,
    forward_speed_m_s: float,
    gait_cycle_s: float,
    swing_height_m: float,
    single_double_support_ratio: float,
):
    """Return left/right x, z, and stance states for one periodic step."""
    phase = array_lib.mod(gait_phase, _TWO_PI)
    local_phase = array_lib.stack([phase, array_lib.mod(phase + np.pi, _TWO_PI)])
    transition_half_width = np.pi / (2.0 * (single_double_support_ratio + 1.0))
    swing_start = transition_half_width
    swing_end = np.pi - transition_half_width
    swing_length = swing_end - swing_start
    stance_length = _TWO_PI - swing_length
    swing = (local_phase > swing_start) & (local_phase < swing_end)
    swing_progress = array_lib.clip(
        (local_phase - swing_start) / swing_length, 0.0, 1.0
    )

    # One foot advances by speed * cycle each cycle.  The amplitude below also
    # makes the stance-foot world velocity zero when the torso moves at speed.
    half_range = forward_speed_m_s * gait_cycle_s * stance_length / (2.0 * _TWO_PI)
    stance_slope = -2.0 * half_range / stance_length
    p2 = swing_progress * swing_progress
    p3 = p2 * swing_progress
    x_swing = (
        (2.0 * p3 - 3.0 * p2 + 1.0) * -half_range
        + (p3 - 2.0 * p2 + swing_progress) * swing_length * stance_slope
        + (-2.0 * p3 + 3.0 * p2) * half_range
        + (p3 - p2) * swing_length * stance_slope
    )
    stance_progress = array_lib.where(
        local_phase >= swing_end,
        (local_phase - swing_end) / stance_length,
        (local_phase + _TWO_PI - swing_end) / stance_length,
    )
    x_stance = half_range * (1.0 - 2.0 * stance_progress)
    foot_x = array_lib.where(swing, x_swing, x_stance)

    triangular = array_lib.where(
        swing_progress < 0.5,
        2.0 * swing_progress,
        2.0 * (1.0 - swing_progress),
    )
    triangular = array_lib.clip(triangular, 0.0, 1.0)
    smooth_height = triangular * triangular * (3.0 - 2.0 * triangular)
    foot_z = array_lib.where(swing, swing_height_m * smooth_height, 0.0)
    return foot_x, foot_z, ~swing


def periodic_foot_reference(
    gait_phase: jax.Array,
    forward_speed_m_s: float,
    gait_cycle_s: float,
    swing_height_m: float,
    single_double_support_ratio: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return left/right foot x, height, and stance references."""
    return _periodic_foot_reference(
        jp,
        gait_phase,
        forward_speed_m_s,
        gait_cycle_s,
        swing_height_m,
        single_double_support_ratio,
    )


@dataclass(frozen=True)
class ZMPReferenceDiagnostics:
    """Host-side validation of the generated WR2 lookup table."""

    command_samples_m_s: tuple[float, ...]
    command_samples_xy_m_s: tuple[tuple[float, float], ...]
    root_height_m: float
    com_height_m: float
    foot_half_width_m: float
    com_lateral_amplitude_m: float
    max_ik_position_residual_m: float
    max_ik_orientation_residual_rad: float


@dataclass(frozen=True)
class ZMPReferenceSample:
    """One host-side frame for validation and visualization."""

    phase_rad: float
    command_forward_m_s: float
    command_lateral_m_s: float
    joint_position_rad: np.ndarray
    com_lateral_m: float
    zmp_lateral_m: float
    com_lateral_acceleration_m_s2: float
    root_height_m: float
    foot_forward_m: np.ndarray
    foot_lateral_m: np.ndarray
    foot_height_m: np.ndarray
    stance_mask: np.ndarray


_TABLE_CACHE: dict[
    tuple[object, ...],
    tuple[np.ndarray, np.ndarray, ZMPReferenceDiagnostics],
] = {}


class WR2ZMPReference:
    """Command lookup containing a periodic ZMP plan and six-DOF leg IK."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        keyframe_id: int,
        actuator_names: tuple[str, ...],
        command_forward_range_m_s: tuple[float, float],
        command_lateral_range_m_s: tuple[float, float] = (0.0, 0.0),
        command_vectors_m_s: tuple[tuple[float, float], ...] | None = None,
        gait_cycle_s: float,
        swing_height_m: float,
        com_height_m: float | None,
        root_height_m: float | None = None,
        single_double_support_ratio: float,
        phase_samples: int,
        command_samples: int,
        ik_damping: float,
        ik_max_iterations: int,
        max_position_residual_m: float,
        max_orientation_residual_rad: float,
    ) -> None:
        if phase_samples < 4 or phase_samples % 2:
            raise ValueError(
                "ZMP phase_samples must be an even integer of at least four"
            )
        if command_samples < 1:
            raise ValueError("ZMP command_samples must be positive")

        home_qpos = np.asarray(model.key_qpos[keyframe_id], dtype=np.float64)
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in actuator_names
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            raise ValueError("ZMP reference could not resolve every actuator joint")
        joint_qpos_indices = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
        home_joint_position = home_qpos[joint_qpos_indices]
        if root_height_m is None:
            # New CAD knees stop at 90 degrees: the old extra 10 mm crouch
            # cannot preserve flat-foot orientation at a 40 mm swing apex.
            root_height_m = float(home_qpos[2] - 0.005)

        command_grid = np.asarray(
            command_vectors_m_s if command_vectors_m_s is not None else reference_command_grid(
                command_forward_range_m_s, command_lateral_range_m_s, command_samples
            ), dtype=np.float64,
        )

        model_signature = (
            model.nq,
            model.nv,
            bytes(model.names),
            tuple(np.round(home_joint_position, 10)),
            tuple(np.round(model.jnt_range[joint_ids].reshape(-1), 10)),
            model.body_pos.tobytes(), model.body_quat.tobytes(),
            model.body_mass.tobytes(), model.site_pos.tobytes(),
        )
        cache_key = (
            model_signature,
            tuple(actuator_names),
            tuple(np.round(command_grid.reshape(-1), 10)),
            float(gait_cycle_s),
            float(swing_height_m),
            com_height_m,
            root_height_m,
            float(single_double_support_ratio),
            int(phase_samples),
            float(ik_damping),
            int(ik_max_iterations),
        )
        cached = _TABLE_CACHE.get(cache_key)
        if cached is None:
            cached = self._build_lookup(
                model,
                home_qpos=home_qpos,
                actuator_names=actuator_names,
                joint_ids=joint_ids,
                joint_qpos_indices=joint_qpos_indices,
                command_grid=command_grid,
                gait_cycle_s=gait_cycle_s,
                swing_height_m=swing_height_m,
                com_height_m=com_height_m,
                root_height_m=root_height_m,
                single_double_support_ratio=single_double_support_ratio,
                phase_samples=phase_samples,
                ik_damping=ik_damping,
                ik_max_iterations=ik_max_iterations,
            )
            _TABLE_CACHE[cache_key] = cached

        command_grid, joint_lookup, diagnostics = cached
        if diagnostics.max_ik_position_residual_m > max_position_residual_m:
            raise ValueError(
                "WR2 ZMP IK residual exceeds its configured bound: "
                f"{diagnostics.max_ik_position_residual_m:.6f} > "
                f"{max_position_residual_m:.6f} m"
            )
        if diagnostics.max_ik_orientation_residual_rad > max_orientation_residual_rad:
            raise ValueError(
                "WR2 ZMP IK orientation residual exceeds its configured bound: "
                f"{diagnostics.max_ik_orientation_residual_rad:.6f} > "
                f"{max_orientation_residual_rad:.6f} rad"
            )

        self._command_grid = jp.asarray(command_grid, dtype=jp.float32)
        self._joint_lookup = jp.asarray(joint_lookup, dtype=jp.float32)
        self._home_joint_position = jp.asarray(home_joint_position, dtype=jp.float32)
        self._phase_samples = int(phase_samples)
        self._command_grid_numpy = np.asarray(command_grid)
        self._gait_cycle_s = float(gait_cycle_s)
        self._swing_height_m = float(swing_height_m)
        self._com_height_m = diagnostics.com_height_m
        self._root_height_m = diagnostics.root_height_m
        self._home_root_height_m = float(home_qpos[2])
        self._single_double_support_ratio = float(single_double_support_ratio)
        self._foot_half_width_m = diagnostics.foot_half_width_m
        self.diagnostics = diagnostics

    @staticmethod
    def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 0.5 * sum(
            (np.cross(current[:, index], target[:, index]) for index in range(3)),
            start=np.zeros(3),
        )

    @classmethod
    def _solve_leg_ik(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        home_qpos: np.ndarray,
        site_id: int,
        qpos_indices: np.ndarray,
        dof_indices: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        damping: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, float, float]:
        qpos = home_qpos.copy()
        for _ in range(max_iterations):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            position_error = target_position - data.site_xpos[site_id]
            orientation_error = cls._orientation_error(
                data.site_xmat[site_id].reshape(3, 3), target_orientation
            )
            task_error = np.concatenate([position_error, orientation_error])
            if (
                np.linalg.norm(position_error) < 1e-5
                and np.linalg.norm(orientation_error) < 1e-4
            ):
                break

            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                site_id,
            )
            task_jacobian = np.vstack(
                [
                    jacobian_position[:, dof_indices],
                    jacobian_rotation[:, dof_indices],
                ]
            )
            regularized = task_jacobian @ task_jacobian.T + damping * np.eye(
                task_jacobian.shape[0]
            )
            joint_update = task_jacobian.T @ np.linalg.solve(regularized, task_error)
            qpos[qpos_indices] = np.clip(
                qpos[qpos_indices] + np.clip(joint_update, -0.08, 0.08),
                lower,
                upper,
            )

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        final_position_error = target_position - data.site_xpos[site_id]
        final_orientation_error = cls._orientation_error(
            data.site_xmat[site_id].reshape(3, 3), target_orientation
        )
        return (
            qpos[qpos_indices].copy(),
            float(np.linalg.norm(final_position_error)),
            float(np.linalg.norm(final_orientation_error)),
        )

    @classmethod
    def _build_lookup(
        cls,
        model: mujoco.MjModel,
        *,
        home_qpos: np.ndarray,
        actuator_names: tuple[str, ...],
        joint_ids: np.ndarray,
        joint_qpos_indices: np.ndarray,
        command_grid: np.ndarray,
        gait_cycle_s: float,
        swing_height_m: float,
        com_height_m: float | None,
        root_height_m: float,
        single_double_support_ratio: float,
        phase_samples: int,
        ik_damping: float,
        ik_max_iterations: int,
    ) -> tuple[np.ndarray, np.ndarray, ZMPReferenceDiagnostics]:
        data = mujoco.MjData(model)
        data.qpos[:] = home_qpos
        mujoco.mj_forward(model, data)

        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if torso_id < 0:
            raise ValueError("WR2 ZMP reference requires the torso body")
        world_from_torso = data.xmat[torso_id].reshape(3, 3).copy()
        torso_position = data.xpos[torso_id].copy()

        side_data: dict[str, dict[str, np.ndarray | int]] = {}
        actuator_index = {name: index for index, name in enumerate(actuator_names)}
        foot_lateral_positions = []
        for side in ("left", "right"):
            names = tuple(f"{side}_{suffix}" for suffix in _LEG_JOINT_SUFFIXES)
            try:
                output_indices = np.asarray(
                    [actuator_index[name] for name in names], dtype=np.int32
                )
            except KeyError as exc:
                raise ValueError(
                    f"WR2 ZMP reference is missing leg actuator {exc.args[0]}"
                ) from exc
            side_joint_ids = joint_ids[output_indices]
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_center"
            )
            if site_id < 0:
                raise ValueError(f"WR2 ZMP reference is missing {side}_foot_center")
            site_position = data.site_xpos[site_id].copy()
            site_position_torso = world_from_torso.T @ (site_position - torso_position)
            foot_lateral_positions.append(site_position_torso[1])
            side_data[side] = {
                "output_indices": output_indices,
                "qpos_indices": np.asarray(
                    model.jnt_qposadr[side_joint_ids], dtype=np.int32
                ),
                "dof_indices": np.asarray(
                    model.jnt_dofadr[side_joint_ids], dtype=np.int32
                ),
                "lower": np.asarray(model.jnt_range[side_joint_ids, 0]),
                "upper": np.asarray(model.jnt_range[side_joint_ids, 1]),
                "site_id": int(site_id),
                "site_position": site_position,
                "site_orientation": data.site_xmat[site_id].reshape(3, 3).copy(),
            }

        if com_height_m is None:
            # TB keeps LIPM CoM height separate from pelvis/root height. Measure
            # this model's actual crouched stance instead of borrowing root Z.
            reference_qpos = home_qpos.copy()
            reference_qpos[2] = root_height_m
            for values in side_data.values():
                joints, _, _ = cls._solve_leg_ik(
                    model, data, home_qpos=reference_qpos,
                    site_id=int(values["site_id"]),
                    qpos_indices=np.asarray(values["qpos_indices"]),
                    dof_indices=np.asarray(values["dof_indices"]),
                    lower=np.asarray(values["lower"]), upper=np.asarray(values["upper"]),
                    target_position=np.asarray(values["site_position"]),
                    target_orientation=np.asarray(values["site_orientation"]),
                    damping=ik_damping, max_iterations=ik_max_iterations,
                )
                reference_qpos[np.asarray(values["qpos_indices"])] = joints
            data.qpos[:] = reference_qpos
            mujoco.mj_forward(model, data)
            com_height_m = float(data.subtree_com[torso_id, 2])
        if com_height_m <= 0 or root_height_m <= 0:
            raise ValueError("Reference root and CoM heights must be positive")
        foot_half_width_m = float(np.mean(np.abs(foot_lateral_positions)))
        angular_frequency = _TWO_PI / gait_cycle_s
        zmp_gain = 1.0 + com_height_m * angular_frequency**2 / _GRAVITY_M_S2
        com_lateral_amplitude_m = foot_half_width_m / zmp_gain

        phase_grid = np.arange(phase_samples, dtype=np.float64) * (
            _TWO_PI / phase_samples
        )
        home_joint_position = home_qpos[joint_qpos_indices]
        lookup = np.broadcast_to(
            home_joint_position,
            (len(command_grid), phase_samples, home_joint_position.size),
        ).copy()
        max_position_residual = 0.0
        max_orientation_residual = 0.0
        root_height_offset = float(home_qpos[2] - root_height_m)

        for command_index, (forward_speed, lateral_speed) in enumerate(command_grid):
            for phase_index, phase in enumerate(phase_grid):
                com_y = _lipm_lateral_reference(
                    np, phase, foot_half_width_m, com_height_m, gait_cycle_s,
                    float(lateral_speed), single_double_support_ratio,
                )[0]
                foot_x, foot_z, _ = _periodic_foot_reference(
                    np,
                    phase,
                    float(forward_speed),
                    gait_cycle_s,
                    swing_height_m,
                    single_double_support_ratio,
                )
                foot_y, _, _ = _periodic_foot_reference(
                    np, phase, float(lateral_speed), gait_cycle_s,
                    swing_height_m, single_double_support_ratio,
                )
                for side_index, side in enumerate(("left", "right")):
                    values = side_data[side]
                    target_delta_torso = np.asarray(
                        [
                            foot_x[side_index],
                            foot_y[side_index] - com_y,
                            foot_z[side_index] + root_height_offset,
                        ]
                    )
                    target_position = np.asarray(values["site_position"]) + (
                        world_from_torso @ target_delta_torso
                    )
                    joint_position, position_residual, orientation_residual = (
                        cls._solve_leg_ik(
                            model,
                            data,
                            home_qpos=home_qpos,
                            site_id=int(values["site_id"]),
                            qpos_indices=np.asarray(values["qpos_indices"]),
                            dof_indices=np.asarray(values["dof_indices"]),
                            lower=np.asarray(values["lower"]),
                            upper=np.asarray(values["upper"]),
                            target_position=target_position,
                            target_orientation=np.asarray(values["site_orientation"]),
                            damping=ik_damping,
                            max_iterations=ik_max_iterations,
                        )
                    )
                    lookup[
                        command_index,
                        phase_index,
                        np.asarray(values["output_indices"]),
                    ] = joint_position
                    max_position_residual = max(
                        max_position_residual, position_residual
                    )
                    max_orientation_residual = max(
                        max_orientation_residual, orientation_residual
                    )

        diagnostics = ZMPReferenceDiagnostics(
            command_samples_m_s=tuple(float(value) for value in np.unique(command_grid[:, 0])),
            command_samples_xy_m_s=tuple(tuple(map(float, value)) for value in command_grid),
            root_height_m=float(root_height_m),
            com_height_m=float(com_height_m),
            foot_half_width_m=foot_half_width_m,
            com_lateral_amplitude_m=com_lateral_amplitude_m,
            max_ik_position_residual_m=max_position_residual,
            max_ik_orientation_residual_rad=max_orientation_residual,
        )
        return command_grid, lookup, diagnostics

    def sample(
        self, gait_phase: float, command_forward_m_s: float,
        command_lateral_m_s: float = 0.0,
    ) -> ZMPReferenceSample:
        """Return a complete host-side reference frame for inspection."""
        command_index = int(
            np.argmin(np.linalg.norm(
                self._command_grid_numpy - [command_forward_m_s, command_lateral_m_s],
                axis=1,
            ))
        )
        matched_command, matched_lateral = map(float, self._command_grid_numpy[command_index])
        is_walking = np.linalg.norm([command_forward_m_s, command_lateral_m_s]) >= 1e-6
        phase = float(np.mod(gait_phase, _TWO_PI))
        joint_position = np.asarray(
            self.joint_position(
                jp.asarray(phase),
                jp.asarray([matched_command, matched_lateral, 0.0]),
                jp.asarray(is_walking),
            )
        )
        com_y, zmp_y, com_acceleration_y = _lipm_lateral_reference(
                np, phase,
                self._foot_half_width_m,
                self._com_height_m,
                self._gait_cycle_s,
                matched_lateral, self._single_double_support_ratio,
        )
        foot_x, foot_z, stance = _periodic_foot_reference(
            np,
            phase,
            matched_command,
            self._gait_cycle_s,
            self._swing_height_m,
            self._single_double_support_ratio,
        )
        foot_y, _, _ = _periodic_foot_reference(
            np, phase, matched_lateral, self._gait_cycle_s,
            self._swing_height_m, self._single_double_support_ratio,
        )
        if not is_walking:
            matched_command = matched_lateral = 0.0
            com_y = zmp_y = com_acceleration_y = 0.0
            foot_x = foot_y = foot_z = np.zeros(2)
            stance = np.ones(2, dtype=np.bool_)
        return ZMPReferenceSample(
            phase_rad=phase,
            command_forward_m_s=matched_command,
            command_lateral_m_s=matched_lateral,
            joint_position_rad=joint_position,
            com_lateral_m=float(com_y),
            zmp_lateral_m=float(zmp_y),
            com_lateral_acceleration_m_s2=float(com_acceleration_y),
            root_height_m=self._root_height_m if is_walking else self._home_root_height_m,
            foot_forward_m=np.asarray(foot_x, dtype=np.float64),
            foot_lateral_m=np.asarray(foot_y, dtype=np.float64),
            foot_height_m=np.asarray(foot_z, dtype=np.float64),
            stance_mask=np.asarray(stance, dtype=np.bool_),
        )

    def joint_position(
        self,
        gait_phase: jax.Array,
        command: jax.Array,
        is_walking: jax.Array,
        heading_error_rad: jax.Array = 0.0,
    ) -> jax.Array:
        """Interpolate the nearest-command joint reference at ``gait_phase``."""
        cosine, sine = jp.cos(heading_error_rad), jp.sin(heading_error_rad)
        command_xy = jp.stack([
            cosine * command[0] - sine * command[1],
            sine * command[0] + cosine * command[1],
        ])
        command_index = jp.argmin(jp.linalg.norm(self._command_grid - command_xy, axis=1))
        phase_position = jp.mod(gait_phase, _TWO_PI) * self._phase_samples / _TWO_PI
        phase_floor = jp.floor(phase_position).astype(jp.int32)
        lower_index = jp.mod(phase_floor, self._phase_samples)
        upper_index = jp.mod(lower_index + 1, self._phase_samples)
        fraction = phase_position - jp.floor(phase_position)
        lower = self._joint_lookup[command_index, lower_index]
        upper = self._joint_lookup[command_index, upper_index]
        reference = lower + fraction * (upper - lower)
        return jp.where(is_walking, reference, self._home_joint_position)
