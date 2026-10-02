"""WR2 zero-moment-point walking reference for the privileged PPO critic.

The deployable actor does not consume this reference.  Like ToddlerBot's
``WalkZMPReference``, it gives the asymmetric critic a phase- and
command-conditioned bilateral leg trajectory.  WR2 has five rather than six
degrees of freedom per leg, so the lookup is generated from the canonical
MuJoCo model with damped numerical inverse kinematics.
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
    "knee_pitch",
    "ankle_pitch",
    "ankle_roll",
)


def periodic_lipm_lateral_reference(
    gait_phase: jax.Array,
    foot_half_width_m: float,
    com_height_m: float,
    gait_cycle_s: float,
) -> jax.Array:
    """Return periodic lateral CoM and implied ZMP positions.

    The CoM follows a sinusoid.  Its amplitude is selected so the Linear
    Inverted Pendulum Model relation ``zmp = com - h/g * com_acceleration``
    places the ZMP beneath the alternating support-foot centers.  Phase zero
    starts the left-foot swing, so the ZMP is on the right at pi/2.
    """
    angular_frequency = _TWO_PI / gait_cycle_s
    zmp_gain = 1.0 + com_height_m * angular_frequency**2 / _GRAVITY_M_S2
    com_y = -(foot_half_width_m / zmp_gain) * jp.sin(gait_phase)
    com_acceleration_y = -(angular_frequency**2) * com_y
    zmp_y = com_y - com_height_m / _GRAVITY_M_S2 * com_acceleration_y
    return jp.stack([com_y, zmp_y])


@dataclass(frozen=True)
class ZMPReferenceDiagnostics:
    """Host-side validation of the generated WR2 lookup table."""

    command_samples_m_s: tuple[float, ...]
    foot_half_width_m: float
    com_lateral_amplitude_m: float
    max_ik_position_residual_m: float
    max_ik_orientation_residual_rad: float


_TABLE_CACHE: dict[
    tuple[object, ...],
    tuple[np.ndarray, np.ndarray, ZMPReferenceDiagnostics],
] = {}


class WR2ZMPReference:
    """Command lookup containing a periodic ZMP plan and five-DOF leg IK."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        keyframe_id: int,
        actuator_names: tuple[str, ...],
        command_forward_range_m_s: tuple[float, float],
        gait_cycle_s: float,
        swing_height_m: float,
        com_height_m: float,
        phase_samples: int,
        command_samples: int,
        ik_damping: float,
        ik_max_iterations: int,
        max_position_residual_m: float,
        max_orientation_residual_rad: float,
    ) -> None:
        if phase_samples < 4 or phase_samples % 2:
            raise ValueError("ZMP phase_samples must be an even integer of at least four")
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

        low, high = command_forward_range_m_s
        if np.isclose(low, high):
            command_grid = np.asarray([low], dtype=np.float64)
        else:
            command_grid = np.linspace(low, high, command_samples, dtype=np.float64)

        model_signature = (
            model.nq,
            model.nv,
            bytes(model.names),
            tuple(np.round(home_joint_position, 10)),
            tuple(np.round(model.jnt_range[joint_ids].reshape(-1), 10)),
        )
        cache_key = (
            model_signature,
            tuple(actuator_names),
            tuple(np.round(command_grid, 10)),
            float(gait_cycle_s),
            float(swing_height_m),
            float(com_height_m),
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
            task_error = np.concatenate([position_error, orientation_error[:2]])
            if (
                np.linalg.norm(position_error) < 1e-5
                and np.linalg.norm(orientation_error[:2]) < 1e-4
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
                    jacobian_rotation[:2, dof_indices],
                ]
            )
            regularized = (
                task_jacobian @ task_jacobian.T
                + damping * np.eye(task_jacobian.shape[0])
            )
            joint_update = task_jacobian.T @ np.linalg.solve(
                regularized, task_error
            )
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
            float(np.linalg.norm(final_orientation_error[:2])),
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
        com_height_m: float,
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
            (command_grid.size, phase_samples, home_joint_position.size),
        ).copy()
        max_position_residual = 0.0
        max_orientation_residual = 0.0

        def smoothstep(value: float) -> float:
            value = float(np.clip(value, 0.0, 1.0))
            return value * value * (3.0 - 2.0 * value)

        for command_index, forward_speed in enumerate(command_grid):
            stride_half_range = forward_speed * gait_cycle_s / 4.0
            for phase_index, phase in enumerate(phase_grid):
                com_y = -com_lateral_amplitude_m * np.sin(phase)
                for side, phase_offset in (("left", 0.0), ("right", np.pi)):
                    local_phase = float(np.mod(phase + phase_offset, _TWO_PI))
                    if local_phase < np.pi:
                        progress = local_phase / np.pi
                        foot_x = stride_half_range * (
                            -1.0 + 2.0 * smoothstep(progress)
                        )
                        triangular = (
                            2.0 * progress
                            if progress < 0.5
                            else 2.0 * (1.0 - progress)
                        )
                        foot_z = swing_height_m * smoothstep(triangular)
                    else:
                        progress = (local_phase - np.pi) / np.pi
                        foot_x = stride_half_range * (1.0 - 2.0 * progress)
                        foot_z = 0.0

                    values = side_data[side]
                    target_delta_torso = np.asarray([foot_x, -com_y, foot_z])
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
            command_samples_m_s=tuple(float(value) for value in command_grid),
            foot_half_width_m=foot_half_width_m,
            com_lateral_amplitude_m=com_lateral_amplitude_m,
            max_ik_position_residual_m=max_position_residual,
            max_ik_orientation_residual_rad=max_orientation_residual,
        )
        return command_grid, lookup, diagnostics

    def joint_position(
        self,
        gait_phase: jax.Array,
        command: jax.Array,
        is_walking: jax.Array,
    ) -> jax.Array:
        """Interpolate the nearest-command joint reference at ``gait_phase``."""
        command_index = jp.argmin(jp.abs(self._command_grid - command[0]))
        phase_position = (
            jp.mod(gait_phase, _TWO_PI) * self._phase_samples / _TWO_PI
        )
        phase_floor = jp.floor(phase_position).astype(jp.int32)
        lower_index = jp.mod(phase_floor, self._phase_samples)
        upper_index = jp.mod(lower_index + 1, self._phase_samples)
        fraction = phase_position - jp.floor(phase_position)
        lower = self._joint_lookup[command_index, lower_index]
        upper = self._joint_lookup[command_index, upper_index]
        reference = lower + fraction * (upper - lower)
        return jp.where(is_walking, reference, self._home_joint_position)
