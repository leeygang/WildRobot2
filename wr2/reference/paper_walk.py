"""ToddlerBot's ZMP preview/footstep recipe with WR2 geometry and six-DOF IK.

Reference: local toddlerbot/algorithms/{zmp_planner,zmp_walk}.py and
Tedrake et al., https://doi.org/10.1109/HUMANOIDS.2015.7363473.
Planning is host-side; training uses a generated lookup, with no TB dependency.
"""

from dataclasses import dataclass
from functools import lru_cache

import jax.numpy as jp
import mujoco
import numpy as np
from scipy.linalg import expm, solve_continuous_are

from wr2.reference.walk_zmp import WR2ZMPReference, _LEG_JOINT_SUFFIXES


@lru_cache(maxsize=32)
def _preview_matrices(height, scale):
    a = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    b = np.asarray([[0.0], [1.0]])
    c = np.asarray([[1.0, 0.0]])
    d = -height / 9.81
    # Length scales as scale**2 and acceleration as g: preserve the TB cost.
    r = np.asarray([[0.1 * scale**4 + d**2]])
    n = c.T * d
    s = solve_continuous_are(a, b, c.T @ c, r, s=n)
    nb = n.T + b.T @ s
    feedback = -np.linalg.solve(r, nb)
    a2 = nb.T @ np.linalg.solve(r, b.T) - a.T
    b2 = 2 * (c.T - nb.T / r[0, 0] * d)
    return feedback, a2, b2, b, d, r[0, 0]


def preview_com(time_steps, desired_zmp, initial_com, height, dt, scale=1.0):
    """TB's backward preview pass and forward Euler integration in both axes."""
    feedback, a2, b2, b, d, r = _preview_matrices(float(height), float(scale))
    final = desired_zmp[-1]
    offsets = desired_zmp[:-1] - final
    beta = -np.linalg.solve(a2, b2)[:, None, :] * offsets[None, :, :]
    alpha = np.zeros_like(beta)
    for i in range(len(offsets) - 1, -1, -1):
        endpoint = (
            -beta[:, i]
            if i == len(offsets) - 1
            else alpha[:, i + 1] + beta[:, i + 1] - beta[:, i]
        )
        alpha[:, i] = expm(-a2 * (time_steps[i + 1] - time_steps[i])) @ endpoint
    times = np.arange(int(np.ceil(time_steps[-1] / dt))) * dt
    segments = np.clip(
        np.searchsorted(time_steps, times, side="right") - 1, 0, len(offsets) - 1
    )
    exponentials = {}
    feedforward = []
    for t, i in zip(times, segments, strict=True):
        offset = round(float(t - time_steps[i]), 10)
        if offset not in exponentials:
            exponentials[offset] = expm(a2 * offset)
        gamma = d / r * offsets[i] - 0.5 / r * (b.T @ beta[:, i])[0]
        feedforward.append(
            (-0.5 / r * b.T @ exponentials[offset] @ alpha[:, i])[0] + gamma
        )
    com = np.zeros((len(times), 2))
    velocity = np.zeros_like(com)
    acceleration = np.zeros_like(com)
    com[0] = initial_com
    acceleration[0] = (feedback @ np.stack([com[0] - final, velocity[0]]))[
        0
    ] + feedforward[0]
    for i in range(1, len(times)):
        com[i] = com[i - 1] + velocity[i - 1] * dt
        velocity[i] = velocity[i - 1] + acceleration[i - 1] * dt
        acceleration[i] = (feedback @ np.stack([com[i] - final, velocity[i]]))[
            0
        ] + feedforward[i]
    return com, com - height / 9.81 * acceleration


def footstep_plan(command, cycle, initial_feet, scale, *, total_time=None):
    """TB's alternating XY/arc footsteps and linear swing, including turns."""
    total_time = 20.0 * scale if total_time is None else total_time
    cycles = int(np.ceil(total_time / cycle))
    stride = np.asarray(command) * total_time / (2 * cycles - 1)
    center = np.mean(initial_feet[:, :2], axis=0)
    width = (initial_feet[0, 1] - initial_feet[1, 1]) / 2
    footsteps = []
    for i in range(2 * cycles):
        side = i % 2
        if np.linalg.norm(command[:2]) > 1e-6:
            xy = initial_feet[side, :2] + i * stride[:2]
            yaw = 0.0
        else:
            yaw = i * stride[2]
            radius = np.sign(command[2]) * 0.1 * scale**2
            lateral = width if side == 0 else -width
            xy = center + [
                radius * np.sin(yaw) - lateral * np.sin(yaw),
                radius * (1 - np.cos(yaw)) + lateral * np.cos(yaw),
            ]
        footsteps.append([*xy, yaw, side])
    footsteps = np.asarray(footsteps)
    double = cycle / 6.0
    single = 2 * double
    time_steps = np.cumsum([0, double] + [single, double] * (len(footsteps) - 1))
    desired_zmp = np.repeat(footsteps[:, :2], 2, axis=0)
    return footsteps, time_steps, desired_zmp


def foot_trajectories(footsteps, time_steps, dt, swing_height, initial_feet):
    """Sample the native TB double support and triangular swing trajectories."""
    frames = int(np.ceil(time_steps[-1] / dt))
    positions = np.zeros((frames, 2, 3))
    yaws = np.zeros((frames, 2))
    torso_yaw = np.zeros(frames)
    stance = np.ones((frames, 2), dtype=bool)
    last_position = initial_feet.copy()
    last_yaw = np.zeros(2)
    last_torso_yaw = 0.0
    cursor = 0
    repeated = np.repeat(footsteps, 2, axis=0)
    for segment in range(len(time_steps) - 1):
        count = min(
            round((time_steps[segment + 1] - time_steps[segment]) / dt), frames - cursor
        )
        sl = slice(cursor, cursor + count)
        if segment % 2 == 0:
            positions[sl], yaws[sl], torso_yaw[sl] = (
                last_position,
                last_yaw,
                last_torso_yaw,
            )
        else:
            support = int(repeated[segment, 3])
            swing = 1 - support
            current_position = last_position.copy()
            current_position[support, :2] = repeated[segment, :2]
            current_yaw = last_yaw.copy()
            current_yaw[support] = repeated[segment, 2]
            target_position = current_position.copy()
            target_position[swing, :2] = repeated[segment + 1, :2]
            target_yaw = current_yaw.copy()
            target_yaw[swing] = repeated[segment + 1, 2]
            fraction = np.arange(count) / count
            positions[sl] = current_position + fraction[:, None, None] * (
                target_position - current_position
            )
            half = count // 2
            lift = np.concatenate(
                [np.arange(half), np.arange(count - half - 1, -1, -1)]
            )
            positions[sl, swing, 2] += swing_height * lift / max(half - 1, 1)
            yaws[sl] = current_yaw + fraction[:, None] * (target_yaw - current_yaw)
            torso_yaw[sl] = last_torso_yaw + fraction * (
                target_yaw[swing] - last_torso_yaw
            )
            stance[sl, swing] = False
            last_position, last_yaw = target_position, target_yaw
            last_torso_yaw = torso_yaw[sl][-1]
        cursor += count
    if cursor < frames:
        positions[cursor:], yaws[cursor:], torso_yaw[cursor:] = (
            last_position,
            last_yaw,
            last_torso_yaw,
        )
    return positions, yaws, torso_yaw, stance


@dataclass(frozen=True)
class PaperReferenceDiagnostics:
    command_samples_xy_m_s: tuple
    command_samples_xyz: tuple
    root_height_m: float
    com_height_m: float
    foot_half_width_m: float
    max_ik_position_residual_m: float
    max_ik_orientation_residual_rad: float
    lookup_length: int
    maximum_motor_offset_rad: float
    outside_policy_range_fraction: float
    maximum_requested_motor_speed_rad_s: float


_TABLE_CACHE = {}


def paper_command_grid(scale):
    """Native training lookup keys, shared with the reference viewer."""
    xy = np.stack(
        np.meshgrid(
            np.arange(-0.1, 0.2001, 0.02) * scale,
            np.arange(-0.1, 0.1001, 0.02) * scale,
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    turns = np.arange(-0.5, 0.5001, 0.02) / scale
    return np.unique(
        np.concatenate(
            [
                np.column_stack([xy, np.zeros(len(xy))]),
                np.column_stack([np.zeros((len(turns), 2)), turns]),
            ]
        ),
        axis=0,
    )


class PaperWalkReference:
    """Generated TB-style motor/contact lookup shared by rewards and critic."""

    def __init__(
        self,
        model,
        *,
        keyframe_id,
        actuator_names,
        config,
        control_dt,
        normalization_scale,
        command_vectors=None,
    ):
        self._dt = control_dt
        self._cycle = config.gait_cycle_s
        self._lookup_length = max(1000, config.episode_length)
        zmp = config.zmp_reference
        home = model.key_qpos[keyframe_id].copy()
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in actuator_names
            ]
        )
        self._joint_indices = model.jnt_qposadr[joint_ids]
        root_height = float(
            home[2] - 0.005 if zmp.root_height_m is None else zmp.root_height_m
        )
        scale = normalization_scale
        if command_vectors is None:
            command_vectors = paper_command_grid(scale)
        commands = np.unique(
            np.asarray(command_vectors, dtype=float).reshape(-1, 3), axis=0
        )
        signature = tuple(
            np.asarray(getattr(model, name)).tobytes()
            for name in (
                "names",
                "body_parentid",
                "body_pos",
                "body_quat",
                "body_mass",
                "body_ipos",
                "jnt_type",
                "jnt_bodyid",
                "jnt_pos",
                "jnt_axis",
                "jnt_range",
                "site_pos",
                "site_quat",
                "site_bodyid",
                "qpos0",
            )
        )
        key = (
            signature,
            home.tobytes(),
            tuple(actuator_names),
            commands.tobytes(),
            self._cycle,
            config.swing_height_m,
            root_height,
            zmp.com_height_m,
            control_dt,
            scale,
            self._lookup_length,
            zmp.ik_damping,
            zmp.ik_max_iterations,
            zmp.max_position_residual_m,
            zmp.max_orientation_residual_rad,
            config.policy_action_scale_rad,
        )
        if key not in _TABLE_CACHE:
            _TABLE_CACHE[key] = self._build(
                model,
                home,
                joint_ids,
                actuator_names,
                commands,
                root_height,
                config,
                scale,
            )
        self._host = _TABLE_CACHE[key]
        self._commands = jp.asarray(commands, dtype=jp.float32)
        self._motors = jp.asarray(self._host["motors"], dtype=jp.float32)
        self._stance = jp.asarray(self._host["stance"])
        self._heights = jp.asarray(self._host["heights"], dtype=jp.float32)
        self._standing_motors = jp.asarray(
            self._host["standing"][self._joint_indices], dtype=jp.float32
        )
        self.standing_qpos = self._host["standing"].copy()
        self.diagnostics = self._host["diagnostics"]
        if (
            self.diagnostics.max_ik_position_residual_m > zmp.max_position_residual_m
            or self.diagnostics.max_ik_orientation_residual_rad
            > zmp.max_orientation_residual_rad
        ):
            errors = self._host["ik_errors"]
            command_index, frame = np.unravel_index(
                np.argmax(errors[..., 0]), errors.shape[:2]
            )
            raise ValueError(
                f"Paper reference IK failed: command={commands[command_index].tolist()}, "
                f"lookup_frame={frame}, position={errors[command_index, frame, 0]:.6f}m, "
                f"angle={errors[command_index, frame, 1]:.6f}rad"
            )

    def _build(
        self, model, home, joint_ids, names, commands, root_height, config, scale
    ):
        data = mujoco.MjData(model)
        data.qpos[:] = home
        mujoco.mj_forward(model, data)
        sites = np.asarray(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_center"
                )
                for side in ("left", "right")
            ]
        )
        initial_feet = data.site_xpos[sites].copy()
        orientations = data.site_xmat[sites].reshape(2, 3, 3).copy()
        legs = [
            np.asarray(
                [names.index(f"{side}_{suffix}") for suffix in _LEG_JOINT_SUFFIXES]
            )
            for side in ("left", "right")
        ]
        zmp = config.zmp_reference

        def solve(base, positions, foot_yaws):
            errors = []
            for side, indices in enumerate(legs):
                yaw = foot_yaws[side]
                rotation = np.asarray(
                    [
                        [np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1],
                    ]
                )
                ids = joint_ids[indices]
                kwargs = dict(
                    site_id=int(sites[side]),
                    qpos_indices=model.jnt_qposadr[ids],
                    dof_indices=model.jnt_dofadr[ids],
                    lower=model.jnt_range[ids, 0],
                    upper=model.jnt_range[ids, 1],
                    target_position=positions[side],
                    target_orientation=rotation @ orientations[side],
                    damping=zmp.ik_damping,
                    max_iterations=zmp.ik_max_iterations,
                )
                joints, position_error, angle_error = WR2ZMPReference._solve_leg_ik(
                    model, data, home_qpos=base, **kwargs
                )
                if (
                    position_error > zmp.max_position_residual_m
                    or angle_error > zmp.max_orientation_residual_rad
                ):
                    # A straight-knee warm start can converge to the wrong
                    # branch. Retry from the crouched canonical leg seed.
                    seed = base.copy()
                    seed[model.jnt_qposadr[ids]] = home[model.jnt_qposadr[ids]]
                    retry = WR2ZMPReference._solve_leg_ik(
                        model, data, home_qpos=seed, **kwargs
                    )
                    if (
                        retry[1] ** 2 + retry[2] ** 2
                        < position_error**2 + angle_error**2
                    ):
                        joints, position_error, angle_error = retry
                base[model.jnt_qposadr[ids]] = joints
                errors.append((position_error, angle_error))
            return base, np.max(errors, axis=0)

        standing = home.copy()
        standing[2] = root_height
        standing, standing_errors = solve(standing, initial_feet, np.zeros(2))
        data.qpos[:] = standing
        mujoco.mj_forward(model, data)
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        initial_com = data.subtree_com[torso_id].copy()
        com_height = float(
            initial_com[2] if zmp.com_height_m is None else zmp.com_height_m
        )
        com_offset = initial_com[:2] - standing[:2]
        motors, masks, lifts, poses, cops, desired_cops, ik_errors = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        maximum_errors = standing_errors.copy()
        first_cycle = int(np.ceil(self._cycle / self._dt))
        end_frame = first_cycle + self._lookup_length
        for command in commands:
            # Cover the full 20s training episode and preview two additional
            # cycles after the retained frames. The finite planner's stopping
            # tail must not become a repeating walking reference.
            footsteps, times, desired = footstep_plan(
                command,
                self._cycle,
                initial_feet,
                scale,
                total_time=self._lookup_length * self._dt + 3 * self._cycle,
            )
            feet, yaws, torso_yaw, stance = foot_trajectories(
                footsteps, times, self._dt, config.swing_height_m, initial_feet
            )
            com, cop = preview_com(
                times, desired, initial_com[:2], com_height, self._dt, scale
            )
            if np.linalg.norm(command) < 1e-6:
                feet[:] = initial_feet
                stance[:] = True
                desired[:] = initial_com[:2]
                cop[:] = initial_com[:2]
            qposes = []
            command_errors = []
            previous = standing.copy()
            for i in range(first_cycle, end_frame):
                if np.linalg.norm(command) < 1e-6:
                    qpos, errors = standing.copy(), np.zeros(2)
                else:
                    qpos = previous.copy()
                    yaw = torso_yaw[i]
                    rotation = np.asarray(
                        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
                    )
                    qpos[:2] = com[i] - rotation @ com_offset
                    qpos[2] = root_height
                    qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                    qpos, errors = solve(qpos, feet[i], yaws[i])
                previous = qpos
                maximum_errors = np.maximum(maximum_errors, errors)
                qposes.append(qpos)
                command_errors.append(errors)
            qposes = np.asarray(qposes)
            motors.append(qposes[:, self._joint_indices])
            masks.append(
                stance[first_cycle:end_frame]
                if np.linalg.norm(command) > 1e-6
                else np.ones_like(stance[first_cycle:end_frame])
            )
            lifts.append(feet[first_cycle:end_frame, :, 2] - initial_feet[:, 2])
            poses.append(qposes)
            cops.append(cop[first_cycle:end_frame])
            indices = np.clip(
                np.searchsorted(
                    times, np.arange(first_cycle, end_frame) * self._dt, side="right"
                )
                - 1,
                0,
                len(desired) - 1,
            )
            desired_cops.append(desired[indices])
            ik_errors.append(command_errors)
        motors = np.asarray(motors)
        active = np.concatenate(legs)
        offsets = np.abs(motors[..., active] - standing[self._joint_indices[active]])
        diagnostics = PaperReferenceDiagnostics(
            tuple(map(tuple, np.unique(commands[:, :2], axis=0))),
            tuple(map(tuple, commands)),
            root_height,
            com_height,
            float(np.ptp(initial_feet[:, 1]) / 2),
            *map(float, maximum_errors),
            motors.shape[1],
            float(offsets.max()),
            float(np.mean(offsets > config.policy_action_scale_rad)),
            float(
                max(
                    np.max(np.abs(np.diff(motors[..., active], axis=1))),
                    np.max(
                        np.abs(
                            motors[:, 1, active] - standing[self._joint_indices[active]]
                        )
                    ),
                )
                / self._dt
            ),
        )
        return {
            "motors": motors,
            "stance": np.asarray(masks),
            "heights": np.asarray(lifts),
            "qpos": np.asarray(poses),
            "cop": np.asarray(cops),
            "desired_cop": np.asarray(desired_cops),
            "standing": standing,
            "commands": commands,
            "diagnostics": diagnostics,
            "sites": sites,
            "standing_cop": initial_com[:2].copy(),
            "ik_errors": np.asarray(ik_errors),
        }

    def get_reference(self, time_s, command):
        """TB's nearest command and episode-time indexing; zero/time-zero is static."""
        index = jp.argmin(jp.linalg.norm(self._commands - command, axis=1))
        frame = jp.mod(
            jp.round(time_s / self._dt).astype(jp.int32), self._motors.shape[1]
        )
        static = (jp.linalg.norm(command) < 1e-6) | (time_s < 1e-6)
        motors = jp.where(static, self._standing_motors, self._motors[index, frame])
        stance = jp.where(static, jp.ones(2, dtype=bool), self._stance[index, frame])
        heights = jp.where(static, jp.zeros(2), self._heights[index, frame])
        return motors, stance, heights

    def sample_time(self, time_s, command):
        """Full host frame for independent geometry and actuator diagnostics."""
        index = int(np.argmin(np.linalg.norm(self._host["commands"] - command, axis=1)))
        frame = round(time_s / self._dt) % self._motors.shape[1]
        if np.linalg.norm(command) < 1e-6 or time_s < 1e-6:
            return (
                self.standing_qpos.copy(),
                np.ones(2, dtype=bool),
                self._host["standing_cop"],
                self._host["standing_cop"],
            )
        return (
            self._host["qpos"][index, frame].copy(),
            self._host["stance"][index, frame],
            self._host["cop"][index, frame],
            self._host["desired_cop"][index, frame],
        )
