"""Phase-guided Brax/MJX walking environment for WR2."""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from mujoco.mjx._src import support as mjx_support

from wr2.locomotion.configs import WalkingEnvConfig, load_training_config
from wr2.reference.walk_zmp import WR2ZMPReference, periodic_foot_reference
from wr2.sim.robot import RobotDescription


_TWO_PI = 2.0 * np.pi


def _smoothstep(value: jax.Array) -> jax.Array:
    value = jp.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def expected_foot_heights(
    gait_phase: jax.Array,
    swing_height_m: float,
    is_walking: jax.Array,
) -> jax.Array:
    """Return ToddlerBot's left/right half-cycle swing-height targets."""
    phase = jp.mod(gait_phase, _TWO_PI)

    def swing_profile(progress: jax.Array) -> jax.Array:
        triangular = jp.where(progress < 0.5, 2.0 * progress, 2.0 * (1.0 - progress))
        return swing_height_m * _smoothstep(triangular)

    left_progress = phase / np.pi
    right_progress = (phase - np.pi) / np.pi
    left_height = jp.where(phase < np.pi, swing_profile(left_progress), 0.0)
    right_height = jp.where(phase >= np.pi, swing_profile(right_progress), 0.0)
    targets = jp.stack([left_height, right_height])
    return jp.where(is_walking, targets, jp.zeros(2))


def expected_foot_contacts(
    gait_phase: jax.Array,
    is_walking: jax.Array,
    single_double_support_ratio: float = 2.0,
) -> jax.Array:
    """Return desired left/right stance contact for one gait cycle."""
    _, _, contacts = periodic_foot_reference(
        gait_phase,
        0.0,
        1.0,
        0.0,
        single_double_support_ratio,
    )
    return jp.where(is_walking, contacts, jp.ones(2, dtype=jp.bool_))


def normalized_action_rate_cost(
    action: jax.Array, previous_action: jax.Array
) -> jax.Array:
    """Return ToddlerBot's sum-squared consecutive policy-action change."""
    return jp.sum(jp.square(action - previous_action))


def support_contact_active(
    contact_distance: jax.Array,
    contact_force_world: jax.Array,
    threshold_n: float,
) -> jax.Array:
    """Return ToddlerBot-style support contacts from upward world force."""
    return (contact_distance < 0.0) & (contact_force_world[:, 2] > threshold_n)


def _rotate_vector(quaternion: jax.Array, vector: jax.Array) -> jax.Array:
    """Rotate a vector using a normalized wxyz quaternion."""
    quaternion = quaternion / jp.linalg.norm(quaternion)
    scalar = quaternion[0]
    xyz = quaternion[1:]
    return (
        (2.0 * scalar * scalar - 1.0) * vector
        + 2.0 * jp.dot(xyz, vector) * xyz
        + 2.0 * scalar * jp.cross(xyz, vector)
    )


def _inverse_rotate_vector(quaternion: jax.Array, vector: jax.Array) -> jax.Array:
    return _rotate_vector(quaternion * jp.array([1.0, -1.0, -1.0, -1.0]), vector)


def feet_lateral_distance(
    torso_quaternion: jax.Array, foot_delta_world: jax.Array
) -> jax.Array:
    """Return yaw-frame lateral foot separation, matching ToddlerBot."""
    heading_xy = _rotate_vector(torso_quaternion, jp.asarray([1.0, 0.0, 0.0]))[:2]
    heading_xy = heading_xy / jp.maximum(jp.linalg.norm(heading_xy), 1e-6)
    lateral_xy = jp.stack([-heading_xy[1], heading_xy[0]])
    return jp.abs(jp.dot(foot_delta_world[:2], lateral_xy))


def feet_orientation_error(
    foot_projected_gravity: jax.Array,
    home_foot_projected_gravity: jax.Array,
) -> jax.Array:
    """Return foot tilt relative to each foot's walk-home sole orientation."""
    foot_projected_gravity = foot_projected_gravity / jp.linalg.norm(
        foot_projected_gravity, axis=1, keepdims=True
    )
    home_foot_projected_gravity = home_foot_projected_gravity / jp.linalg.norm(
        home_foot_projected_gravity, axis=1, keepdims=True
    )
    cross = jp.cross(foot_projected_gravity, home_foot_projected_gravity)
    # This is ToddlerBot's sin(tilt) cost generalized to a model whose foot
    # body's local vertical axis is Y instead of Z.
    return jp.sum(jp.sqrt(jp.sum(jp.square(cross), axis=1) + 1e-12))


def _quaternion_multiply(left: jax.Array, right: jax.Array) -> jax.Array:
    """Multiply two wxyz quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jp.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _roll_pitch_quaternion(roll: jax.Array, pitch: jax.Array) -> jax.Array:
    """Return a wxyz quaternion with zero yaw."""
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    return jp.asarray(
        [
            jp.cos(half_roll) * jp.cos(half_pitch),
            jp.sin(half_roll) * jp.cos(half_pitch),
            jp.cos(half_roll) * jp.sin(half_pitch),
            -jp.sin(half_roll) * jp.sin(half_pitch),
        ]
    )


class WR2WalkingEnv(PipelineEnv):
    """Flat-ground environment using only deployable actor observations."""

    def __init__(
        self,
        config: WalkingEnvConfig | None = None,
        *,
        add_observation_noise: bool = True,
    ):
        self.config = config or load_training_config().environment
        self.robot = RobotDescription.load(include_local_calibration=False)
        self.add_observation_noise = add_observation_noise
        self._joint_velocity_observation_scale = float(
            self.robot.config["observation"]["scales"]["joint_velocity"]
        )
        self._privileged_linear_velocity_scale = float(
            self.config.privileged_linear_velocity_scale
        )
        self._privileged_actuator_force_scale = float(
            self.config.privileged_actuator_force_scale
        )

        scene_path = self.robot.directory / "scene_mjx.xml"
        if not scene_path.exists():
            raise FileNotFoundError(
                f"Missing {scene_path}; run python -m wr2.tools.post_process"
            )
        mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "walk_home")
        if key_id < 0:
            raise ValueError("MJX model is missing the walk_home keyframe")

        foot_site_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_center"
                )
                for side in ("left", "right")
            ],
            dtype=np.int32,
        )
        if np.any(foot_site_ids < 0):
            raise ValueError("MJX model is missing a configured foot-center site")
        home_data = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, home_data, key_id)
        mujoco.mj_forward(mj_model, home_data)

        system = mjcf.load_model(mj_model)
        n_frames = round(self.robot.control_period_s / self.robot.simulation_timestep_s)
        super().__init__(system, backend="mjx", n_frames=n_frames)

        self._default_qpos = jp.asarray(mj_model.key_qpos[key_id])
        # The MJX keyframe control vector is zero because its actuators accept
        # torque. Policy targets still use the canonical walk-home positions.
        self._home_ctrl = jp.asarray(self.robot.home_position_rad)
        self._ctrl_lower = jp.asarray(self.robot.lower_limit_rad)
        self._ctrl_upper = jp.asarray(self.robot.upper_limit_rad)
        self._foot_site_ids = jp.asarray(foot_site_ids)
        self._home_foot_site_z = jp.asarray(home_data.site_xpos[foot_site_ids, 2])
        foot_body_ids = np.asarray(
            [
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
                for side in ("left", "right")
            ],
            dtype=np.int32,
        )
        if np.any(foot_body_ids < 0):
            raise ValueError("MJX model is missing a configured foot body")
        world_from_foot = home_data.xmat[foot_body_ids].reshape(2, 3, 3)
        self._home_foot_projected_gravity = jp.asarray(
            np.einsum(
                "bij,j->bi",
                np.transpose(world_from_foot, (0, 2, 1)),
                np.asarray([0.0, 0.0, -1.0]),
            )
        )
        self._gait_phase_increment = float(
            _TWO_PI * self.robot.control_period_s / self.config.gait_cycle_s
        )
        servo_config = self.robot.config["actuators"]["htd45hServo"]
        self._torque_exposure_reference_nm = float(
            servo_config["maximum_validated_load_nm"]
        )
        if self._torque_exposure_reference_nm <= 0.0:
            raise ValueError("Torque-exposure reference must be positive")
        self._command_resolution_rad = float(servo_config["command_resolution_rad"])
        self._servo_kp = float(servo_config["kp_sim"])
        self._servo_kv = float(servo_config["kv_sim"])
        self._servo_torque_limit_nm = float(servo_config["torque_limit_nm"])
        self._servo_max_speed_rad_s = float(servo_config["vendor_no_load_speed_rad_s"])
        self._servo_peak_torque_speed_rad_s = float(
            servo_config["peak_torque_speed_rad_s"]
        )
        self._servo_torque_at_max_speed_nm = float(
            servo_config["torque_at_max_speed_nm"]
        )
        self._servo_brake_torque_limit_nm = float(servo_config["brake_torque_limit_nm"])
        self._servo_passive_active_ratio = float(servo_config["passive_active_ratio"])
        target_speed_limit = float(servo_config["training_target_speed_limit_rad_s"])
        # Commands are integer servo units. Using a whole-unit step keeps both
        # the vendor no-load speed ceiling and command quantization exact.
        self._max_command_step_units = max(
            1,
            int(
                np.floor(
                    target_speed_limit
                    * self.robot.control_period_s
                    / self._command_resolution_rad
                )
            ),
        )
        self._max_command_step_rad = (
            self._max_command_step_units * self._command_resolution_rad
        )
        if self.config.torque_exposure_time_constant_s <= 0.0:
            raise ValueError("Torque-exposure time constant must be positive")
        self._torque_exposure_decay = float(
            np.exp(
                -self.robot.control_period_s
                / self.config.torque_exposure_time_constant_s
            )
        )

        joint_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.robot.actuator_names
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            raise ValueError("At least one configured actuator joint is missing")
        self._joint_qpos_indices = jp.asarray(
            [int(mj_model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
        )
        self._joint_qvel_indices = jp.asarray(
            [int(mj_model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
        )
        zmp_config = self.config.zmp_reference
        self._zmp_reference = WR2ZMPReference(
            mj_model,
            keyframe_id=key_id,
            actuator_names=self.robot.actuator_names,
            command_forward_range_m_s=self.config.command_forward_range_m_s,
            gait_cycle_s=self.config.gait_cycle_s,
            swing_height_m=self.config.swing_height_m,
            com_height_m=zmp_config.com_height_m,
            single_double_support_ratio=zmp_config.single_double_support_ratio,
            phase_samples=zmp_config.phase_samples,
            command_samples=zmp_config.command_samples,
            ik_damping=zmp_config.ik_damping,
            ik_max_iterations=zmp_config.ik_max_iterations,
            max_position_residual_m=zmp_config.max_position_residual_m,
            max_orientation_residual_rad=(zmp_config.max_orientation_residual_rad),
        )

        active_mask = np.asarray(
            [motor.group in self.config.active_groups for motor in self.robot.motors],
            dtype=np.float32,
        )
        active_indices = np.flatnonzero(active_mask).astype(np.int32)
        if active_indices.size == 0:
            raise ValueError("environment.active_groups selects no actuators")
        self._active_mask = jp.asarray(active_mask)
        self._active_indices = jp.asarray(active_indices)
        self._action_size = int(active_indices.size)
        margin = self.robot.action.joint_limit_margin_rad
        safe_lower = self.robot.lower_limit_rad + margin
        safe_upper = self.robot.upper_limit_rad - margin
        if np.any(safe_lower >= safe_upper):
            raise ValueError("Action margin leaves an empty joint range")
        active_lower = safe_lower[active_indices]
        active_upper = safe_upper[active_indices]
        self._active_target_midpoint = jp.asarray(0.5 * (active_lower + active_upper))
        self._active_target_half_range = jp.asarray(0.5 * (active_upper - active_lower))
        full_home_action = self.robot.action.home_action(
            self.robot.home_position_rad,
            self.robot.lower_limit_rad,
            self.robot.upper_limit_rad,
        )
        self._home_action = jp.asarray(full_home_action[active_indices])
        pose_weights = np.asarray(self.config.pose_weights, dtype=np.float32)
        if pose_weights.shape != (self.robot.actuator_count,):
            raise ValueError(
                "environment.pose_weights must contain one value per WR2 actuator"
            )
        self._pose_weights = jp.asarray(pose_weights) * self._active_mask
        self._single_observation_size = self.robot.single_observation_size
        self._observation_history_frames = self.robot.observation_history_frames
        self._privileged_single_observation_size = (
            self._single_observation_size
            + self.robot.actuator_count
            + 3
            + self.robot.actuator_count
            + 2
            + 2
        )
        self._root_link_index = system.link_names.index("torso")
        self._foot_link_indices = jp.asarray(
            [
                system.link_names.index("left_foot"),
                system.link_names.index("right_foot"),
            ]
        )
        self._foot_geom_ids = jp.asarray(
            [
                mujoco.mj_name2id(
                    mj_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{side}_foot_collision",
                )
                for side in ("left", "right")
            ]
        )
        if bool(np.any(np.asarray(self._foot_geom_ids) < 0)):
            raise ValueError("MJX model is missing a configured foot collision geom")
        self._floor_geom_id = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        )
        if self._floor_geom_id < 0:
            raise ValueError("MJX scene is missing the floor geom")

    @property
    def action_size(self) -> int:
        """Policy controls only the configured active actuator groups."""
        return self._action_size

    @property
    def home_action(self) -> jax.Array:
        """Normalized active-joint action that exactly reproduces walk_home."""
        return self._home_action

    @property
    def zmp_reference_diagnostics(self):
        """Host-side checks for the generated privileged gait lookup."""
        return self._zmp_reference.diagnostics

    def _is_walking(self, command: jax.Array) -> jax.Array:
        return jp.linalg.norm(command) >= self.config.command_active_threshold_m_s

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        keys = jax.random.split(rng, 4)
        forward = jax.random.uniform(
            keys[0],
            (),
            minval=self.config.command_forward_range_m_s[0],
            maxval=self.config.command_forward_range_m_s[1],
        )
        lateral = jax.random.uniform(
            keys[1],
            (),
            minval=self.config.command_lateral_range_m_s[0],
            maxval=self.config.command_lateral_range_m_s[1],
        )
        yaw = jax.random.uniform(
            keys[2],
            (),
            minval=self.config.command_yaw_range_rad_s[0],
            maxval=self.config.command_yaw_range_rad_s[1],
        )
        command = jp.stack([forward, lateral, yaw])
        is_zero = jax.random.bernoulli(keys[3], self.config.zero_command_probability)
        return jp.where(is_zero, jp.zeros(3), command)

    def _kinematic_observation(self, pipeline_state):
        torso_quat = pipeline_state.x.rot[self._root_link_index]
        torso_quat = jp.where(torso_quat[0] < 0.0, -torso_quat, torso_quat)
        angular_velocity = _inverse_rotate_vector(
            torso_quat, pipeline_state.xd.ang[self._root_link_index]
        )
        linear_velocity = _inverse_rotate_vector(
            torso_quat, pipeline_state.xd.vel[self._root_link_index]
        )
        projected_gravity = _inverse_rotate_vector(
            torso_quat, jp.array([0.0, 0.0, -1.0])
        )
        return torso_quat, angular_velocity, linear_velocity, projected_gravity

    def _observation(
        self,
        pipeline_state,
        previous_action: jax.Array,
        command: jax.Array,
        gait_phase: jax.Array,
        rng: jax.Array,
        imu_state: dict[str, jax.Array],
        backlash: jax.Array,
        foot_contact: jax.Array,
        desired_contact: jax.Array,
        zmp_reference_active: jax.Array | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        joint_position = pipeline_state.q[self._joint_qpos_indices]
        joint_velocity = pipeline_state.qd[self._joint_qvel_indices]
        _, angular_velocity, linear_velocity, projected_gravity = (
            self._kinematic_observation(pipeline_state)
        )
        true_joint_position = joint_position
        true_joint_velocity = joint_velocity
        true_angular_velocity = angular_velocity
        true_projected_gravity = projected_gravity

        if self.add_observation_noise:
            keys = jax.random.split(rng, 8)
            noise = self.config.observation_noise
            # ToddlerBot models encoder backlash as a torque-direction-dependent
            # offset rather than independent white noise.
            joint_position = joint_position + 0.5 * backlash * jp.tanh(
                pipeline_state.actuator_force
                / jp.maximum(self._torque_exposure_reference_nm, 1e-6)
            )
            joint_position = (
                joint_position
                + noise.joint_position_std_rad
                * jax.random.normal(keys[0], joint_position.shape)
            )
            joint_velocity = (
                joint_velocity
                + noise.joint_velocity_std_rad_s
                * jax.random.normal(keys[1], joint_velocity.shape)
            )
            gyro_colored = noise.gyro_colored_alpha * imu_state[
                "gyro_colored"
            ] + jp.sqrt(
                1.0 - noise.gyro_colored_alpha**2
            ) * noise.gyro_colored_std_rad_s * jax.random.normal(
                keys[2], angular_velocity.shape
            )
            gyro_bias = imu_state[
                "gyro_bias"
            ] + noise.gyro_bias_walk_std_rad_s * jax.random.normal(
                keys[3], angular_velocity.shape
            )
            current_angular_velocity = (
                angular_velocity
                + gyro_colored
                + gyro_bias
                + noise.gyro_std_rad_s
                * jax.random.normal(keys[4], angular_velocity.shape)
            )
            gravity_colored = noise.projected_gravity_colored_alpha * imu_state[
                "gravity_colored"
            ] + jp.sqrt(
                1.0 - noise.projected_gravity_colored_alpha**2
            ) * noise.projected_gravity_colored_std * jax.random.normal(
                keys[5], projected_gravity.shape
            )
            gravity_bias = imu_state[
                "gravity_bias"
            ] + noise.projected_gravity_bias_walk_std * jax.random.normal(
                keys[6], projected_gravity.shape
            )
            current_projected_gravity = (
                projected_gravity
                + gravity_colored
                + gravity_bias
                + noise.projected_gravity_std
                * jax.random.normal(keys[7], projected_gravity.shape)
            )
            current_projected_gravity = current_projected_gravity / jp.linalg.norm(
                current_projected_gravity
            )
            angular_velocity = jp.where(
                imu_state["delay_one_step"],
                imu_state["previous_gyro"],
                current_angular_velocity,
            )
            projected_gravity = jp.where(
                imu_state["delay_one_step"],
                imu_state["previous_gravity"],
                current_projected_gravity,
            )
            next_imu_state = {
                **imu_state,
                "gyro_colored": gyro_colored,
                "gyro_bias": gyro_bias,
                "gravity_colored": gravity_colored,
                "gravity_bias": gravity_bias,
                "previous_gyro": current_angular_velocity,
                "previous_gravity": current_projected_gravity,
            }
        else:
            next_imu_state = imu_state

        phase_and_command = [
            jp.stack([jp.sin(gait_phase), jp.cos(gait_phase)]),
            command,
        ]
        actor = jp.concatenate(
            [
                *phase_and_command,
                joint_position - self._home_ctrl,
                self._joint_velocity_observation_scale * joint_velocity,
                previous_action,
                angular_velocity,
                projected_gravity,
            ]
        )
        if zmp_reference_active is None:
            zmp_reference_active = self._is_walking(command)
        zmp_joint_position = self._zmp_reference.joint_position(
            gait_phase,
            command,
            zmp_reference_active,
        )
        privileged = jp.concatenate(
            [
                *phase_and_command,
                true_joint_position - self._home_ctrl,
                self._joint_velocity_observation_scale * true_joint_velocity,
                previous_action,
                true_angular_velocity,
                true_projected_gravity,
                true_joint_position - zmp_joint_position,
                self._privileged_linear_velocity_scale * linear_velocity,
                self._privileged_actuator_force_scale * pipeline_state.actuator_force,
                foot_contact.astype(jp.float32),
                desired_contact.astype(jp.float32),
            ]
        )
        return {"state": actor, "privileged_state": privileged}, next_imu_state

    def _foot_contact(self, pipeline_state) -> jax.Array:
        # This MuJoCo release indexes contact metadata through a host array, so
        # use static contact slots (as ToddlerBot does) rather than a traced
        # vmap index.
        contact_force = jp.stack(
            [
                mjx_support.contact_force(self.sys, pipeline_state, index, True)
                for index in range(pipeline_state.contact.dist.shape[0])
            ]
        )
        active = support_contact_active(
            pipeline_state.contact.dist,
            contact_force[:, :3],
            self.config.contact_force_threshold_n,
        )
        return jp.stack(
            [
                jp.any(
                    active
                    & (
                        (
                            (pipeline_state.contact.geom1 == geom_id)
                            & (pipeline_state.contact.geom2 == self._floor_geom_id)
                        )
                        | (
                            (pipeline_state.contact.geom2 == geom_id)
                            & (pipeline_state.contact.geom1 == self._floor_geom_id)
                        )
                    )
                )
                for geom_id in self._foot_geom_ids
            ]
        )

    def _stack_observation(
        self, observation: jax.Array, history: jax.Array, frame_size: int
    ) -> jax.Array:
        """Insert the newest frame first, matching ToddlerBot's frame stack."""
        return jp.roll(history, frame_size).at[:frame_size].set(observation)

    def reset(self, rng: jax.Array) -> State:
        (
            rng,
            joint_rng,
            velocity_rng,
            command_rng,
            phase_rng,
            observation_rng,
            actuator_bias_rng,
            torso_roll_rng,
            torso_pitch_rng,
            backlash_rng,
            actuator_rng,
            imu_delay_rng,
        ) = jax.random.split(rng, 12)
        qpos = self._default_qpos.at[self._joint_qpos_indices].add(
            jax.random.uniform(
                joint_rng,
                (self.robot.actuator_count,),
                minval=-self.config.reset_joint_noise_rad,
                maxval=self.config.reset_joint_noise_rad,
            )
        )
        randomization = self.config.randomization
        if randomization.enabled:
            roll = jax.random.uniform(
                torso_roll_rng,
                (),
                minval=randomization.initial_torso_roll_rad[0],
                maxval=randomization.initial_torso_roll_rad[1],
            )
            pitch = jax.random.uniform(
                torso_pitch_rng,
                (),
                minval=randomization.initial_torso_pitch_rad[0],
                maxval=randomization.initial_torso_pitch_rad[1],
            )
            qpos = qpos.at[3:7].set(
                _quaternion_multiply(_roll_pitch_quaternion(roll, pitch), qpos[3:7])
            )
        qvel = jax.random.uniform(
            velocity_rng,
            (self.sys.nv,),
            minval=-self.config.reset_velocity_noise_rad_s,
            maxval=self.config.reset_velocity_noise_rad_s,
        )
        bias_range = randomization.target_bias_rad
        actuator_target_bias = (
            jax.random.uniform(
                actuator_bias_rng,
                (self.robot.actuator_count,),
                minval=bias_range[0],
                maxval=bias_range[1],
            )
            if randomization.enabled
            else jp.zeros(self.robot.actuator_count)
        )
        backlash = (
            jax.random.uniform(
                backlash_rng,
                (self.robot.actuator_count,),
                minval=randomization.backlash_rad[0],
                maxval=randomization.backlash_rad[1],
            )
            if randomization.enabled
            else jp.zeros(self.robot.actuator_count)
        )
        actuator_keys = jax.random.split(actuator_rng, 6)

        def actuator_scale(key, bounds):
            return (
                jax.random.uniform(
                    key,
                    (self.robot.actuator_count,),
                    minval=bounds[0],
                    maxval=bounds[1],
                )
                if randomization.enabled
                else jp.ones(self.robot.actuator_count)
            )

        actuator_noise = {
            "kp": actuator_scale(actuator_keys[0], randomization.kp_scale),
            "kv": actuator_scale(actuator_keys[1], randomization.kv_scale),
            "torque_limit": actuator_scale(
                actuator_keys[2], randomization.torque_limit_scale
            ),
            "max_speed": actuator_scale(
                actuator_keys[3], randomization.max_speed_scale
            ),
            "brake_torque": actuator_scale(
                actuator_keys[4], randomization.brake_torque_scale
            ),
            "passive_active_ratio": actuator_scale(
                actuator_keys[5], randomization.passive_active_ratio_scale
            ),
        }
        initial_target = jp.clip(
            self._quantize_target(self._home_ctrl + actuator_target_bias),
            self._ctrl_lower,
            self._ctrl_upper,
        )
        pipeline_state = self.pipeline_init(
            qpos, qvel, ctrl=jp.zeros(self.robot.actuator_count)
        )
        command = self._sample_command(command_rng)
        is_walking = self._is_walking(command)
        random_phase = jax.random.uniform(phase_rng, (), maxval=_TWO_PI)
        gait_phase = jp.where(
            is_walking & self.config.randomize_gait_phase_on_reset,
            random_phase,
            0.0,
        )
        previous_action = self._home_action
        foot_contact = self._foot_contact(pipeline_state)
        desired_contact = expected_foot_contacts(
            gait_phase,
            is_walking,
            self.config.zmp_reference.single_double_support_ratio,
        )
        _, initial_gyro, _, initial_gravity = self._kinematic_observation(
            pipeline_state
        )
        imu_state = {
            "gyro_colored": jp.zeros(3),
            "gyro_bias": jp.zeros(3),
            "gravity_colored": jp.zeros(3),
            "gravity_bias": jp.zeros(3),
            "previous_gyro": initial_gyro,
            "previous_gravity": initial_gravity,
            "delay_one_step": (
                jax.random.bernoulli(
                    imu_delay_rng,
                    self.config.observation_noise.imu_one_step_delay_probability,
                )
                if self.add_observation_noise
                else jp.asarray(False)
            ),
        }
        observation_frames, imu_state = self._observation(
            pipeline_state,
            previous_action,
            command,
            gait_phase,
            observation_rng,
            imu_state,
            backlash,
            foot_contact,
            desired_contact,
            zmp_reference_active=jp.asarray(False),
        )
        observation = {
            "state": self._stack_observation(
                observation_frames["state"],
                jp.zeros(
                    self._single_observation_size * self._observation_history_frames
                ),
                self._single_observation_size,
            ),
            "privileged_state": self._stack_observation(
                observation_frames["privileged_state"],
                jp.zeros(
                    self._privileged_single_observation_size
                    * self._observation_history_frames
                ),
                self._privileged_single_observation_size,
            ),
        }
        zero = jp.zeros(())
        metrics = {
            "reward": zero,
            "reward_per_step": zero,
            "velocity_xy": zero,
            "velocity_tracking_per_step": zero,
            "yaw_rate": zero,
            "yaw_tracking_per_step": zero,
            "angular_velocity_xy": zero,
            "upright": zero,
            "upright_per_step": zero,
            "torso_height": zero,
            "height_tracking_per_step": zero,
            "pose": zero,
            "action_rate": zero,
            "joint_velocity": zero,
            "foot_slip": zero,
            "mechanical_power": zero,
            "actuator_torque_squared": zero,
            "feet_phase": zero,
            "contact_phase": zero,
            "both_feet_contact": zero,
            "close_feet": zero,
            "feet_orientation": zero,
            "actuator_torque_rms_nm_per_step": zero,
            "actuator_torque_peak_nm_per_step": zero,
            "sustained_torque_exposure_per_step": zero,
            "command_forward_m_s_per_step": zero,
            "forward_velocity_m_s_per_step": zero,
            "forward_velocity_error_m_s_per_step": zero,
            "lateral_velocity_m_s_per_step": zero,
            "yaw_rate_rad_s_per_step": zero,
            "yaw_rate_error_rad_s_per_step": zero,
            "torso_height_m_per_step": zero,
            "torso_tilt_rad_per_step": zero,
            "action_abs_mean_per_step": zero,
            "action_max_abs_per_step": zero,
            "action_deviation_from_home_abs_mean_per_step": zero,
            "action_saturation_fraction_per_step": zero,
            "action_near_boundary_fraction_per_step": zero,
            "target_excursion_from_home_rad_per_step": zero,
            "target_excursion_over_tb_range_fraction_per_step": zero,
            "target_clipping_fraction_per_step": zero,
            "target_slew_limiting_fraction_per_step": zero,
            "left_foot_contact_per_step": zero,
            "right_foot_contact_per_step": zero,
            "double_support_per_step": zero,
            "feet_phase_tracking_per_step": zero,
            "contact_phase_match_per_step": zero,
            "left_foot_height_m_per_step": zero,
            "right_foot_height_m_per_step": zero,
            "left_foot_target_height_m_per_step": zero,
            "right_foot_target_height_m_per_step": zero,
            "feet_lateral_distance_m_per_step": zero,
            "fall": zero,
            "nonfinite_state": zero,
            "left_foot_contact": zero,
            "right_foot_contact": zero,
        }
        info = {
            "rng": rng,
            "command": command,
            "gait_phase": gait_phase,
            "reset_command": command,
            "reset_gait_phase": gait_phase,
            "actuator_target_bias": actuator_target_bias,
            "actuator_noise": actuator_noise,
            "backlash": backlash,
            "imu_state": imu_state,
            "reset_imu_state": imu_state,
            "torque_exposure": jp.zeros(self.robot.actuator_count),
            "previous_action": previous_action,
            "previous_target": initial_target,
            "reset_target": initial_target,
            "step_count": jp.zeros((), dtype=jp.int32),
        }
        return State(pipeline_state, observation, zero, zero, metrics, info)

    def _controller_torque(
        self,
        pipeline_state,
        target: jax.Array,
        actuator_noise: dict[str, jax.Array],
    ) -> jax.Array:
        """Apply ToddlerBot's PD plus asymmetric torque-speed envelope."""
        joint_position = pipeline_state.q[self._joint_qpos_indices]
        joint_velocity = pipeline_state.qd[self._joint_qvel_indices]
        joint_acceleration = pipeline_state.qacc[self._joint_qvel_indices]
        error = target - joint_position
        kp = self._servo_kp * actuator_noise["kp"]
        kv = self._servo_kv * actuator_noise["kv"]
        passive_ratio = (
            self._servo_passive_active_ratio * actuator_noise["passive_active_ratio"]
        )
        effective_kp = jp.where(
            joint_acceleration * error < 0.0, kp * passive_ratio, kp
        )
        raw_torque = effective_kp * error - kv * joint_velocity

        torque_limit = self._servo_torque_limit_nm * actuator_noise["torque_limit"]
        max_speed = self._servo_max_speed_rad_s * actuator_noise["max_speed"]
        peak_torque_speed = jp.minimum(
            self._servo_peak_torque_speed_rad_s, max_speed - 1e-6
        )
        speed = jp.abs(joint_velocity)
        taper_fraction = jp.clip(
            (speed - peak_torque_speed)
            / jp.maximum(max_speed - peak_torque_speed, 1e-6),
            0.0,
            1.0,
        )
        acceleration_limit = torque_limit + taper_fraction * (
            self._servo_torque_at_max_speed_nm - torque_limit
        )
        acceleration_limit = jp.where(
            speed <= peak_torque_speed, torque_limit, acceleration_limit
        )
        acceleration_limit = jp.maximum(acceleration_limit, 0.0)
        brake_limit = self._servo_brake_torque_limit_nm * actuator_noise["brake_torque"]
        bounded = jp.where(
            joint_velocity > 0.0,
            jp.clip(raw_torque, -brake_limit, acceleration_limit),
            jp.clip(raw_torque, -acceleration_limit, brake_limit),
        )
        overspeed_accelerating = (speed > max_speed) & (joint_velocity * error > 0.0)
        braking = jp.where(joint_velocity > 0.0, -brake_limit, brake_limit)
        return jp.where(overspeed_accelerating, braking, bounded)

    def _pipeline_step_target(
        self,
        pipeline_state,
        target: jax.Array,
        actuator_noise: dict[str, jax.Array],
    ):
        """Hold a position target while integrating ten 2 ms torque steps."""

        def integrate(current, _):
            torque = self._controller_torque(current, target, actuator_noise)
            return self._pipeline.step(self.sys, current, torque, self._debug), None

        return jax.lax.scan(integrate, pipeline_state, (), self._n_frames)[0]

    def step(self, state: State, action: jax.Array) -> State:
        rng, observation_rng, command_rng = jax.random.split(state.info["rng"], 3)
        action = jp.clip(action, -1.0, 1.0)
        if self.config.action_delay_steps == 1:
            applied_action = state.info["previous_action"]
        elif self.config.action_delay_steps == 0:
            applied_action = action
        else:
            raise ValueError("The initial environment supports only 0 or 1 delay steps")

        applied_active_target = (
            self._active_target_midpoint
            + self._active_target_half_range * applied_action
        )
        policy_target = self._home_ctrl.at[self._active_indices].set(
            applied_active_target
        )
        unconstrained_target = policy_target + state.info["actuator_target_bias"]
        desired_target = jp.clip(
            unconstrained_target, self._ctrl_lower, self._ctrl_upper
        )
        quantized_unclipped = self._quantize_target(desired_target)
        quantized_target = jp.clip(
            quantized_unclipped, self._ctrl_lower, self._ctrl_upper
        )
        previous_target = state.info["previous_target"]
        target_delta = quantized_target - previous_target
        limited_delta = jp.clip(
            target_delta,
            -self._max_command_step_rad,
            self._max_command_step_rad,
        )
        target = previous_target + limited_delta
        target = jp.clip(target, self._ctrl_lower, self._ctrl_upper)
        pipeline_state = self._pipeline_step_target(
            state.pipeline_state, target, state.info["actuator_noise"]
        )
        torso_quat, angular_velocity, linear_velocity, projected_gravity = (
            self._kinematic_observation(pipeline_state)
        )
        joint_position = pipeline_state.q[self._joint_qpos_indices]
        joint_velocity = pipeline_state.qd[self._joint_qvel_indices]
        command = state.info["command"]
        is_walking = self._is_walking(command)
        gait_phase = jp.where(
            is_walking,
            jp.mod(state.info["gait_phase"] + self._gait_phase_increment, _TWO_PI),
            0.0,
        )

        velocity_xy = jp.exp(
            -jp.sum(jp.square(linear_velocity[:2] - command[:2]))
            / self.config.velocity_reward_sigma**2
        )
        yaw_rate = jp.exp(
            -jp.square(angular_velocity[2] - command[2])
            / self.config.yaw_tracking_sigma**2
        )
        angular_velocity_xy_cost = jp.sum(jp.square(angular_velocity[:2]))
        torso_tilt_rad = jp.arccos(jp.clip(-projected_gravity[2], -1.0, 1.0))
        upright = jp.exp(-20.0 * jp.square(torso_tilt_rad))
        torso_height = jp.exp(
            -jp.square(
                pipeline_state.x.pos[self._root_link_index, 2]
                - self.config.target_torso_height_m
            )
            / self.config.height_tracking_sigma**2
        )
        pose = jp.sum(jp.square(joint_position - self._home_ctrl) * self._pose_weights)
        requested_active_target = (
            self._active_target_midpoint + self._active_target_half_range * action
        )
        action_rate = normalized_action_rate_cost(
            action,
            state.info["previous_action"],
        )
        joint_velocity_cost = jp.sum(jp.square(joint_velocity))
        foot_contact = self._foot_contact(pipeline_state)
        desired_foot_height = expected_foot_heights(
            gait_phase,
            self.config.swing_height_m,
            is_walking,
        )
        foot_height = (
            pipeline_state.site_xpos[self._foot_site_ids, 2] - self._home_foot_site_z
        )
        foot_height_error_squared = jp.sum(jp.square(foot_height - desired_foot_height))
        feet_phase = jp.exp(
            -foot_height_error_squared / self.config.feet_phase_tracking_sigma_m2
        ) * (1.0 + jp.max(desired_foot_height) / self.config.swing_height_m)
        desired_contact = expected_foot_contacts(
            gait_phase,
            is_walking,
            self.config.zmp_reference.single_double_support_ratio,
        )
        contact_phase = jp.mean((foot_contact == desired_contact).astype(jp.float32))
        both_feet_contact = is_walking.astype(jp.float32) * jp.all(foot_contact).astype(
            jp.float32
        )

        foot_delta_world = (
            pipeline_state.site_xpos[self._foot_site_ids[1]]
            - pipeline_state.site_xpos[self._foot_site_ids[0]]
        )
        feet_lateral_distance_m = feet_lateral_distance(torso_quat, foot_delta_world)
        close_feet = (
            feet_lateral_distance_m < self.config.min_feet_lateral_distance_m
        ).astype(jp.float32)
        foot_projected_gravity = jax.vmap(_inverse_rotate_vector)(
            pipeline_state.x.rot[self._foot_link_indices],
            jp.broadcast_to(jp.array([0.0, 0.0, -1.0]), (2, 3)),
        )
        feet_orientation = feet_orientation_error(
            foot_projected_gravity,
            self._home_foot_projected_gravity,
        )
        foot_velocity_xy = pipeline_state.xd.vel[self._foot_link_indices, :2]
        foot_slip = jp.sum(
            foot_contact.astype(foot_velocity_xy.dtype)
            * jp.sum(jp.square(foot_velocity_xy), axis=1)
        )
        mechanical_power = jp.sum(
            jp.abs(pipeline_state.actuator_force * joint_velocity)
        )
        actuator_torque_squared = jp.sum(jp.square(pipeline_state.actuator_force))
        actuator_torque_rms_nm = jp.sqrt(
            jp.mean(jp.square(pipeline_state.actuator_force))
        )
        actuator_torque_peak_nm = jp.max(jp.abs(pipeline_state.actuator_force))
        normalized_torque_squared = jp.square(
            pipeline_state.actuator_force / self._torque_exposure_reference_nm
        )
        torque_exposure = (
            self._torque_exposure_decay * state.info["torque_exposure"]
            + (1.0 - self._torque_exposure_decay) * normalized_torque_squared
        )
        sustained_torque_exposure = jp.max(torque_exposure)

        weights = self.config.rewards
        reward_rate = (
            weights.velocity_xy * velocity_xy
            + weights.yaw_rate * yaw_rate
            + weights.angular_velocity_xy * angular_velocity_xy_cost
            + weights.upright * upright
            + weights.torso_height * torso_height
            + weights.alive
            + weights.pose * pose
            + weights.action_rate * action_rate
            + weights.joint_velocity * joint_velocity_cost
            + weights.foot_slip * foot_slip
            + weights.mechanical_power * mechanical_power
            + weights.actuator_torque_squared * actuator_torque_squared
            + weights.feet_phase * feet_phase
            + weights.contact_phase * contact_phase
            + weights.both_feet_contact * both_feet_contact
            + weights.close_feet * close_feet
            + weights.feet_orientation * feet_orientation
        )
        # ToddlerBot treats configured weights as reward rates and integrates
        # them over the control period before passing the result to PPO.
        reward = reward_rate * self.robot.control_period_s

        step_count = state.info["step_count"] + 1
        torso_z = pipeline_state.x.pos[self._root_link_index, 2]
        unhealthy = (torso_z < self.config.terminate_height_m) | (
            projected_gravity[2] > self.config.terminate_projected_gravity_z
        )
        nonfinite = ~jp.all(jp.isfinite(pipeline_state.q))
        timeout = step_count >= self.config.episode_length
        done = (unhealthy | nonfinite | timeout).astype(jp.float32)
        active_count = float(self.action_size)
        action_abs_mean = jp.mean(jp.abs(action))
        action_max_abs = jp.max(jp.abs(action))
        action_deviation_from_home_abs_mean = jp.mean(
            jp.abs(action - self._home_action)
        )
        policy_saturation = jp.abs(action) >= 0.99
        policy_near_boundary = jp.abs(action) >= 0.90
        target_excursion = jp.abs(
            requested_active_target - self._home_ctrl[self._active_indices]
        )
        target_excursion_mean = jp.mean(target_excursion)
        target_excursion_over_tb_range = target_excursion > 0.25
        target_clipping = (
            jp.abs(
                unconstrained_target[self._active_indices]
                - desired_target[self._active_indices]
            )
            > 1e-7
        ) | (
            jp.abs(
                quantized_unclipped[self._active_indices]
                - quantized_target[self._active_indices]
            )
            > 1e-7
        )
        target_slew_limiting = (
            jp.abs(
                target_delta[self._active_indices] - limited_delta[self._active_indices]
            )
            > 1e-7
        )
        target_clipping_fraction = jp.sum(target_clipping) / active_count
        target_slew_limiting_fraction = jp.sum(target_slew_limiting) / active_count
        action_saturation_fraction = (
            jp.sum(policy_saturation | target_clipping) / active_count
        )
        action_near_boundary_fraction = (
            jp.sum(policy_near_boundary | target_clipping) / active_count
        )
        target_excursion_over_tb_range_fraction = (
            jp.sum(target_excursion_over_tb_range) / active_count
        )

        sampled_command = self._sample_command(command_rng)
        resample_command = jp.mod(step_count, self.config.command_resample_steps) == 0
        next_command = jp.where(resample_command, sampled_command, command)
        next_is_walking = self._is_walking(next_command)
        next_gait_phase = jp.where(next_is_walking, gait_phase, 0.0)
        next_gait_phase = jp.where(
            resample_command & next_is_walking & ~is_walking,
            0.0,
            next_gait_phase,
        )
        next_desired_contact = expected_foot_contacts(
            next_gait_phase,
            next_is_walking,
            self.config.zmp_reference.single_double_support_ratio,
        )
        observation_frames, imu_state = self._observation(
            pipeline_state,
            action,
            next_command,
            next_gait_phase,
            observation_rng,
            state.info["imu_state"],
            state.info["backlash"],
            foot_contact,
            next_desired_contact,
        )
        observation = {
            "state": self._stack_observation(
                observation_frames["state"],
                state.obs["state"],
                self._single_observation_size,
            ),
            "privileged_state": self._stack_observation(
                observation_frames["privileged_state"],
                state.obs["privileged_state"],
                self._privileged_single_observation_size,
            ),
        }
        reset_episode = done.astype(jp.bool_)

        def reset_if_done(current, reset):
            return jp.where(reset_episode, reset, current)

        reset_imu_state = jax.tree.map(
            reset_if_done,
            imu_state,
            state.info["reset_imu_state"],
        )
        info = {
            **state.info,
            "rng": rng,
            # Brax AutoReset restores only the initial pipeline state and
            # observation. Reset WR2's recurrent episode state here so a
            # timeout/fall cannot become a permanent one-step episode.
            "command": reset_if_done(next_command, state.info["reset_command"]),
            "gait_phase": reset_if_done(
                next_gait_phase, state.info["reset_gait_phase"]
            ),
            "previous_action": reset_if_done(action, self._home_action),
            "previous_target": reset_if_done(target, state.info["reset_target"]),
            "imu_state": reset_imu_state,
            "torque_exposure": reset_if_done(
                torque_exposure, jp.zeros_like(torque_exposure)
            ),
            "step_count": reset_if_done(step_count, jp.zeros_like(step_count)),
        }
        metrics = {
            "reward": reward,
            "reward_per_step": reward,
            "velocity_xy": velocity_xy,
            "velocity_tracking_per_step": velocity_xy,
            "yaw_rate": yaw_rate,
            "yaw_tracking_per_step": yaw_rate,
            "angular_velocity_xy": -angular_velocity_xy_cost,
            "upright": upright,
            "upright_per_step": upright,
            "torso_height": torso_height,
            "height_tracking_per_step": torso_height,
            "pose": -pose,
            "action_rate": -action_rate,
            "joint_velocity": -joint_velocity_cost,
            "foot_slip": -foot_slip,
            "mechanical_power": -mechanical_power,
            "actuator_torque_squared": -actuator_torque_squared,
            "feet_phase": feet_phase,
            "contact_phase": contact_phase,
            "both_feet_contact": -both_feet_contact,
            "close_feet": -close_feet,
            "feet_orientation": -feet_orientation,
            "actuator_torque_rms_nm_per_step": actuator_torque_rms_nm,
            "actuator_torque_peak_nm_per_step": actuator_torque_peak_nm,
            "sustained_torque_exposure_per_step": sustained_torque_exposure,
            "command_forward_m_s_per_step": command[0],
            "forward_velocity_m_s_per_step": linear_velocity[0],
            "forward_velocity_error_m_s_per_step": jp.abs(
                linear_velocity[0] - command[0]
            ),
            "lateral_velocity_m_s_per_step": linear_velocity[1],
            "yaw_rate_rad_s_per_step": angular_velocity[2],
            "yaw_rate_error_rad_s_per_step": jp.abs(angular_velocity[2] - command[2]),
            "torso_height_m_per_step": torso_z,
            "torso_tilt_rad_per_step": torso_tilt_rad,
            "action_abs_mean_per_step": action_abs_mean,
            "action_max_abs_per_step": action_max_abs,
            "action_deviation_from_home_abs_mean_per_step": (
                action_deviation_from_home_abs_mean
            ),
            "action_saturation_fraction_per_step": action_saturation_fraction,
            "action_near_boundary_fraction_per_step": (action_near_boundary_fraction),
            "target_excursion_from_home_rad_per_step": target_excursion_mean,
            "target_excursion_over_tb_range_fraction_per_step": (
                target_excursion_over_tb_range_fraction
            ),
            "target_clipping_fraction_per_step": target_clipping_fraction,
            "target_slew_limiting_fraction_per_step": target_slew_limiting_fraction,
            "left_foot_contact_per_step": foot_contact[0].astype(jp.float32),
            "right_foot_contact_per_step": foot_contact[1].astype(jp.float32),
            "double_support_per_step": both_feet_contact,
            "feet_phase_tracking_per_step": feet_phase,
            "contact_phase_match_per_step": contact_phase,
            "left_foot_height_m_per_step": foot_height[0],
            "right_foot_height_m_per_step": foot_height[1],
            "left_foot_target_height_m_per_step": desired_foot_height[0],
            "right_foot_target_height_m_per_step": desired_foot_height[1],
            "feet_lateral_distance_m_per_step": feet_lateral_distance_m,
            "fall": unhealthy.astype(jp.float32),
            "nonfinite_state": nonfinite.astype(jp.float32),
            "left_foot_contact": foot_contact[0].astype(jp.float32),
            "right_foot_contact": foot_contact[1].astype(jp.float32),
        }
        return state.replace(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _quantize_target(self, target: jax.Array) -> jax.Array:
        return (
            jp.round(target / self._command_resolution_rad)
            * self._command_resolution_rad
        )
