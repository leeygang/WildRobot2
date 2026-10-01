"""Phase-guided Brax/MJX walking environment for WR2."""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from wr2.locomotion.configs import WalkingEnvConfig, load_training_config
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
    """Return left/right ground-relative swing targets for one gait cycle."""
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
    gait_phase: jax.Array, is_walking: jax.Array
) -> jax.Array:
    """Return desired left/right stance contact for one gait cycle."""
    left_stance = jp.mod(gait_phase, _TWO_PI) >= np.pi
    contacts = jp.stack([left_stance, ~left_stance])
    return jp.where(is_walking, contacts, jp.ones(2, dtype=jp.bool_))


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
        if not np.isclose(self.config.action_scale_rad, self.robot.action.scale_rad):
            raise ValueError(
                "Walking action scale must match the versioned robot contract"
            )
        self._joint_velocity_observation_scale = float(
            self.robot.config["observation"]["scales"]["joint_velocity"]
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
        self._home_ctrl = jp.asarray(mj_model.key_ctrl[key_id])
        self._ctrl_lower = jp.asarray(mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_upper = jp.asarray(mj_model.actuator_ctrlrange[:, 1])
        self._foot_site_ids = jp.asarray(foot_site_ids)
        self._home_foot_site_z = jp.asarray(home_data.site_xpos[foot_site_ids, 2])
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

        active_mask = np.asarray(
            [motor.group in self.config.active_groups for motor in self.robot.motors],
            dtype=np.float32,
        )
        self._active_mask = jp.asarray(active_mask)
        pose_weights = np.asarray(self.config.pose_weights, dtype=np.float32)
        if pose_weights.shape != (self.robot.actuator_count,):
            raise ValueError(
                "environment.pose_weights must contain one value per WR2 actuator"
            )
        self._pose_weights = jp.asarray(pose_weights) * self._active_mask
        self._single_observation_size = self.robot.single_observation_size
        self._observation_history_frames = self.robot.observation_history_frames
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
    ) -> jax.Array:
        joint_position = pipeline_state.q[self._joint_qpos_indices]
        joint_velocity = pipeline_state.qd[self._joint_qvel_indices]
        _, angular_velocity, _, projected_gravity = self._kinematic_observation(
            pipeline_state
        )

        if self.add_observation_noise:
            keys = jax.random.split(rng, 4)
            noise = self.config.observation_noise
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
            angular_velocity = (
                angular_velocity
                + noise.gyro_std_rad_s
                * jax.random.normal(keys[2], angular_velocity.shape)
            )
            projected_gravity = (
                projected_gravity
                + noise.projected_gravity_std
                * jax.random.normal(keys[3], projected_gravity.shape)
            )
            projected_gravity = projected_gravity / jp.linalg.norm(projected_gravity)

        return jp.concatenate(
            [
                jp.stack([jp.sin(gait_phase), jp.cos(gait_phase)]),
                command,
                joint_position - self._home_ctrl,
                self._joint_velocity_observation_scale * joint_velocity,
                previous_action,
                angular_velocity,
                projected_gravity,
            ]
        )

    def _foot_contact(self, pipeline_state) -> jax.Array:
        active = pipeline_state.contact.dist < 0.0
        return jp.stack(
            [
                jp.any(
                    active
                    & (
                        (pipeline_state.contact.geom1 == geom_id)
                        | (pipeline_state.contact.geom2 == geom_id)
                    )
                )
                for geom_id in self._foot_geom_ids
            ]
        )

    def _stack_observation(
        self, observation: jax.Array, history: jax.Array
    ) -> jax.Array:
        """Insert the newest frame first, matching ToddlerBot's frame stack."""
        return jp.roll(history, self._single_observation_size).at[
            : self._single_observation_size
        ].set(observation)

    def reset(self, rng: jax.Array) -> State:
        (
            rng,
            joint_rng,
            velocity_rng,
            command_rng,
            phase_rng,
            observation_rng,
            actuator_bias_rng,
        ) = jax.random.split(rng, 7)
        qpos = self._default_qpos.at[self._joint_qpos_indices].add(
            jax.random.uniform(
                joint_rng,
                (self.robot.actuator_count,),
                minval=-self.config.reset_joint_noise_rad,
                maxval=self.config.reset_joint_noise_rad,
            )
        )
        qvel = jax.random.uniform(
            velocity_rng,
            (self.sys.nv,),
            minval=-self.config.reset_velocity_noise_rad_s,
            maxval=self.config.reset_velocity_noise_rad_s,
        )
        bias_range = self.config.randomization.target_bias_rad
        actuator_target_bias = (
            jax.random.uniform(
                actuator_bias_rng,
                (self.robot.actuator_count,),
                minval=bias_range[0],
                maxval=bias_range[1],
            )
            if self.config.randomization.enabled
            else jp.zeros(self.robot.actuator_count)
        )
        initial_target = jp.clip(
            self._quantize_target(self._home_ctrl + actuator_target_bias),
            self._ctrl_lower,
            self._ctrl_upper,
        )
        pipeline_state = self.pipeline_init(qpos, qvel, ctrl=initial_target)
        command = self._sample_command(command_rng)
        is_walking = self._is_walking(command)
        random_phase = jax.random.uniform(phase_rng, (), maxval=_TWO_PI)
        gait_phase = jp.where(
            is_walking & self.config.randomize_gait_phase_on_reset,
            random_phase,
            0.0,
        )
        previous_action = jp.zeros(self.robot.actuator_count)
        observation_frame = self._observation(
            pipeline_state,
            previous_action,
            command,
            gait_phase,
            observation_rng,
        )
        observation = self._stack_observation(
            observation_frame, jp.zeros(self.robot.observation_size)
        )
        zero = jp.zeros(())
        metrics = {
            "reward": zero,
            "reward_per_step": zero,
            "velocity_xy": zero,
            "velocity_tracking_per_step": zero,
            "yaw_rate": zero,
            "yaw_tracking_per_step": zero,
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
            "feet_distance": zero,
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
            "action_saturation_fraction_per_step": zero,
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
            "actuator_target_bias": actuator_target_bias,
            "torque_exposure": jp.zeros(self.robot.actuator_count),
            "previous_action": previous_action,
            "previous_target": initial_target,
            "step_count": jp.zeros((), dtype=jp.int32),
        }
        return State(pipeline_state, observation, zero, zero, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        rng, observation_rng, command_rng = jax.random.split(state.info["rng"], 3)
        action = jp.clip(action, -1.0, 1.0) * self._active_mask
        if self.config.action_delay_steps == 1:
            applied_action = state.info["previous_action"]
        elif self.config.action_delay_steps == 0:
            applied_action = action
        else:
            raise ValueError("The initial environment supports only 0 or 1 delay steps")

        desired_target = jp.clip(
            self._home_ctrl
            + state.info["actuator_target_bias"]
            + self.config.action_scale_rad * applied_action,
            self._ctrl_lower,
            self._ctrl_upper,
        )
        quantized_target = self._quantize_target(desired_target)
        previous_target = state.info["previous_target"]
        target = previous_target + jp.clip(
            quantized_target - previous_target,
            -self._max_command_step_rad,
            self._max_command_step_rad,
        )
        target = jp.clip(target, self._ctrl_lower, self._ctrl_upper)
        pipeline_state = self.pipeline_step(state.pipeline_state, target)
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
            / self.config.velocity_tracking_sigma**2
        )
        yaw_rate = jp.exp(
            -jp.square(angular_velocity[2] - command[2])
            / self.config.yaw_tracking_sigma**2
        )
        upright = jp.exp(-10.0 * jp.sum(jp.square(projected_gravity[:2])))
        torso_height = jp.exp(
            -jp.square(
                pipeline_state.x.pos[self._root_link_index, 2]
                - self.config.target_torso_height_m
            )
            / self.config.height_tracking_sigma**2
        )
        pose = jp.sum(
            jp.square(joint_position - self._home_ctrl) * self._pose_weights
        )
        action_rate = jp.sum(jp.square(action - state.info["previous_action"]))
        joint_velocity_cost = jp.sum(jp.square(joint_velocity))
        foot_contact = self._foot_contact(pipeline_state)
        desired_foot_height = expected_foot_heights(
            gait_phase, self.config.swing_height_m, is_walking
        )
        foot_height = (
            pipeline_state.site_xpos[self._foot_site_ids, 2]
            - self._home_foot_site_z
        )
        foot_height_error_squared = jp.sum(
            jp.square(foot_height - desired_foot_height)
        )
        feet_phase = jp.exp(
            -foot_height_error_squared
            / self.config.feet_phase_tracking_sigma_m2
        ) * (
            1.0
            + jp.max(desired_foot_height) / self.config.swing_height_m
        )
        desired_contact = expected_foot_contacts(gait_phase, is_walking)
        contact_phase = jp.mean((foot_contact == desired_contact).astype(jp.float32))
        both_feet_contact = is_walking.astype(jp.float32) * jp.all(
            foot_contact
        ).astype(jp.float32)

        foot_delta_world = (
            pipeline_state.site_xpos[self._foot_site_ids[1]]
            - pipeline_state.site_xpos[self._foot_site_ids[0]]
        )
        foot_delta_torso = _inverse_rotate_vector(torso_quat, foot_delta_world)
        feet_lateral_distance = jp.abs(foot_delta_torso[1])
        feet_distance_error = jp.maximum(
            self.config.min_feet_lateral_distance_m - feet_lateral_distance, 0.0
        ) + jp.maximum(
            feet_lateral_distance - self.config.max_feet_lateral_distance_m, 0.0
        )
        feet_distance = jp.exp(-jp.square(feet_distance_error / 0.02))
        foot_projected_gravity = jax.vmap(_inverse_rotate_vector)(
            pipeline_state.x.rot[self._foot_link_indices],
            jp.broadcast_to(jp.array([0.0, 0.0, -1.0]), (2, 3)),
        )
        feet_orientation = jp.mean(
            jp.exp(-5.0 * jp.sum(jp.square(foot_projected_gravity[:, :2]), axis=1))
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
            + weights.feet_distance * feet_distance
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
        active_count = jp.maximum(jp.sum(self._active_mask), 1.0)
        action_abs_mean = jp.sum(jp.abs(action) * self._active_mask) / active_count
        action_saturation_fraction = (
            jp.sum((jp.abs(action) >= 0.99).astype(action.dtype) * self._active_mask)
            / active_count
        )
        torso_tilt_rad = jp.arccos(jp.clip(-projected_gravity[2], -1.0, 1.0))

        sampled_command = self._sample_command(command_rng)
        resample_command = (
            jp.mod(step_count, self.config.command_resample_steps) == 0
        )
        next_command = jp.where(resample_command, sampled_command, command)
        next_is_walking = self._is_walking(next_command)
        next_gait_phase = jp.where(next_is_walking, gait_phase, 0.0)
        next_gait_phase = jp.where(
            resample_command & next_is_walking & ~is_walking,
            0.0,
            next_gait_phase,
        )
        observation_frame = self._observation(
            pipeline_state,
            action,
            next_command,
            next_gait_phase,
            observation_rng,
        )
        observation = self._stack_observation(observation_frame, state.obs)
        info = {
            **state.info,
            "rng": rng,
            "command": next_command,
            "gait_phase": next_gait_phase,
            "previous_action": action,
            "previous_target": target,
            "torque_exposure": torque_exposure,
            "step_count": step_count,
        }
        metrics = {
            "reward": reward,
            "reward_per_step": reward,
            "velocity_xy": velocity_xy,
            "velocity_tracking_per_step": velocity_xy,
            "yaw_rate": yaw_rate,
            "yaw_tracking_per_step": yaw_rate,
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
            "feet_distance": feet_distance,
            "feet_orientation": feet_orientation,
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
            "action_saturation_fraction_per_step": action_saturation_fraction,
            "left_foot_contact_per_step": foot_contact[0].astype(jp.float32),
            "right_foot_contact_per_step": foot_contact[1].astype(jp.float32),
            "double_support_per_step": both_feet_contact,
            "feet_phase_tracking_per_step": feet_phase,
            "contact_phase_match_per_step": contact_phase,
            "left_foot_height_m_per_step": foot_height[0],
            "right_foot_height_m_per_step": foot_height[1],
            "left_foot_target_height_m_per_step": desired_foot_height[0],
            "right_foot_target_height_m_per_step": desired_foot_height[1],
            "feet_lateral_distance_m_per_step": feet_lateral_distance,
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
