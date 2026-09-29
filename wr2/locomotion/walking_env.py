"""Minimal Brax/MJX standing-to-walking environment for WR2."""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from wr2.locomotion.config import WalkingEnvConfig
from wr2.sim.robot import RobotDescription


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
        self.config = config or WalkingEnvConfig()
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

        system = mjcf.load_model(mj_model)
        n_frames = round(self.robot.control_period_s / self.robot.simulation_timestep_s)
        super().__init__(system, backend="mjx", n_frames=n_frames)

        self._default_qpos = jp.asarray(mj_model.key_qpos[key_id])
        self._home_ctrl = jp.asarray(mj_model.key_ctrl[key_id])
        self._ctrl_lower = jp.asarray(mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_upper = jp.asarray(mj_model.actuator_ctrlrange[:, 1])

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

    def reset(self, rng: jax.Array) -> State:
        rng, joint_rng, velocity_rng, command_rng, observation_rng = jax.random.split(
            rng, 5
        )
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
        pipeline_state = self.pipeline_init(qpos, qvel, ctrl=self._home_ctrl)
        command = self._sample_command(command_rng)
        previous_action = jp.zeros(self.robot.actuator_count)
        observation = self._observation(
            pipeline_state, previous_action, command, observation_rng
        )
        zero = jp.zeros(())
        metrics = {
            "reward": zero,
            "velocity_xy": zero,
            "yaw_rate": zero,
            "upright": zero,
            "torso_height": zero,
            "pose": zero,
            "action_rate": zero,
            "joint_velocity": zero,
            "foot_slip": zero,
            "mechanical_power": zero,
            "left_foot_contact": zero,
            "right_foot_contact": zero,
        }
        info = {
            "rng": rng,
            "command": command,
            "previous_action": previous_action,
            "step_count": jp.zeros((), dtype=jp.int32),
        }
        return State(pipeline_state, observation, zero, zero, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        rng, observation_rng = jax.random.split(state.info["rng"])
        action = jp.clip(action, -1.0, 1.0) * self._active_mask
        if self.config.action_delay_steps == 1:
            applied_action = state.info["previous_action"]
        elif self.config.action_delay_steps == 0:
            applied_action = action
        else:
            raise ValueError("The initial environment supports only 0 or 1 delay steps")

        target = jp.clip(
            self._home_ctrl + self.config.action_scale_rad * applied_action,
            self._ctrl_lower,
            self._ctrl_upper,
        )
        pipeline_state = self.pipeline_step(state.pipeline_state, target)
        _, angular_velocity, linear_velocity, projected_gravity = (
            self._kinematic_observation(pipeline_state)
        )
        joint_position = pipeline_state.q[self._joint_qpos_indices]
        joint_velocity = pipeline_state.qd[self._joint_qvel_indices]
        command = state.info["command"]

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
        pose = jp.sum(jp.square(joint_position - self._home_ctrl))
        action_rate = jp.sum(jp.square(action - state.info["previous_action"]))
        joint_velocity_cost = jp.sum(jp.square(joint_velocity))
        foot_contact = self._foot_contact(pipeline_state)
        foot_velocity_xy = pipeline_state.xd.vel[self._foot_link_indices, :2]
        foot_slip = jp.sum(
            foot_contact.astype(foot_velocity_xy.dtype)
            * jp.sum(jp.square(foot_velocity_xy), axis=1)
        )
        mechanical_power = jp.sum(
            jp.abs(pipeline_state.actuator_force * joint_velocity)
        )

        weights = self.config.rewards
        reward = (
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
        )

        step_count = state.info["step_count"] + 1
        torso_z = pipeline_state.x.pos[self._root_link_index, 2]
        unhealthy = (torso_z < self.config.terminate_height_m) | (
            projected_gravity[2] > self.config.terminate_projected_gravity_z
        )
        nonfinite = ~jp.all(jp.isfinite(pipeline_state.q))
        timeout = step_count >= self.config.episode_length
        done = (unhealthy | nonfinite | timeout).astype(jp.float32)

        observation = self._observation(
            pipeline_state, action, command, observation_rng
        )
        info = {
            **state.info,
            "rng": rng,
            "previous_action": action,
            "step_count": step_count,
        }
        metrics = {
            "reward": reward,
            "velocity_xy": velocity_xy,
            "yaw_rate": yaw_rate,
            "upright": upright,
            "torso_height": torso_height,
            "pose": -pose,
            "action_rate": -action_rate,
            "joint_velocity": -joint_velocity_cost,
            "foot_slip": -foot_slip,
            "mechanical_power": -mechanical_power,
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
