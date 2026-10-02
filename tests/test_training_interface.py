import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.envs import training as env_training

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.ppo import make_network_factory
from wr2.locomotion.train import (
    _acquisition_checkpoint_score,
    _create_run_directory,
)
from wr2.locomotion.walking_env import (
    WR2WalkingEnv,
    feet_lateral_distance,
    feet_orientation_error,
    normalized_action_rate_cost,
    support_contact_active,
)
from wr2.locomotion.walking_metrics import walking_score
from wr2.sensing.imu import canonicalize_sensor_sample
from wr2.sim import (
    RobotDescription,
    RobotObservation,
    build_wr2_proprio_v3,
    update_wr2_observation_history,
)


class TrainingInterfaceTest(unittest.TestCase):
    def test_acquisition_checkpoint_ranking_prefers_safe_motion_over_standing(self):
        common = {
            "episode_length": 1000.0,
            "target_episode_length": 1000,
            "fall_rate": 0.0,
            "command_forward_m_s": 0.10,
            "action_saturation": 0.0,
            "nonfinite_state": 0.0,
            "mean_step_peak_torque_nm": 1.0,
        }
        standing = _acquisition_checkpoint_score(
            **common,
            forward_velocity_m_s=0.0,
            contact_match=0.5,
            double_support=1.0,
        )
        walking = _acquisition_checkpoint_score(
            **common,
            forward_velocity_m_s=0.08,
            contact_match=0.8,
            double_support=0.3,
        )
        unsafe = _acquisition_checkpoint_score(
            **{**common, "action_saturation": 0.16},
            forward_velocity_m_s=0.10,
            contact_match=0.9,
            double_support=0.2,
        )
        self.assertGreater(walking, standing)
        self.assertEqual(unsafe, -math.inf)

    def test_walking_score_rejects_stationary_double_support(self):
        walking = walking_score(
            episode_length=1000,
            target_episode_length=1000,
            velocity_error_m_s=0.01,
            velocity_sigma_m_s=0.08,
            contact_match=0.95,
            double_support=0.1,
        )
        standing = walking_score(
            episode_length=1000,
            target_episode_length=1000,
            velocity_error_m_s=0.15,
            velocity_sigma_m_s=0.08,
            contact_match=0.5,
            double_support=1.0,
        )
        self.assertGreater(walking, 50.0 * standing)

    def test_training_run_directory_is_generated_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            first_id, first_path = _create_run_directory(output_root, None, seed=3)
            second_id, second_path = _create_run_directory(output_root, None, seed=3)

            self.assertRegex(first_id, r"^wr2_ppo_\d{8}_\d{6}_seed3(?:_\d{2})?$")
            self.assertTrue(first_path.is_dir())
            self.assertTrue(second_path.is_dir())
            self.assertNotEqual(first_id, second_id)

    def test_explicit_training_run_id_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "single directory name"):
                _create_run_directory(Path(temporary_directory), "../outside", seed=0)

    @classmethod
    def setUpClass(cls):
        cls.robot = RobotDescription.load(include_local_calibration=False)
        cls.training_config = load_training_config()

    def test_robot_description_matches_canonical_model(self):
        self.assertEqual(self.robot.actuator_count, 17)
        self.assertEqual(self.robot.actuator_names[0], "waist_yaw_drive")
        self.assertEqual(self.robot.single_observation_size, 55)
        self.assertEqual(self.robot.observation_history_frames, 15)
        self.assertEqual(self.robot.observation_size, 825)
        self.assertEqual(
            self.robot.config["observation"]["layout_id"], "wr2_proprio_v3"
        )
        self.assertEqual(self.robot.control_period_s, 0.02)
        self.assertEqual(self.robot.simulation_timestep_s, 0.002)
        np.testing.assert_array_equal(self.robot.neutral_position_rad, np.zeros(17))
        expected_walk_home = np.zeros(17)
        for name, value in {
            "left_hip_pitch": -0.12,
            "left_knee_pitch": -0.24,
            "left_ankle_pitch": -0.12,
            "right_hip_pitch": 0.12,
            "right_knee_pitch": 0.24,
            "right_ankle_pitch": 0.12,
        }.items():
            expected_walk_home[self.robot.actuator_names.index(name)] = value
        np.testing.assert_allclose(
            self.robot.home_position_rad, expected_walk_home, atol=1e-7
        )
        servo = self.robot.config["actuators"]["htd45hServo"]
        self.assertEqual(
            servo["model_status"], "provisional_wr2_unit_a_static_envelope"
        )
        self.assertEqual(
            set(servo["wr1_fitted_parameters"]),
            {"kp_sim", "damping", "frictionloss"},
        )
        self.assertEqual(servo["wr2_overridden_parameters"], ["kp_sim"])
        self.assertEqual(servo["kp_sim"], 16.0)
        self.assertEqual(servo["maximum_validated_load_nm"], 1.0157)
        self.assertEqual(servo["maximum_validated_load_duration_s"], 3.0)
        self.assertEqual(servo["thermal_short_hold_repeat_count"], 2)
        self.assertEqual(servo["thermal_short_hold_duration_s"], 180.0)
        self.assertEqual(servo["thermal_short_hold_peak_temperature_c"], 72.0)
        self.assertEqual(servo["thermal_short_hold_quantized_target_deg"], 7.92)
        self.assertEqual(servo["thermal_short_hold_apparent_stiffness_nm_per_rad"], 9.5)
        self.assertEqual(
            servo["thermal_short_hold_status"],
            "repeatable_pass_not_at_equilibrium",
        )
        self.assertEqual(servo["software_temperature_shutdown_c"], 80.0)
        self.assertEqual(servo["rated_voltage_v"], 11.1)
        self.assertEqual(servo["operating_voltage_range_v"], [9.6, 12.6])
        self.assertAlmostEqual(servo["vendor_stall_torque_nm"], 4.4129925)
        self.assertEqual(servo["vendor_stall_current_a"], 3.0)
        self.assertAlmostEqual(servo["vendor_no_load_speed_rad_s"], 5.8177642)
        self.assertEqual(servo["peak_torque_speed_rad_s"], 0.0)
        self.assertEqual(servo["torque_at_max_speed_nm"], 0.0)
        self.assertEqual(servo["brake_torque_limit_nm"], 4.0)
        self.assertLessEqual(servo["torque_limit_nm"], servo["vendor_stall_torque_nm"])
        resolution = servo["command_resolution_rad"]
        maximum_step_units = int(
            np.floor(
                servo["training_target_speed_limit_rad_s"]
                * self.robot.control_period_s
                / resolution
            )
        )
        self.assertEqual(maximum_step_units, 27)
        self.assertLessEqual(
            maximum_step_units * resolution / self.robot.control_period_s,
            servo["vendor_no_load_speed_rad_s"],
        )
        randomization = self.training_config.environment.randomization
        self.assertEqual(
            tuple(value * servo["kp_sim"] for value in randomization.kp_scale),
            (8.0, 32.0),
        )
        self.assertEqual(
            tuple(
                value * servo["torque_limit_nm"]
                for value in randomization.torque_limit_scale
            ),
            (1.0, 4.0),
        )
        self.assertAlmostEqual(randomization.target_bias_rad[0], np.deg2rad(-2.0))
        self.assertAlmostEqual(randomization.target_bias_rad[1], np.deg2rad(2.0))
        self.assertEqual(randomization.body_mass_scale, (0.8, 1.2))
        self.assertAlmostEqual(randomization.initial_torso_roll_rad[0], -0.1)
        self.assertAlmostEqual(randomization.backlash_rad[1], np.deg2rad(2.0))

        self.assertLess(
            self.training_config.environment.rewards.actuator_torque_squared, 0.0
        )
        self.assertGreater(
            self.training_config.environment.torque_exposure_time_constant_s, 0.0
        )
        environment = self.training_config.environment
        self.assertEqual(environment.command_forward_range_m_s, (0.08, 0.18))
        self.assertFalse(environment.randomization.enabled)
        self.assertEqual(environment.command_resample_steps, 150)
        self.assertEqual(environment.gait_cycle_s, 0.72)
        self.assertEqual(environment.privileged_linear_velocity_scale, 2.0)
        self.assertEqual(environment.privileged_actuator_force_scale, 0.1)
        self.assertEqual(environment.swing_height_m, 0.04)
        self.assertEqual(environment.feet_phase_tracking_sigma_m2, 0.0007)
        self.assertEqual(environment.velocity_reward_sigma, 0.15)
        self.assertAlmostEqual(environment.velocity_tracking_sigma**-2, 1000.0)
        self.assertEqual(environment.rewards.velocity_xy, 2.0)
        self.assertEqual(environment.rewards.feet_phase, 7.5)
        self.assertEqual(environment.rewards.action_rate, -2.0)
        self.assertEqual(environment.rewards.pose, -0.5)
        self.assertEqual(len(environment.pose_weights), self.robot.actuator_count)
        # Match ToddlerBot's active no-ZMP-imitation reward set: contact match
        # is an evaluation metric, while close feet and foot tilt are costs.
        self.assertEqual(environment.rewards.contact_phase, 0.0)
        self.assertEqual(environment.rewards.both_feet_contact, 0.0)
        self.assertEqual(environment.rewards.angular_velocity_xy, -1.0)
        self.assertEqual(environment.rewards.close_feet, -10.0)
        self.assertEqual(environment.rewards.feet_orientation, -5.0)
        self.assertEqual(self.training_config.ppo.num_timesteps, 1_000_000_000)
        self.assertEqual(self.training_config.ppo.evaluation_forward_command_m_s, 0.10)
        self.assertEqual(self.training_config.ppo.learning_rate, 3e-5)
        self.assertEqual(self.training_config.ppo.clipping_epsilon, 0.2)
        self.assertEqual(
            self.training_config.network.policy_hidden_layer_sizes,
            (512, 256, 128),
        )
        self.assertEqual(self.training_config.network.distribution_type, "tanh_normal")
        self.assertEqual(self.training_config.network.init_noise_std, 0.13)
        self.assertFalse(self.training_config.ppo.normalize_observations)
        self.assertEqual(self.training_config.ppo.training_metrics_steps, 2_000_000)
        self.assertTrue(self.training_config.checkpoints.save_every_evaluation)
        self.assertTrue(self.training_config.checkpoints.keep_best)
        self.assertEqual(
            self.training_config.checkpoints.selection_metric, "acquisition"
        )
        self.assertEqual(self.training_config.output.root, "results/wr2_walking")

    def test_action_rate_matches_tb_normalized_policy_action_semantics(self):
        environment = WR2WalkingEnv(
            replace(
                self.training_config.environment,
                reset_joint_noise_rad=0.0,
                reset_velocity_noise_rad_s=0.0,
            ),
            add_observation_noise=False,
        )
        previous_action = environment.home_action
        action_delta = jp.asarray(
            [0.10, -0.05, 0.02, -0.03, 0.04, -0.08, 0.06, -0.01, 0.07, -0.02]
        )
        action = previous_action + action_delta
        expected = float(jp.sum(jp.square(action_delta)))

        self.assertAlmostEqual(
            float(normalized_action_rate_cost(action, previous_action)),
            expected,
            places=7,
        )
        state = environment.reset(jax.random.PRNGKey(23))
        next_state = jax.jit(environment.step)(state, action)
        jax.block_until_ready(next_state.reward)
        self.assertAlmostEqual(
            -float(next_state.metrics["action_rate"]),
            expected,
            places=6,
        )

        legacy_radian_cost = float(
            jp.sum(
                jp.square(environment._active_target_half_range * action_delta / 0.25)
            )
        )
        self.assertGreater(legacy_radian_cost, 5.0 * expected)

    def test_privileged_critic_uses_tb_feature_scales(self):
        environment = WR2WalkingEnv(
            replace(
                self.training_config.environment,
                reset_joint_noise_rad=0.0,
                reset_velocity_noise_rad_s=0.0,
            ),
            add_observation_noise=False,
        )
        state = environment.reset(jax.random.PRNGKey(29))
        privileged_frame = np.asarray(state.obs["privileged_state"][:96])
        _, _, linear_velocity, _ = environment._kinematic_observation(
            state.pipeline_state
        )

        np.testing.assert_allclose(
            privileged_frame[72:75],
            2.0 * np.asarray(linear_velocity),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            privileged_frame[75:92],
            0.1 * np.asarray(state.pipeline_state.actuator_force),
            atol=1e-7,
        )

    def test_contact_and_foot_width_match_tb_world_frame_semantics(self):
        active = support_contact_active(
            jp.asarray([-0.01, -0.01, 0.01]),
            jp.asarray(
                [
                    [20.0, 0.0, 0.5],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 10.0],
                ]
            ),
            1.0,
        )
        np.testing.assert_array_equal(np.asarray(active), [False, True, False])

        yaw_90 = jp.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        delta_world = jp.asarray([0.10, 0.0, 0.20])
        self.assertAlmostEqual(
            float(feet_lateral_distance(yaw_90, delta_world)),
            0.10,
            places=6,
        )

    def test_walk_home_foot_orientation_is_the_zero_tilt_reference(self):
        environment = WR2WalkingEnv(
            self.training_config.environment,
            add_observation_noise=False,
        )
        home_gravity = environment._home_foot_projected_gravity
        home_cost = float(feet_orientation_error(home_gravity, home_gravity))
        legacy_local_z_cost = float(jp.sum(jp.linalg.norm(home_gravity[:, :2], axis=1)))

        self.assertLess(home_cost, 3e-6)
        self.assertGreater(legacy_local_z_cost, 1.9)

    def test_training_wrapper_resets_wr2_episode_state_after_timeout(self):
        environment_config = replace(
            self.training_config.environment,
            episode_length=5,
            reset_joint_noise_rad=0.0,
            reset_velocity_noise_rad_s=0.0,
        )
        environment = WR2WalkingEnv(environment_config, add_observation_noise=False)
        wrapped = env_training.wrap(
            environment,
            episode_length=environment_config.episode_length,
            action_repeat=1,
        )
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(7), 2))
        commanded_action = environment.home_action + 0.001
        action = jp.broadcast_to(commanded_action, (2, environment.action_size))
        step = jax.jit(wrapped.step)

        done_history = []
        step_count_history = []
        completed_lengths = []
        completed_previous_actions = []
        for _ in range(12):
            state = step(state, action)
            jax.block_until_ready(state.done)
            done = np.asarray(state.done)
            done_history.append(done.copy())
            step_count_history.append(np.asarray(state.info["step_count"]).copy())
            if np.any(done):
                completed_lengths.extend(
                    np.asarray(state.info["episode_metrics"]["length"])[done > 0]
                )
                completed_previous_actions.extend(
                    np.asarray(state.info["previous_action"])[done > 0]
                )

        np.testing.assert_array_equal(
            np.asarray(done_history)[:, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        )
        np.testing.assert_array_equal(
            np.asarray(step_count_history)[:, 0],
            [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],
        )
        np.testing.assert_array_equal(completed_lengths, [5, 5, 5, 5])
        np.testing.assert_allclose(
            completed_previous_actions,
            np.broadcast_to(np.asarray(environment.home_action), (4, 10)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(state.info["previous_action"]),
            np.broadcast_to(np.asarray(commanded_action), (2, 10)),
            atol=1e-7,
        )

    def test_acquisition_velocity_reward_has_signal_without_weakening_score(self):
        command_error_m_s = 0.20
        reward_at_rest = np.exp(
            -(command_error_m_s**2)
            / self.training_config.environment.velocity_reward_sigma**2
        )
        strict_score_at_rest = np.exp(
            -(command_error_m_s**2)
            / self.training_config.environment.velocity_tracking_sigma**2
        )

        self.assertGreater(reward_at_rest, 0.1)
        self.assertLess(strict_score_at_rest, 1e-12)

    def test_policy_distribution_enforces_normalized_action_bounds(self):
        active = self.robot.active_actuator_indices(("leg",))
        full_home_action = self.robot.action.home_action(
            self.robot.home_position_rad,
            self.robot.lower_limit_rad,
            self.robot.upper_limit_rad,
        )
        home_action = full_home_action[active]
        networks = make_network_factory(
            self.training_config.network,
            home_action=home_action,
        )(
            {"state": self.robot.observation_size, "privileged_state": 1440},
            10,
        )
        distribution = networks.parametric_action_distribution
        self.assertEqual(distribution.param_size, 20)

        logits = jp.concatenate(
            [
                jp.linspace(-20.0, 20.0, 10),
                jp.zeros(10),
            ]
        )
        action = np.asarray(distribution.mode(logits))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

        parameters = networks.policy_network.init(jax.random.PRNGKey(0))
        initial_logits = networks.policy_network.apply(
            None,
            parameters,
            {
                "state": jp.zeros((1, self.robot.observation_size)),
                "privileged_state": jp.zeros((1, 1440)),
            },
        )
        np.testing.assert_allclose(
            np.asarray(distribution.mode(initial_logits))[0],
            home_action,
            atol=1e-6,
        )
        initial_distribution = distribution.create_dist(initial_logits)
        np.testing.assert_allclose(
            np.asarray(initial_distribution.scale)[0],
            np.full(10, self.training_config.network.init_noise_std),
            atol=1e-6,
        )

    def test_zero_action_maps_to_joint_range_midpoint(self):
        targets = self.robot.action.targets(
            np.zeros(17),
            self.robot.home_position_rad,
            self.robot.lower_limit_rad,
            self.robot.upper_limit_rad,
        )
        np.testing.assert_allclose(
            targets,
            0.5 * (self.robot.lower_limit_rad + self.robot.upper_limit_rad),
            atol=1e-7,
        )

    def test_nonzero_home_action_maps_exactly_to_walk_home(self):
        home_action = self.robot.action.home_action(
            self.robot.home_position_rad,
            self.robot.lower_limit_rad,
            self.robot.upper_limit_rad,
        )
        targets = self.robot.action.targets(
            home_action,
            self.robot.home_position_rad,
            self.robot.lower_limit_rad,
            self.robot.upper_limit_rad,
        )
        self.assertGreater(np.max(np.abs(home_action)), 0.7)
        np.testing.assert_allclose(targets, self.robot.home_position_rad, atol=1e-7)

    def test_action_is_clipped_and_respects_joint_limits(self):
        targets = self.robot.action.targets(
            np.full(17, 10.0),
            self.robot.home_position_rad,
            self.robot.lower_limit_rad,
            self.robot.upper_limit_rad,
        )
        margin = self.robot.action.joint_limit_margin_rad
        self.assertTrue(np.all(targets <= self.robot.upper_limit_rad - margin))
        self.assertTrue(np.all(targets >= self.robot.lower_limit_rad + margin))

    def test_active_action_expands_without_hidden_joint_limit_clipping(self):
        active = self.robot.active_actuator_indices(("leg",))
        self.assertEqual(active.size, 10)
        for value in (-1.0, 1.0):
            targets = self.robot.action.active_targets(
                np.full(active.size, value),
                active_indices=active,
                home_position_rad=self.robot.home_position_rad,
                lower_limit_rad=self.robot.lower_limit_rad,
                upper_limit_rad=self.robot.upper_limit_rad,
            )
            self.assertTrue(np.all(targets <= self.robot.upper_limit_rad))
            self.assertTrue(np.all(targets >= self.robot.lower_limit_rad))
            inactive = np.setdiff1d(np.arange(self.robot.actuator_count), active)
            np.testing.assert_array_equal(
                targets[inactive], self.robot.home_position_rad[inactive]
            )
            expected = (
                self.robot.lower_limit_rad[active]
                if value < 0.0
                else self.robot.upper_limit_rad[active]
            )
            np.testing.assert_allclose(targets[active], expected, atol=1e-7)

    def test_action_contract_can_reach_a_four_centimeter_swing_pose(self):
        active = self.robot.active_actuator_indices(("leg",))
        target = self.robot.home_position_rad.copy()
        target[self.robot.actuator_names.index("left_hip_pitch")] -= 0.787
        target[self.robot.actuator_names.index("left_knee_pitch")] -= 0.811
        midpoint = 0.5 * (
            self.robot.lower_limit_rad[active] + self.robot.upper_limit_rad[active]
        )
        half_range = 0.5 * (
            self.robot.upper_limit_rad[active] - self.robot.lower_limit_rad[active]
        )
        action = (target[active] - midpoint) / half_range
        self.assertLess(np.max(np.abs(action)), 1.0)
        mapped = self.robot.action.active_targets(
            action,
            active_indices=active,
            home_position_rad=self.robot.home_position_rad,
            lower_limit_rad=self.robot.lower_limit_rad,
            upper_limit_rad=self.robot.upper_limit_rad,
        )
        np.testing.assert_allclose(mapped[active], target[active], atol=1e-6)

    def test_identity_pose_observation_layout(self):
        observation = RobotObservation(
            time_s=0.0,
            joint_position_rad=np.zeros(17, dtype=np.float32),
            joint_velocity_rad_s=np.zeros(17, dtype=np.float32),
            torso_to_world_quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity_torso_rad_s=np.zeros(3, dtype=np.float32),
        )
        active = self.robot.active_actuator_indices(("leg",))
        actor_observation = build_wr2_proprio_v3(
            observation,
            gait_phase_rad=np.pi / 2.0,
            home_position_rad=self.robot.home_position_rad,
            previous_active_action=np.zeros(active.size),
            active_indices=active,
            command_velocity=np.zeros(3),
        )
        self.assertEqual(actor_observation.shape, (55,))
        np.testing.assert_allclose(actor_observation[:2], [1, 0], atol=1e-6)
        np.testing.assert_allclose(actor_observation[-3:], [0, 0, -1])

        history = update_wr2_observation_history(
            actor_observation,
            history_frames=self.robot.observation_history_frames,
        )
        self.assertEqual(history.shape, (self.robot.observation_size,))
        np.testing.assert_array_equal(history[:55], actor_observation)
        np.testing.assert_array_equal(history[55:], np.zeros(55 * 14))

        next_frame = actor_observation + 1.0
        history = update_wr2_observation_history(
            next_frame,
            history_frames=self.robot.observation_history_frames,
            previous_history=history,
        )
        np.testing.assert_array_equal(history[:55], next_frame)
        np.testing.assert_array_equal(history[55:110], actor_observation)

    def test_imu_mounting_rotation_is_removed(self):
        mount = np.asarray(self.robot.imu_sensor_to_torso_quat_wxyz)
        sample = canonicalize_sensor_sample(
            sensor_to_world_quat_wxyz=mount,
            angular_velocity_sensor_rad_s=[1.0, 0.0, 0.0],
            sensor_to_torso_quat_wxyz=mount,
        )
        np.testing.assert_allclose(
            sample.torso_to_world_quat_wxyz, [1, 0, 0, 0], atol=1e-6
        )
        np.testing.assert_allclose(
            sample.projected_gravity_torso, [0, 0, -1], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
