import tempfile
import unittest
from pathlib import Path

import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.train import _create_run_directory
from wr2.locomotion.walking_metrics import walking_score
from wr2.sensing.imu import canonicalize_sensor_sample
from wr2.sim import RobotDescription, RobotObservation, build_wr2_proprio_v2


class TrainingInterfaceTest(unittest.TestCase):
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
        self.assertEqual(self.robot.observation_size, 62)
        self.assertEqual(self.robot.config["observation"]["layout_id"], "wr2_proprio_v2")
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

        self.assertLess(
            self.training_config.environment.rewards.actuator_torque_squared, 0.0
        )
        self.assertGreater(
            self.training_config.environment.torque_exposure_time_constant_s, 0.0
        )
        environment = self.training_config.environment
        self.assertEqual(environment.command_forward_range_m_s, (0.1, 0.25))
        self.assertEqual(environment.command_resample_steps, 150)
        self.assertEqual(environment.gait_cycle_s, 0.72)
        self.assertEqual(environment.swing_height_m, 0.03)
        self.assertLess(environment.velocity_tracking_sigma, 0.1)
        self.assertGreater(environment.rewards.feet_phase, 0.0)
        self.assertLess(environment.rewards.both_feet_contact, 0.0)
        self.assertEqual(self.training_config.ppo.num_timesteps, 1_000_000_000)
        self.assertEqual(
            self.training_config.ppo.evaluation_forward_command_m_s, 0.20
        )
        self.assertEqual(self.training_config.ppo.learning_rate, 3e-5)
        self.assertEqual(self.training_config.ppo.clipping_epsilon, 0.2)
        self.assertEqual(
            self.training_config.network.policy_hidden_layer_sizes,
            (512, 256, 128),
        )
        self.assertEqual(self.training_config.network.distribution_type, "normal")
        self.assertTrue(self.training_config.checkpoints.save_every_evaluation)
        self.assertTrue(self.training_config.checkpoints.keep_best)
        self.assertEqual(self.training_config.output.root, "results/wr2_walking")

    def test_zero_action_maps_to_home(self):
        targets = self.robot.action.targets(
            np.zeros(17),
            self.robot.home_position_rad,
            self.robot.lower_limit_rad,
            self.robot.upper_limit_rad,
        )
        np.testing.assert_array_equal(targets, self.robot.home_position_rad)

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

    def test_identity_pose_observation_layout(self):
        observation = RobotObservation(
            time_s=0.0,
            joint_position_rad=np.zeros(17, dtype=np.float32),
            joint_velocity_rad_s=np.zeros(17, dtype=np.float32),
            torso_to_world_quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity_torso_rad_s=np.zeros(3, dtype=np.float32),
        )
        actor_observation = build_wr2_proprio_v2(
            observation,
            gait_phase_rad=np.pi / 2.0,
            home_position_rad=self.robot.home_position_rad,
            previous_action=np.zeros(17),
            command_velocity=np.zeros(3),
        )
        self.assertEqual(actor_observation.shape, (62,))
        np.testing.assert_allclose(actor_observation[:2], [1, 0], atol=1e-6)
        np.testing.assert_allclose(actor_observation[-3:], [0, 0, -1])

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
