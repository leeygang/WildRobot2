import unittest

import numpy as np

from wr2.locomotion.config import DynamicsRandomization
from wr2.sensing.imu import canonicalize_sensor_sample
from wr2.sim import RobotDescription, RobotObservation, build_wr2_proprio_v1


class TrainingInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RobotDescription.load(include_local_calibration=False)

    def test_robot_description_matches_canonical_model(self):
        self.assertEqual(self.robot.actuator_count, 17)
        self.assertEqual(self.robot.actuator_names[0], "waist_yaw_drive")
        self.assertEqual(self.robot.observation_size, 60)
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
        self.assertEqual(servo["model_status"], "provisional_wr1_low_load_transfer")
        self.assertEqual(
            set(servo["fitted_parameters"]),
            {"kp_sim", "damping", "frictionloss"},
        )
        self.assertEqual(servo["maximum_validated_load_nm"], 0.56)
        self.assertEqual(servo["rated_voltage_v"], 11.1)
        self.assertEqual(servo["operating_voltage_range_v"], [9.6, 12.6])
        self.assertAlmostEqual(servo["vendor_stall_torque_nm"], 4.4129925)
        self.assertEqual(servo["vendor_stall_current_a"], 3.0)
        self.assertAlmostEqual(servo["vendor_no_load_speed_rad_s"], 5.8177642)
        self.assertLessEqual(
            servo["torque_limit_nm"], servo["vendor_stall_torque_nm"]
        )
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
        randomization = DynamicsRandomization()
        configured_ranges = servo["initial_training_randomization"]
        for name, configured_range in configured_ranges.items():
            self.assertEqual(tuple(configured_range), getattr(randomization, name))

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
        actor_observation = build_wr2_proprio_v1(
            observation,
            home_position_rad=self.robot.home_position_rad,
            previous_action=np.zeros(17),
            command_velocity=np.zeros(3),
        )
        self.assertEqual(actor_observation.shape, (60,))
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
