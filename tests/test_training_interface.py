import unittest

import numpy as np

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
        np.testing.assert_array_equal(self.robot.home_position_rad, np.zeros(17))

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
