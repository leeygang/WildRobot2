from types import SimpleNamespace
import unittest

import numpy as np

from wr2.tools.validate_hip_action import coverage, hip_center, summarize


def fixture():
    # Scrambled actuator order exercises name-based motor indices, not qpos
    # order or an assumption that hip-pitch coordinates are adjacent.
    return SimpleNamespace(
        robot=SimpleNamespace(actuator_names=['knee', 'right_hip_pitch', 'left_hip_pitch', 'ankle'],
                              home_position_rad=np.array([-.24, .12, -.12, .12])),
        _home_ctrl=np.array([-.70, .29, -.29, .41]),
        _ctrl_lower=np.full(4, -2.), _ctrl_upper=np.full(4, 2.),
        _command_resolution_rad=.01,
        config=SimpleNamespace(normalization_scale=.8, policy_action_scale_rad=.25,
                               randomization=SimpleNamespace(target_bias_rad=(-.02, .02))),
        _zmp_reference=SimpleNamespace(_host={
            'commands': np.array([[.16, 0., 0.]]),
            'motors': np.array([[[0., -.13, .13, 0.], [0., -.12, .12, 0.], [0., -.10, .10, 0.]]]),
            'stance': np.ones((1, 3, 2), bool),
        }),
    )


class HipActionValidationTest(unittest.TestCase):
    def test_only_named_hip_centers_change_and_grounded_pose_is_representable(self):
        env = fixture()
        old_home = env._home_ctrl.copy()
        center, hips = hip_center(env, True)
        np.testing.assert_array_equal(hips, [2, 1])
        np.testing.assert_array_equal(center, [-.70, .12, -.12, .41])
        np.testing.assert_array_equal(env._home_ctrl, old_home)
        action = (old_home-center)/.25
        np.testing.assert_allclose(action[hips], [-.68, .68])
        np.testing.assert_allclose(center+.25*action, old_home)

    def test_posterior_coverage_handles_opposite_joint_signs_and_bias_intersection(self):
        env = fixture()
        center, hips = hip_center(env, True)
        rows = coverage(env, center, hips)
        self.assertEqual([r['joint'] for r in rows], ['left_hip_pitch', 'right_hip_pitch'])
        for row in rows:
            self.assertEqual(row['posterior_stance_outside_fraction'], 0.)
            self.assertAlmostEqual(row['stance_outside_with_worst_bias_fraction'], 2/3)
        old, hips = hip_center(env, False)
        self.assertTrue(all(r['posterior_stance_outside_fraction'] == 1. for r in coverage(env, old, hips)))

    def test_summary_keeps_terminal_sample_and_excludes_padding(self):
        length = 100
        valid = np.arange(length)[:, None] < 80
        pose = np.tile([0., 0., .3, 1., 0., 0., 0.], (length, 1, 1))
        pose[:, 0, 0] = np.arange(1, length+1)*.002
        pose[80:] = 999.
        trace = {
            'valid': valid, 'fall': np.arange(length)[:, None] == 79,
            'nonfinite': np.zeros((length, 1), bool),
            'initial_qpos': np.array([[0., 0., .3, 1., 0., 0., 0.]]),
            'qpos': pose, 'root_delta_m': np.tile([.002, 0.], (length, 1, 1)),
            'command_m_s': np.tile([.10, 0., 0.], (length, 1, 1)),
            'contact': np.ones((length, 1, 2), bool),
            'self_contact': np.zeros((length, 1)),
            'floor_force_n': np.tile([10., 11.], (length, 1, 1)),
            'target_rad': np.zeros((length, 1, 2)),
            'motor_rad': np.zeros((length, 1, 2)),
            'torque_nm': np.full((length, 1, 2), .5),
            'target_error_rad': np.zeros((length, 1)),
            'joint_violation_rad': np.zeros((length, 1)),
        }
        trace['torque_nm'][79] = .6
        trace['torque_nm'][80:] = 999.
        row = summarize(trace, .02, 1.)[0]
        self.assertEqual(row['valid_steps'], 80)
        self.assertTrue(row['fell'])
        self.assertFalse(row['completed'])
        self.assertAlmostEqual(row['forward_speed_TB_m_s'], .10)
        self.assertEqual(row['maximum_final_substep_torque_nm'], .6)
        self.assertEqual(row['root_z90_span_TB_m'], 0.)
        self.assertEqual(row['walking_steps'], 5)
        self.assertAlmostEqual(row['walking_forward_speed_TB_m_s'], .10)


if __name__ == '__main__':
    unittest.main()
