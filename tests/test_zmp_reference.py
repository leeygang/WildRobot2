import unittest

import jax.numpy as jp
import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.train import _validate_training_zmp_reference
from wr2.locomotion.walking_env import expected_foot_contacts, expected_foot_heights
from wr2.reference.walk_zmp import periodic_foot_reference
from wr2.reference.zmp_validation import (
    create_reference_context,
    validate_zmp_reference,
)


class ZMPReferenceValidationTest(unittest.TestCase):
    COMMANDS_M_S = (0.10, 0.15, 0.20, 0.25)

    @classmethod
    def setUpClass(cls):
        cls.context = create_reference_context(cls.COMMANDS_M_S)

    def test_target_command_geometry_passes_reference_gate(self):
        for command in self.COMMANDS_M_S:
            with self.subTest(command=command):
                result = validate_zmp_reference(
                    self.context,
                    command,
                    samples=72,
                )
                self.assertTrue(result.passed, msg="\n".join(result.failures))
                self.assertLessEqual(result.worst_stance_bottom_z_m, 0.003)
                self.assertGreaterEqual(result.worst_swing_bottom_z_m, -0.002)
                self.assertGreaterEqual(result.minimum_zmp_support_margin_m, -0.0005)
                self.assertLessEqual(result.maximum_lipm_residual_m, 1e-6)

    def test_ppo_preflight_covers_training_lookup_and_exact_eval_command(self):
        results = _validate_training_zmp_reference(load_training_config())
        commands = tuple(result.command_forward_m_s for result in results)
        np.testing.assert_allclose(
            commands,
            (0.08, 0.105, 0.13, 0.155, 0.18, 0.10),
            atol=1e-9,
        )
        self.assertTrue(all(result.passed for result in results))

    def test_support_schedule_has_tb_style_double_support(self):
        expected = {
            0.0: (True, True),
            np.pi / 2.0: (False, True),
            np.pi: (True, True),
            3.0 * np.pi / 2.0: (True, False),
        }
        for phase, stance_expected in expected.items():
            with self.subTest(phase=phase):
                _, _, stance = periodic_foot_reference(
                    jp.asarray(phase),
                    0.15,
                    0.72,
                    0.04,
                    2.0,
                )
                np.testing.assert_array_equal(np.asarray(stance), stance_expected)

                expected_contact = expected_foot_contacts(
                    jp.asarray(phase),
                    jp.asarray(True),
                    2.0,
                )
                np.testing.assert_array_equal(
                    np.asarray(expected_contact),
                    stance_expected,
                )

    def test_reward_height_target_matches_tb_half_cycle_semantics(self):
        expected = {
            0.0: (0.0, 0.0),
            np.pi / 2.0: (0.04, 0.0),
            np.pi: (0.0, 0.0),
            3.0 * np.pi / 2.0: (0.0, 0.04),
        }
        for phase, height_expected in expected.items():
            with self.subTest(phase=phase):
                reward_height = expected_foot_heights(
                    jp.asarray(phase),
                    0.04,
                    jp.asarray(True),
                )
                np.testing.assert_allclose(
                    np.asarray(reward_height),
                    height_expected,
                    atol=1e-8,
                )

        standing = expected_foot_heights(
            jp.asarray(np.pi / 2.0),
            0.04,
            jp.asarray(False),
        )
        np.testing.assert_array_equal(np.asarray(standing), (0.0, 0.0))

    def test_stance_foot_is_stationary_in_world_x(self):
        command = 0.20
        cycle_time = 0.72
        first_phase = np.pi
        second_phase = np.pi + 0.1
        first_x, _, first_stance = periodic_foot_reference(
            jp.asarray(first_phase), command, cycle_time, 0.04, 2.0
        )
        second_x, _, second_stance = periodic_foot_reference(
            jp.asarray(second_phase), command, cycle_time, 0.04, 2.0
        )
        self.assertTrue(bool(np.asarray(first_stance)[0]))
        self.assertTrue(bool(np.asarray(second_stance)[0]))
        root_delta = command * cycle_time * (second_phase - first_phase) / (2.0 * np.pi)
        world_delta = root_delta + float(
            np.asarray(second_x)[0] - np.asarray(first_x)[0]
        )
        self.assertAlmostEqual(world_delta, 0.0, places=7)

    def test_periodic_foot_reference_is_continuous_at_cycle_boundary(self):
        before = periodic_foot_reference(
            jp.asarray(2.0 * np.pi - 1e-5),
            0.20,
            0.72,
            0.04,
            2.0,
        )
        after = periodic_foot_reference(
            jp.asarray(1e-5),
            0.20,
            0.72,
            0.04,
            2.0,
        )
        np.testing.assert_allclose(
            np.asarray(before[0]), np.asarray(after[0]), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(before[1]), np.asarray(after[1]), atol=1e-8
        )
        np.testing.assert_array_equal(np.asarray(before[2]), np.asarray(after[2]))


if __name__ == "__main__":
    unittest.main()
