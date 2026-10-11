import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.evaluate import (
    _conditional_tracking,
    parse_args,
    transition_environment,
    evaluate_checkpoint,
)
from wr2.locomotion.train import (
    _checkpoint_evaluation_xy_commands,
    _validate_training_zmp_reference,
)
from wr2.locomotion.walking_env import WR2WalkingEnv, sample_tb_xy_command
from wr2.reference.walk_zmp import periodic_foot_reference, reference_command_grid
from wr2.reference.zmp_validation import (
    create_reference_context,
    validate_zmp_reference,
)


XY_CONFIG = Path("wr2/locomotion/configs/ppo_walking_xy.yaml")


class XYWalkingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_training_config(XY_CONFIG)
        cls.env = WR2WalkingEnv(cls.config.environment, add_observation_noise=False)

    def test_sampler_matches_current_tb_with_turn_chance_zero(self):
        # Golden outputs from local TB WalkEnv._sample_command, using its
        # walk.gin XY ranges/deadzones, zero_chance=.2 and turn_chance=0.
        expected = {
            0: [0.0573543794, -0.0053272708, 0.0],
            1: [0.0693797022, 0.0279448517, 0.0],
            7: [0.0644765943, 0.0058506294, 0.0],
            41: [0.0146212671, -0.0485279560, 0.0],
            707: [0.0537002422, -0.0187254995, 0.0],
            808: [0.0189312771, 0.0470175631, 0.0],
            909: [0.0, 0.0, 0.0],
            1234: [-0.0069545559, 0.0496529825, 0.0],
        }
        for seed, command in expected.items():
            with self.subTest(seed=seed):
                np.testing.assert_allclose(
                    sample_tb_xy_command(
                        jax.random.PRNGKey(seed), self.config.environment
                    ),
                    command,
                    atol=1e-7,
                )

    def test_sampler_covers_all_quadrants_and_keeps_standing_and_bounds(self):
        keys = jax.random.split(jax.random.PRNGKey(222), 10000)
        commands = np.asarray(jax.jit(jax.vmap(self.env._sample_command))(keys))
        standing = np.linalg.norm(commands, axis=1) == 0
        self.assertGreater(standing.mean(), 0.18)
        self.assertLess(standing.mean(), 0.22)
        walking = commands[~standing]
        np.testing.assert_array_equal(commands[:, 2], 0.0)
        self.assertTrue((np.linalg.norm(walking, axis=1) >= 0.05 - 1e-7).all())
        self.assertTrue(
            ((walking[:, 0] / 0.1) ** 2 + (walking[:, 1] / 0.05) ** 2 <= 1 + 1e-6).all()
        )
        for x in (-1, 1):
            for y in (-1, 1):
                self.assertGreater(
                    ((walking[:, 0] * x > 0) & (walking[:, 1] * y > 0)).mean(), 0.20
                )

    def test_config_preserves_tb_recipe_and_waist_is_not_a_leg_action(self):
        forward = load_training_config(
            "wr2/locomotion/configs/ppo_walking_mirror_off_seed2.yaml"
        )
        self.assertEqual(self.config.environment.rewards, forward.environment.rewards)
        self.assertEqual(
            self.config.environment.randomization, forward.environment.randomization
        )
        self.assertEqual(self.config.network, forward.network)
        self.assertEqual(self.config.environment.command_yaw_range_rad_s, (0.0, 0.0))
        self.assertEqual(self.config.ppo.mirror_loss_coeff, 0.0)
        self.assertEqual(self.env.action_size, 10)
        self.assertNotIn(
            self.env.robot.actuator_names.index("waist_yaw_drive"),
            np.asarray(self.env._active_indices),
        )
        self.assertEqual(
            self.env.observation_size, {"state": (855,), "privileged_state": (1470,)}
        )

    def test_reference_geometry_covers_full_lookup_and_heading_envelope(self):
        results = _validate_training_zmp_reference(self.config)
        # All 49 lookup points plus four exact diagonal evaluation references.
        self.assertEqual(len(results), 53)
        self.assertTrue(all(result.passed for result in results))
        self.assertGreater(
            min(result.minimum_zmp_support_margin_m for result in results), 0.007
        )
        self.assertLess(max(result.maximum_lipm_residual_m for result in results), 1e-6)

    def test_reference_uses_xy_and_tb_heading_correction(self):
        reference = self.env._zmp_reference
        phase = jp.asarray(np.pi / 3)
        left = reference.joint_position(
            phase, jp.asarray([0.0, 0.05, 0.0]), jp.asarray(True)
        )
        right = reference.joint_position(
            phase, jp.asarray([0.0, -0.05, 0.0]), jp.asarray(True)
        )
        self.assertFalse(np.allclose(left, right))
        corrected = reference.joint_position(
            phase,
            jp.asarray([0.05, 0.0, 0.0]),
            jp.asarray(True),
            heading_error_rad=jp.asarray(np.pi / 2),
        )
        np.testing.assert_allclose(corrected, left, atol=1e-6)
        home = reference.joint_position(phase, jp.zeros(3), jp.asarray(False))
        np.testing.assert_allclose(home, self.env.robot.home_position_rad, atol=1e-7)

    def test_xy_stance_feet_are_stationary_in_world(self):
        for command in (-0.1, -0.05, 0.05, 0.1):
            phase = np.pi
            first, _, stance = periodic_foot_reference(
                jp.asarray(phase), command, 0.72, 0.04, 2.0
            )
            second, _, _ = periodic_foot_reference(
                jp.asarray(phase + 0.1), command, 0.72, 0.04, 2.0
            )
            self.assertTrue(bool(stance[0]))
            world_delta = command * 0.72 * 0.1 / (2 * np.pi) + float(
                second[0] - first[0]
            )
            self.assertAlmostEqual(world_delta, 0.0, places=7)

    def test_checkpoint_evaluation_covers_signed_axes_and_diagonals(self):
        commands = _checkpoint_evaluation_xy_commands(self.config)
        self.assertEqual(
            set(commands),
            {(x, 0.0) for x in (-0.1, -0.05, 0.05, 0.1)}
            | {(0.0, y) for y in (-0.05, 0.05)}
            | {(x / np.sqrt(2), y / np.sqrt(2))
               for x in (-0.1, 0.1) for y in (-0.05, 0.05)},
        )
        for x, y in commands:
            self.assertLessEqual((x / 0.1) ** 2 + (y / 0.05) ** 2, 1.0 + 1e-9)
        self.assertEqual(len(reference_command_grid((0.05, 0.1), (0.0, 0.0), 5)), 5)

    def test_cli_supports_xy_commands_standing_and_signed_forward(self):
        base = ["--checkpoint", "policy.pt", "--output", "report.json"]
        args = parse_args(base + ["--xy-commands", "0.1,0;-0.1,0;0,0.05;0,-0.05;0,0"])
        self.assertEqual(args.xy_commands[-1], (0.0, 0.0))
        self.assertEqual(
            parse_args(base + ["--commands=-0.1,0.1"]).commands, (-0.1, 0.1)
        )
        with contextlib.redirect_stderr(io.StringIO()):
            for value in ("0.1,0,0", "nan,0", "0,inf", "0,0;0,0"):
                with self.subTest(value=value), self.assertRaises(SystemExit):
                    parse_args(base + ["--xy-commands", value])

    def test_scripted_lateral_start_and_stop_do_not_become_forward_commands(self):
        env = transition_environment(
            replace(
                self.config.environment,
                command_forward_range_m_s=(0.0, 0.0),
                command_lateral_range_m_s=(0.05, 0.05),
            ),
            add_observation_noise=False,
        )
        for step, command in (
            (149, [0.0, 0.0, 0.0]),
            (150, [0.0, 0.05, 0.0]),
            (450, [0.0, 0.0, 0.0]),
            (600, [0.0, 0.05, 0.0]),
        ):
            np.testing.assert_allclose(
                env._next_command(jp.asarray(step), jp.zeros(3), jax.random.PRNGKey(1)),
                command,
            )

    def test_conditional_xy_metrics_use_active_sample_counts(self):
        metrics = {f"eval/episode_phase{i}_velocity_sum": 0.0 for i in range(4)}
        for category in ("standing", "walking", "start", "stop"):
            for suffix, value in (
                ("sample_count", 10.0),
                ("velocity_error_sum", 0.0),
                ("xy_velocity_error_sum", 0.3),
                ("lateral_velocity_error_sum", 0.3),
                ("lateral_velocity_sum", -0.5),
            ):
                metrics[f"eval/episode_{category}_{suffix}"] = value
        result = _conditional_tracking(metrics)["walking"]
        self.assertAlmostEqual(result["xy_velocity_error_m_s"], 0.03)
        self.assertAlmostEqual(result["lateral_velocity_m_s"], -0.05)
        self.assertEqual(result["velocity_error_m_s"], 0.0)

    def test_xy_reference_interpolation_has_ground_and_joint_limit_checks(self):
        keys = ((0.1, 0.05), (-0.1, -0.05), (0.0, 0.05), (0.0, -0.05), (0.0, 0.0))
        context = create_reference_context(
            training_config=self.config, commands_xy_m_s=keys
        )
        for command in keys:
            result = validate_zmp_reference(context, *command, samples=144)
            self.assertTrue(result.passed, msg=str(result.failures[:5]))
            self.assertLess(result.maximum_joint_step_rad, 0.1)

    def test_lipm_acceleration_matches_independent_finite_difference(self):
        reference = self.env._zmp_reference
        omega = 2 * np.pi / self.config.environment.gait_cycle_s
        phase_step = 1e-4
        for command in ((0.1, 0.05), (-0.1, -0.05), (0.0, 0.05)):
            for phase in (0.4, 1.2, 2.5, 4.7):
                center = reference.sample(phase, *command)
                before = reference.sample(phase - phase_step, *command)
                after = reference.sample(phase + phase_step, *command)
                acceleration = (
                    after.com_lateral_m
                    - 2 * center.com_lateral_m
                    + before.com_lateral_m
                ) / (phase_step / omega) ** 2
                implied_zmp = (
                    center.com_lateral_m
                    - self.config.environment.zmp_reference.com_height_m
                    / 9.81
                    * acceleration
                )
                self.assertAlmostEqual(implied_zmp, center.zmp_lateral_m, places=6)

    def test_lookup_joints_remain_inside_physical_limits(self):
        table = np.asarray(self.env._zmp_reference._joint_lookup)
        self.assertTrue((table >= self.env.robot.lower_limit_rad - 1e-6).all())
        self.assertTrue((table <= self.env.robot.upper_limit_rad + 1e-6).all())

    def test_xy_evaluation_scores_lateral_error_not_zero_forward_error(self):
        environments = []

        class FakeEnvironment:
            def __init__(self, config, **kwargs):
                self.config = config
                environments.append(config)

        class FakeEvaluator:
            def __init__(self, *args, **kwargs):
                pass

            def run_evaluation(self, *args):
                metrics = {
                    "eval/avg_episode_length": 1000.0,
                    "eval/episode_forward_velocity_error_m_s_per_step": 0.0,
                    "eval/episode_xy_velocity_error_m_s_per_step": 0.04,
                    "eval/episode_contact_phase_match_per_step": 0.85,
                    "eval/episode_double_support_per_step": 0.1,
                    "eval/episode_fall": 0.0,
                }
                for name in ("standing", "walking", "start", "stop"):
                    metrics[f"eval/episode_{name}_sample_count"] = 1000.0
                    metrics[f"eval/episode_{name}_velocity_error_sum"] = 0.0
                    metrics[f"eval/episode_{name}_xy_velocity_error_sum"] = 40.0
                    metrics[f"eval/episode_{name}_lateral_velocity_sum"] = 10.0
                for i in range(4):
                    metrics[f"eval/episode_phase{i}_velocity_sum"] = 0.0
                return metrics

        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
            patch("wr2.locomotion.walking_env.WR2WalkingEnv", FakeEnvironment),
            patch(
                "wr2.locomotion.rsl_rl_training.load_rsl_actor_parameters",
                return_value=(),
            ),
            patch(
                "wr2.locomotion.domain_randomization.make_domain_randomizer",
                return_value=None,
            ),
            patch("brax.envs.training.wrap", side_effect=lambda env, **kwargs: env),
            patch("brax.training.acting.Evaluator", FakeEvaluator),
        ):
            checkpoint = Path(folder) / "policy.pt"
            checkpoint.touch()
            report = evaluate_checkpoint(
                checkpoint,
                config_path=XY_CONFIG,
                seeds=(707,),
                forward_commands_m_s=None,
                xy_commands_m_s=((0.0, 0.05), (0.0, -0.05), (0.0, 0.0)),
                num_envs=2,
                allow_cpu=True,
            )
        self.assertEqual(
            report["commands_xy_m_s"], [[0.0, 0.05], [0.0, -0.05], [0.0, 0.0]]
        )
        for item in report["command_results"][:2]:
            row = item["seed_results"][0]
            self.assertLess(row["walking_score"], 0.25)
            self.assertEqual(
                row["conditional_tracking"]["walking"]["xy_velocity_error_m_s"], 0.04
            )
        self.assertIsNone(
            report["command_results"][-1]["seed_results"][0]["walking_score"]
        )
        self.assertEqual(environments[0].command_lateral_range_m_s, (0.05, 0.05))
        self.assertEqual(environments[1].command_lateral_range_m_s, (-0.05, -0.05))


if __name__ == "__main__":
    unittest.main()
