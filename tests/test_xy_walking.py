import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jp
import mujoco
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
    _checkpoint_transition_xy_commands,
    _validate_training_zmp_reference,
    parse_args as parse_training_args,
    train,
)
from wr2.locomotion.walking_metrics import walking_score
from wr2.locomotion.walking_env import WR2WalkingEnv, sample_tb_xy_command
from wr2.reference.walk_zmp import WR2ZMPReference, periodic_foot_reference, reference_command_grid
from wr2.reference.zmp_validation import (
    ZMPGeometryResult,
    create_reference_context,
    validate_zmp_reference,
)


XY_CONFIG = Path("wr2/locomotion/configs/ppo_walking_xy.yaml")
PHYSICS_CONFIG = Path("wr2/locomotion/configs/ppo_walking_xy_physics.yaml")


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

    def test_physics_continuation_preserves_training_recipe_and_retains_checkpoints(self):
        from wr2.agents.walking_training_agent import DEFAULT_AGENT_CONFIG, load_agent_config
        from wr2.locomotion.rsl_rl_training import learning_iterations, evaluation_iterations

        config = load_training_config(PHYSICS_CONFIG)
        self.assertEqual(config.environment, self.config.environment)
        self.assertEqual(config.network, self.config.network)
        self.assertEqual(replace(
            config.ppo, num_timesteps=self.config.ppo.num_timesteps,
            num_evals=self.config.ppo.num_evals, evaluation_randomization=False,
        ), self.config.ppo)
        self.assertEqual(
            config.checkpoints.qualification_goal,
            load_agent_config(DEFAULT_AGENT_CONFIG).stages[-1].goal,
        )
        iterations = learning_iterations(config.ppo.num_timesteps, config.ppo.num_envs, config.ppo.unroll_length)
        self.assertEqual(iterations, 4883)
        self.assertEqual(iterations * 2048 * 20, 200_007_680)
        self.assertEqual(len(evaluation_iterations(iterations, config.ppo.num_evals)), 10)
        self.assertTrue(config.checkpoints.save_every_evaluation)
        self.assertEqual(_checkpoint_evaluation_xy_commands(config)[-1], (0.0, 0.0))
        self.assertEqual(_checkpoint_transition_xy_commands(config), (
            (0.1, 0.0), (-0.1, 0.0), (0.0, -0.05), (0.0, 0.05),
        ))

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
        self.assertEqual(self.env.action_size, 12)
        self.assertNotIn(
            self.env.robot.actuator_names.index("waist_yaw_drive"),
            np.asarray(self.env._active_indices),
        )
        self.assertEqual(
            self.env.observation_size, {"state": (1005,), "privileged_state": (1740,)}
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

    def test_training_progress_keeps_headline_endpoint_and_transition_axes_aligned(self):
        for xy_training, include_xy_error in ((True, True), (False, True), (True, False)):
            with self.subTest(xy_training=xy_training, include_xy_error=include_xy_error):
                metrics = {
                    "eval/avg_episode_length": 1000.0,
                    "eval/episode_forward_velocity_error_m_s_per_step": 0.0,
                    "eval/episode_contact_phase_match_per_step": 0.85,
                    "eval/episode_double_support_per_step": 0.1,
                }
                if include_xy_error:
                    metrics["eval/episode_xy_velocity_error_m_s_per_step"] = 0.04
                config = self.config if xy_training else replace(
                    self.config,
                    environment=replace(
                        self.config.environment, command_lateral_range_m_s=(0.0, 0.0)
                    ),
                )
                for index, _ in enumerate(_checkpoint_evaluation_xy_commands(config)):
                    metrics.update({
                        name.replace("eval/", f"eval_endpoint_{index}/"): value
                        for name, value in tuple(metrics.items())
                        if name.startswith("eval/")
                    })
                metrics["eval_transition/avg_episode_length"] = 1000.0
                for category in ("standing", "walking", "start", "stop"):
                    for suffix, value in (
                        ("sample_count", 10.0),
                        ("velocity_error_sum", 0.0),
                        ("xy_velocity_error_sum", 0.4),
                    ):
                        metrics[f"eval_transition/episode_{category}_{suffix}"] = value

                def fake_training(*args, progress_fn, policy_params_fn, **kwargs):
                    progress_fn(0, metrics)
                    policy_params_fn(0, None, {})

                preflight = ZMPGeometryResult(0.1, 0.0, 1, (), 0.001, 0.0, 0.009, 0.0, 0.0)
                output = io.StringIO()
                with (
                    tempfile.TemporaryDirectory() as folder,
                    contextlib.redirect_stdout(output),
                    patch("sys.argv", ["train", "--allow-cpu", "--run-id", "progress_test"]),
                    patch("wr2.locomotion.walking_env.WR2WalkingEnv", return_value=self.env),
                    patch("wr2.locomotion.train._validate_training_zmp_reference", return_value=(preflight,)),
                    patch("wr2.locomotion.rsl_rl_training.train_rsl_ppo", side_effect=fake_training),
                ):
                    config = replace(config, output=replace(config.output, root=folder))
                    train(parse_training_args(), config)
                    report = json.loads(
                        (Path(folder) / "progress_test/training_metrics.jsonl").read_text()
                    )
                    endpoint_scores = [
                        item["walking_score"] for item in report["endpoint_evaluations"]
                    ]
                    if xy_training and not include_xy_error:
                        self.assertIsNone(report["walking_score"])
                        self.assertTrue(all(score is None for score in endpoint_scores))
                        self.assertFalse((Path(folder) / "progress_test/best_checkpoint.json").exists())
                    else:
                        expected = walking_score(
                            episode_length=1000.0,
                            target_episode_length=1000,
                            velocity_error_m_s=0.04 if xy_training else 0.0,
                            velocity_sigma_m_s=config.environment.velocity_tracking_sigma,
                            contact_match=0.85,
                            double_support=0.1,
                        )
                        self.assertAlmostEqual(report["walking_score"], expected)
                        self.assertTrue(all(score == report["walking_score"] for score in endpoint_scores))
                        best = json.loads(
                            (Path(folder) / "progress_test/best_checkpoint.json").read_text()
                        )
                        self.assertAlmostEqual(best["selection_score"], expected)
                    if xy_training:
                        self.assertIn("XY_MAE standing=0.040", output.getvalue())
                        self.assertIn("XY_MAE=0.040" if include_xy_error else "XY_MAE=n/a", output.getvalue())
                    else:
                        self.assertIn("speed_MAE standing=0.000", output.getvalue())

    def test_xy_qualification_requires_standing_and_every_recovery_direction(self):
        config = load_training_config(PHYSICS_CONFIG)
        base = {
            "eval/avg_episode_length": 1000.0,
            "eval/episode_fall": 0.0,
            "eval/episode_xy_velocity_error_m_s_per_step": 0.01,
            "eval/episode_forward_velocity_error_m_s_per_step": 0.005,
            "eval/episode_lateral_velocity_error_m_s_per_step": 0.005,
            "eval/episode_yaw_rate_error_rad_s_per_step": 0.01,
            "eval/episode_contact_phase_match_per_step": 0.9,
            "eval/episode_left_foot_contact_per_step": 0.65,
            "eval/episode_right_foot_contact_per_step": 0.65,
            "eval/episode_double_support_per_step": 0.1,
            "eval/episode_feet_phase_tracking_per_step": 1.4,
            "eval/episode_actuator_torque_rms_nm_per_step": 0.8,
            "eval/episode_actuator_torque_peak_nm_per_step": 2.5,
            "eval/episode_sustained_torque_exposure_per_step": 0.8,
            "eval/episode_action_saturation_fraction_per_step": 0.01,
            "eval/episode_nonfinite_state": 0.0,
        }
        for mode in ("pass", "missing_transition", "bad_lateral_stop", "nonfinite_standing"):
            with self.subTest(mode=mode):
                metrics = {**base,
                    "eval/episode_forward_velocity_m_s_per_step": 0.095,
                    "eval/episode_lateral_velocity_m_s_per_step": 0.0,
                }
                for index, (x, y) in enumerate(_checkpoint_evaluation_xy_commands(config)):
                    row = {**base,
                        "eval/episode_forward_velocity_m_s_per_step": 0.95 * x,
                        "eval/episode_lateral_velocity_m_s_per_step": 0.95 * y,
                    }
                    if (x, y) == (0.0, 0.0):
                        row["eval/episode_feet_phase_tracking_per_step"] = 1.0
                        if mode == "nonfinite_standing":
                            row["eval/episode_xy_velocity_error_m_s_per_step"] = float("nan")
                    metrics.update({name.replace("eval/", f"eval_endpoint_{index}/"): value for name, value in row.items()})
                for index, _ in enumerate(_checkpoint_transition_xy_commands(config)):
                    if mode == "missing_transition" and index == 3:
                        continue
                    row = dict(base)
                    for category, count in (("standing", 450), ("walking", 550), ("start", 50), ("stop", 50)):
                        row[f"eval/episode_{category}_sample_count"] = count
                        row[f"eval/episode_{category}_xy_velocity_error_sum"] = count * (
                            0.05 if mode == "bad_lateral_stop" and index == 3 and category == "stop" else 0.01
                        )
                    metrics.update({name.replace("eval/", f"eval_transition_endpoint_{index}/"): value for name, value in row.items()})

                def fake_training(*args, progress_fn, policy_params_fn, **kwargs):
                    policy_params_fn(0, None, {})
                    progress_fn(0, metrics)

                preflight = ZMPGeometryResult(0.1, 0.0, 1, (), 0.001, 0.0, 0.009, 0.0, 0.0)
                with (
                    tempfile.TemporaryDirectory() as folder,
                    contextlib.redirect_stdout(io.StringIO()),
                    patch("sys.argv", ["train", "--allow-cpu", "--run-id", "qualified_test"]),
                    patch("wr2.locomotion.walking_env.WR2WalkingEnv", return_value=self.env),
                    patch("wr2.locomotion.train._validate_training_zmp_reference", return_value=(preflight,)),
                    patch("wr2.locomotion.rsl_rl_training.train_rsl_ppo", side_effect=fake_training),
                ):
                    run_config = replace(config, output=replace(config.output, root=folder))
                    train(parse_training_args(), run_config)
                    output = Path(folder) / "qualified_test"
                    report = json.loads((output / "training_metrics.jsonl").read_text())
                    self.assertEqual(report["qualification"]["passed"], mode == "pass")
                    self.assertEqual(len(report["qualification"]["transition_evaluations"]), 4)
                    self.assertEqual((output / "best_checkpoint.json").exists(), mode == "pass")

    def test_invalid_or_missing_endpoint_cannot_fall_back_to_primary_score(self):
        base_metrics = {
            "eval/avg_episode_length": 1000.0,
            "eval/episode_xy_velocity_error_m_s_per_step": 0.04,
            "eval/episode_contact_phase_match_per_step": 0.85,
            "eval/episode_double_support_per_step": 0.1,
        }
        for invalid in (None, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                metrics = dict(base_metrics)
                for index, _ in enumerate(_checkpoint_evaluation_xy_commands(self.config)):
                    metrics.update({
                        name.replace("eval/", f"eval_endpoint_{index}/"): value
                        for name, value in base_metrics.items()
                    })
                key = "eval_endpoint_9/episode_xy_velocity_error_m_s_per_step"
                if invalid is None:
                    del metrics[key]
                else:
                    metrics[key] = invalid

                def fake_training(*args, progress_fn, policy_params_fn, **kwargs):
                    policy_params_fn(0, None, {})
                    progress_fn(0, metrics)

                preflight = ZMPGeometryResult(0.1, 0.0, 1, (), 0.001, 0.0, 0.009, 0.0, 0.0)
                with (
                    tempfile.TemporaryDirectory() as folder,
                    contextlib.redirect_stdout(io.StringIO()),
                    patch("sys.argv", ["train", "--allow-cpu", "--run-id", "invalid_endpoint"]),
                    patch("wr2.locomotion.walking_env.WR2WalkingEnv", return_value=self.env),
                    patch("wr2.locomotion.train._validate_training_zmp_reference", return_value=(preflight,)),
                    patch("wr2.locomotion.rsl_rl_training.train_rsl_ppo", side_effect=fake_training),
                ):
                    config = replace(self.config, output=replace(self.config.output, root=folder))
                    train(parse_training_args(), config)
                    self.assertFalse((Path(folder) / "invalid_endpoint/best_checkpoint.json").exists())

    def test_zmp_cache_rebuilds_after_joint_axis_change(self):
        context = create_reference_context(
            training_config=self.config, commands_xy_m_s=((0.0, 0.05),),
        )
        changed = copy.copy(context.model)
        joint_id = mujoco.mj_name2id(changed, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll")
        axis = changed.jnt_axis[joint_id].copy()
        basis = np.eye(3)[np.argmin(np.abs(axis))]
        orthogonal = basis - axis * np.dot(axis, basis)
        orthogonal /= np.linalg.norm(orthogonal)
        changed.jnt_axis[joint_id] = np.cos(0.2) * axis + np.sin(0.2) * orthogonal
        zmp = self.config.environment.zmp_reference
        reference = WR2ZMPReference(
            changed,
            keyframe_id=mujoco.mj_name2id(changed, mujoco.mjtObj.mjOBJ_KEY, "walk_home"),
            actuator_names=self.env.robot.actuator_names,
            command_forward_range_m_s=(0.0, 0.0),
            command_lateral_range_m_s=(0.05, 0.05),
            command_vectors_m_s=((0.0, 0.05),),
            gait_cycle_s=self.config.environment.gait_cycle_s,
            swing_height_m=self.config.environment.swing_height_m,
            com_height_m=zmp.com_height_m,
            root_height_m=zmp.root_height_m,
            single_double_support_ratio=zmp.single_double_support_ratio,
            phase_samples=zmp.phase_samples,
            command_samples=zmp.command_samples,
            ik_damping=zmp.ik_damping,
            ik_max_iterations=zmp.ik_max_iterations,
            max_position_residual_m=zmp.max_position_residual_m,
            max_orientation_residual_rad=zmp.max_orientation_residual_rad,
        )
        self.assertGreater(
            np.max(np.abs(np.asarray(reference._joint_lookup) - np.asarray(context.reference._joint_lookup))),
            0.001,
        )

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
                    - reference.diagnostics.com_height_m
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
