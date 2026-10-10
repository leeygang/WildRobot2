import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.evaluate import _conditional_tracking, evaluate_checkpoint, parse_args


TRIAL = Path("wr2/locomotion/configs/ppo_walking_mirror_off.yaml")


def evaluation_metrics():
    metrics = {
        "eval/avg_episode_length": 1000.0,
        "eval/episode_forward_velocity_error_m_s_per_step": 0.02,
        "eval/episode_contact_phase_match_per_step": 0.85,
        "eval/episode_double_support_per_step": 0.1,
        "eval/episode_fall": 0.0,
    }
    for category, count, error in (
        ("standing", 800.0, 0.0),
        ("walking", 200.0, 20.0),
        ("start", 25.0, 5.0),
        ("stop", 0.0, 0.0),
    ):
        metrics[f"eval/episode_{category}_sample_count"] = count
        metrics[f"eval/episode_{category}_velocity_error_sum"] = error
    for index in range(4):
        metrics[f"eval/episode_phase{index}_velocity_sum"] = 2.5
    return metrics


class WalkingEvaluationTest(unittest.TestCase):
    def test_training_distribution_can_restart_at_all_six_command_clock_phases(self):
        from wr2.locomotion.walking_env import WR2WalkingEnv

        config = load_training_config(TRIAL).environment
        env = WR2WalkingEnv(config, add_observation_noise=False)
        keys = jax.random.split(jax.random.PRNGKey(808), 384)
        commands = np.asarray(jax.vmap(
            lambda key: env._next_command(jp.asarray(450), jp.zeros(3), key)
        )(keys))
        walking = commands[:, 0] > 0
        self.assertTrue(walking.any())
        self.assertTrue((~walking).any())
        self.assertTrue((commands[walking, 0] >= 0.05).all())
        self.assertTrue((commands[walking, 0] <= 0.1).all())
        np.testing.assert_array_equal(commands[:, 1:], 0.0)
        phases = np.mod(
            np.arange(1, 7) * config.command_resample_steps * env._gait_phase_increment,
            2 * np.pi,
        )
        np.testing.assert_allclose(
            np.mod(np.round(np.rad2deg(phases)), 360), [60, 120, 180, 240, 300, 0]
        )
        # No early command change: a standing state remains standing at step449.
        np.testing.assert_array_equal(
            env._next_command(jp.asarray(449), jp.zeros(3), keys[0]), jp.zeros(3)
        )

    def test_conditional_tracking_is_not_diluted_by_standing(self):
        result = _conditional_tracking(evaluation_metrics())
        self.assertEqual(result["walking"]["samples_per_episode"], 200.0)
        self.assertEqual(result["walking"]["velocity_error_m_s"], 0.1)
        self.assertEqual(result["walking"]["forward_velocity_m_s"], 0.05)
        self.assertEqual(result["standing"]["velocity_error_m_s"], 0.0)
        self.assertEqual(result["start"]["velocity_error_m_s"], 0.2)
        self.assertIsNone(result["stop"]["velocity_error_m_s"])
        metrics = evaluation_metrics()
        metrics["eval/episode_walking_sample_count"] = 0.0
        self.assertIsNone(_conditional_tracking(metrics)["walking"]["forward_velocity_m_s"])

    def test_cli_random_commands_are_explicit_and_mutually_exclusive(self):
        base = ["--checkpoint", "policy.pt", "--output", "report.json"]
        args = parse_args(base + ["--random-commands"])
        self.assertTrue(args.random_commands)
        self.assertFalse(args.transitions)
        self.assertIsNone(args.commands)
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (["--transitions"], ["--commands", "0.05,0.10"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    parse_args(base + ["--random-commands"] + extra)

    def test_random_api_rejects_conflicting_schedules_before_loading_checkpoint(self):
        for transitions, commands in ((True, None), (False, (0.1,))):
            with self.subTest(transitions=transitions), self.assertRaisesRegex(
                ValueError, "cannot be combined"
            ):
                evaluate_checkpoint(
                    Path("missing.pt"), config_path=TRIAL, seeds=(707,),
                    forward_commands_m_s=commands, num_envs=128, allow_cpu=True,
                    transitions=transitions, random_commands=True,
                )

    def test_random_evaluation_preserves_distribution_variation_and_paired_keys(self):
        environments, evaluator_keys, randomization_keys = [], [], []

        class FakeEnvironment:
            def __init__(self, config, *, add_observation_noise):
                self.config = config
                environments.append((config, add_observation_noise))

        class FakeEvaluator:
            def __init__(self, _environment, _policy_factory, *, key, **kwargs):
                evaluator_keys.append(key)

            def run_evaluation(self, _params, _metrics):
                return evaluation_metrics()

        def randomizer(_environment, *, rng):
            randomization_keys.append(rng)

        def wrap(environment, *, randomization_fn, **kwargs):
            randomization_fn(environment)
            return environment

        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
            patch("wr2.locomotion.walking_env.WR2WalkingEnv", FakeEnvironment),
            patch("wr2.locomotion.rsl_rl_training.load_rsl_actor_parameters", return_value=()),
            patch("wr2.locomotion.domain_randomization.make_domain_randomizer", return_value=randomizer),
            patch("brax.envs.training.wrap", side_effect=wrap),
            patch("brax.training.acting.Evaluator", FakeEvaluator),
        ):
            checkpoint = Path(folder) / "policy.pt"
            checkpoint.touch()
            report = evaluate_checkpoint(
                checkpoint, config_path=TRIAL, seeds=(707, 808),
                forward_commands_m_s=None, num_envs=128, allow_cpu=True,
                random_commands=True,
            )
        self.assertEqual(environments, [(load_training_config(TRIAL).environment, True)])
        self.assertEqual(report["command_schedule"], "random_training_distribution")
        self.assertEqual(report["commands_forward_m_s"], [])
        self.assertEqual(report["training_command_distribution"], {
            "forward_range_m_s": [0.05, 0.1],
            "standing_probability": 0.2,
            "resample_steps": 150,
        })
        self.assertTrue(all(report["variation"].values()))
        self.assertIsNone(report["command_results"][0]["command_forward_m_s"])
        for index, seed in enumerate((707, 808)):
            wrap_key, eval_key = jax.random.split(jax.random.PRNGKey(seed))
            np.testing.assert_array_equal(evaluator_keys[index], eval_key)
            np.testing.assert_array_equal(randomization_keys[index], jax.random.split(wrap_key, 128))
            result = report["command_results"][0]["seed_results"][index]
            self.assertIsNone(result["walking_score"])
            self.assertEqual(result["conditional_tracking"]["walking"]["velocity_error_m_s"], 0.1)

    def test_fixed_scores_remain_available_but_scripted_scores_are_not_promotable(self):
        class FakeEvaluator:
            def __init__(self, *args, **kwargs):
                pass

            def run_evaluation(self, *args):
                return evaluation_metrics()

        class FakeEnvironment:
            def __init__(self, config, **kwargs):
                self.config = config

        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
            patch("wr2.locomotion.walking_env.WR2WalkingEnv", FakeEnvironment),
            patch("wr2.locomotion.rsl_rl_training.load_rsl_actor_parameters", return_value=()),
            patch("wr2.locomotion.domain_randomization.make_domain_randomizer", return_value=None),
            patch("brax.envs.training.wrap", side_effect=lambda env, **kwargs: env),
            patch("brax.training.acting.Evaluator", FakeEvaluator),
        ):
            checkpoint = Path(folder) / "policy.pt"
            checkpoint.touch()
            for transitions in (False, True):
                report = evaluate_checkpoint(
                    checkpoint, config_path=TRIAL, seeds=(707,),
                    forward_commands_m_s=(0.05, 0.1), num_envs=128, allow_cpu=True,
                    transitions=transitions,
                )
                self.assertEqual(report["commands_forward_m_s"], [0.05, 0.1])
                for result in report["command_results"]:
                    score = result["seed_results"][0]["walking_score"]
                    if transitions:
                        self.assertIsNone(score)
                    else:
                        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
