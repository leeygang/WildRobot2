import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import torch
from brax.envs.base import State
from rsl_rl.modules import ActorCritic

from wr2.locomotion.rsl_rl_training import (
    _EpisodeMetrics,
    _MultiCommandEvaluator,
    _actor_parameters_for_jax,
    _diagonal_gaussian_kl,
    _jax_policy_factory,
    _training_rng_keys,
    RSLRLWrapper,
    evaluation_iterations,
    learning_iterations,
    load_rsl_actor_parameters,
    rsl_minibatch_size,
)


class RSLRLTrainingTest(unittest.TestCase):
    def test_wrapper_forwards_brax_truncation_as_rsl_timeout(self):
        class FakeEnvironment:
            action_size = 1

            def reset(self, keys):
                count = keys.shape[0]
                zeros = jp.zeros((count,))
                observations = jp.zeros((count, 1))
                return State(
                    pipeline_state=None,
                    obs={"state": observations, "privileged_state": observations},
                    reward=zeros,
                    done=zeros,
                    metrics={},
                    info={},
                )

            def step(self, state, _actions):
                ones = jp.ones_like(state.done)
                info = {
                    "episode_metrics": {"length": ones},
                    "episode_done": ones,
                    "truncation": ones,
                }
                return state.replace(reward=ones, done=ones, info=info)

        wrapper = RSLRLWrapper(
            FakeEnvironment(),
            device=torch.device("cpu"),
            num_envs=2,
            episode_length=5,
            reset_rng=jax.random.PRNGKey(5),
        )

        _, _, dones, infos = wrapper.step(torch.zeros((2, 1)))

        torch.testing.assert_close(dones, torch.ones(2))
        torch.testing.assert_close(infos["time_outs"], torch.ones(2))

    def test_multi_command_evaluator_preserves_primary_and_endpoint_metrics(self):
        class FakeEvaluator:
            def __init__(self, value):
                self.value = value

            def run_evaluation(self, _parameters, training_metrics):
                return {**training_metrics, "eval/value": self.value}

        evaluator = _MultiCommandEvaluator(
            (FakeEvaluator(0.05), FakeEvaluator(0.10)),
            primary_index=1,
        )
        metrics = evaluator.run_evaluation(None, {"training/loss": 3.0})

        self.assertEqual(metrics["eval_endpoint_0/value"], 0.05)
        self.assertEqual(metrics["eval_endpoint_1/value"], 0.10)
        self.assertEqual(metrics["eval/value"], 0.10)
        self.assertEqual(metrics["training/loss"], 3.0)

    def test_iteration_and_evaluation_cadence_matches_requested_steps(self):
        iterations = learning_iterations(1_000_000_000, 2048, 20)
        self.assertEqual(iterations, 24_415)
        evaluations = evaluation_iterations(iterations, 50)
        self.assertEqual(len(evaluations), 49)
        self.assertEqual(max(evaluations), iterations)

    def test_rsl_minibatch_uses_rollout_transitions_not_brax_batch_size(self):
        self.assertEqual(rsl_minibatch_size(4096, 20, 16), 5120)
        self.assertEqual(rsl_minibatch_size(2048, 20, 8), 5120)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            rsl_minibatch_size(3, 5, 4)

    def test_training_rng_streams_are_distinct_and_repeatable(self):
        keys = _training_rng_keys(7)
        repeated = _training_rng_keys(7)
        self.assertEqual(len(keys), 3)
        for key, repeated_key in zip(keys, repeated, strict=True):
            np.testing.assert_array_equal(key, repeated_key)
        self.assertFalse(np.array_equal(keys[0], keys[1]))
        self.assertFalse(np.array_equal(keys[0], keys[2]))
        self.assertFalse(np.array_equal(keys[1], keys[2]))

    def test_episode_reward_uses_brax_sum_reward(self):
        metrics = _EpisodeMetrics()
        metrics.update(
            {
                "length": torch.tensor([4.0]),
                "reward": torch.tensor([0.0]),
                "sum_reward": torch.tensor([8.0]),
                "velocity_per_step": torch.tensor([2.0]),
            },
            torch.tensor([True]),
        )
        means = metrics.means()
        self.assertEqual(means["reward"], 8.0)
        self.assertEqual(means["sum_reward"], 8.0)
        self.assertEqual(means["velocity_per_step"], 0.5)

    def test_diagonal_gaussian_kl_is_zero_only_for_same_policy(self):
        old_mean = torch.zeros((2, 3))
        old_std = torch.ones((2, 3))
        self.assertAlmostEqual(
            float(
                _diagonal_gaussian_kl(
                    old_mean, old_std, old_mean.clone(), old_std.clone()
                )
            ),
            0.0,
            places=7,
        )
        shifted = _diagonal_gaussian_kl(
            old_mean, old_std, torch.full((2, 3), 0.25), old_std
        )
        self.assertGreater(float(shifted), 0.0)

    def test_jax_evaluator_exactly_matches_torch_actor_mean(self):
        torch.manual_seed(3)
        policy = ActorCritic(
            7,
            9,
            3,
            actor_hidden_dims=[8, 6],
            critic_hidden_dims=[8, 6],
            activation="elu",
            init_noise_std=0.5,
            noise_std_type="log",
        )
        observation = torch.randn(5, 7)
        with torch.inference_mode():
            expected = policy.act_inference(observation).numpy()
        parameters = _actor_parameters_for_jax(policy)
        actual, extras = _jax_policy_factory("elu")(parameters)(
            {"state": np.asarray(observation)}, jax.random.PRNGKey(0)
        )
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(extras, {})

    def test_checkpoint_actor_loader_uses_full_state_checkpoint(self):
        policy = ActorCritic(
            4,
            6,
            2,
            actor_hidden_dims=[5],
            critic_hidden_dims=[5],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.pt"
            torch.save(
                {"backend": "rsl_rl", "model_state_dict": policy.state_dict()},
                checkpoint,
            )
            loaded = load_rsl_actor_parameters(checkpoint)

        expected = _actor_parameters_for_jax(policy)
        self.assertEqual(len(loaded), len(expected))
        for loaded_layer, expected_layer in zip(loaded, expected, strict=True):
            for loaded_value, expected_value in zip(
                loaded_layer, expected_layer, strict=True
            ):
                np.testing.assert_array_equal(loaded_value, expected_value)


if __name__ == "__main__":
    unittest.main()
