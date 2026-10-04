import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np
import torch
from rsl_rl.modules import ActorCritic

from wr2.locomotion.rsl_rl_training import (
    _actor_parameters_for_jax,
    _jax_policy_factory,
    evaluation_iterations,
    learning_iterations,
    load_rsl_actor_parameters,
)


class RSLRLTrainingTest(unittest.TestCase):
    def test_iteration_and_evaluation_cadence_matches_requested_steps(self):
        iterations = learning_iterations(1_000_000_000, 2048, 20)
        self.assertEqual(iterations, 24_415)
        evaluations = evaluation_iterations(iterations, 50)
        self.assertEqual(len(evaluations), 49)
        self.assertEqual(max(evaluations), iterations)

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
