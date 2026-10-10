import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.rsl_rl_training import RSLPPOTrainer, _training_rng_keys


CONFIG = Path("wr2/locomotion/configs/ppo_walking_mirror_off_seed1.yaml")
PARENT_CONFIG = Path("wr2/locomotion/configs/ppo_walking_mirror_off.yaml")
SOURCE = Path("results/wr2_walking/wr2_mirror_off_20261010_083343_seed0/checkpoints/000080035840.pt")


class MirrorOffContinuationTest(unittest.TestCase):
    def test_followup_changes_only_environment_seed_and_version_labels(self):
        from wr2.locomotion.train import _apply_cli_overrides, parse_args

        parent = load_training_config(PARENT_CONFIG)
        followup = load_training_config(CONFIG)
        self.assertEqual(followup, replace(
            parent, seed=1, version=followup.version, version_name=followup.version_name
        ))
        with patch("sys.argv", [
            "train", "--config", str(CONFIG), "--restore-checkpoint", str(SOURCE)
        ]):
            args = parse_args()
        self.assertIsNone(args.run_id)
        self.assertEqual(args.restore_checkpoint, SOURCE)
        self.assertEqual(_apply_cli_overrides(followup, args), followup)
        self.assertEqual(followup.ppo.num_timesteps, 200_000_000)
        self.assertEqual(followup.ppo.num_evals, 11)
        self.assertEqual(followup.ppo.mirror_loss_coeff, 0.0)

    def test_seed_one_changes_all_jax_streams_repeatably(self):
        zero, one, repeated = (_training_rng_keys(seed) for seed in (0, 1, 1))
        for old, new, same in zip(zero, one, repeated, strict=True):
            self.assertFalse(np.array_equal(old, new))
            np.testing.assert_array_equal(new, same)

    def test_new_environment_seed_does_not_reset_native_policy_or_adam_rng(self):
        config = load_training_config(CONFIG)
        config = replace(config,
            ppo=replace(config.ppo, num_envs=4, unroll_length=2, num_minibatches=1),
            network=replace(config.network, policy_hidden_layer_sizes=(8, 4), value_hidden_layer_sizes=(8, 4)),
        )
        env = SimpleNamespace(
            device=torch.device("cpu"), num_envs=4, num_actions=2,
            get_observations=lambda: (torch.zeros(4, 4), {
                "observations": {"critic": torch.zeros(4, 6)}
            }),
        )

        def trainer(training_config, restore=None):
            return RSLPPOTrainer(env, training_config, output=Path("unused"),
                restore_checkpoint=restore, progress_fn=lambda *args: None,
                policy_params_fn=lambda *args: None, evaluator=None)

        source = trainer(replace(config, seed=0))
        sum(parameter.sum() for parameter in source.policy.parameters()).backward()
        source.algorithm.optimizer.step()
        source.completed_iterations = 24416
        source.total_steps = 1_000_079_360
        source.algorithm.learning_rate = 1e-5
        for group in source.algorithm.optimizer.param_groups:
            group["lr"] = 1e-5
        with torch.no_grad():
            source.policy.log_std.fill_(np.log(0.0260726))
        saved = source.checkpoint_state()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "source.pt"
            torch.save(saved, checkpoint)
            # train_rsl_rl seeds initial construction, then load restores Torch RNG.
            torch.manual_seed(config.seed)
            restored = trainer(config, checkpoint)
        actual = restored.checkpoint_state()
        for key, tensor in saved["model_state_dict"].items():
            torch.testing.assert_close(actual["model_state_dict"][key], tensor, rtol=0, atol=0)
        old_adam, new_adam = (state["optimizer_state_dict"] for state in (saved, actual))
        self.assertEqual(new_adam["param_groups"], old_adam["param_groups"])
        self.assertEqual(new_adam["state"].keys(), old_adam["state"].keys())
        for index, moments in old_adam["state"].items():
            for key, tensor in moments.items():
                torch.testing.assert_close(new_adam["state"][index][key], tensor, rtol=0, atol=0)
        for key in ("iteration", "total_steps", "learning_rate", "mirror_loss_coeff"):
            self.assertEqual(actual[key], saved[key])
        torch.testing.assert_close(actual["torch_rng_state"], saved["torch_rng_state"], rtol=0, atol=0)
        self.assertIsNone(restored.algorithm.symmetry)


if __name__ == "__main__":
    unittest.main()
