import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jp
import numpy as np
import torch
import yaml
from rsl_rl.modules import ActorCritic

from wr2.locomotion.configs import load_training_config, training_config_to_dict
from wr2.locomotion.rsl_rl_training import RSLPPOTrainer
from wr2.locomotion.symmetry import ActorMirror, mirror_actor
from wr2.sim.robot import RobotDescription


TRIAL = Path("wr2/locomotion/configs/ppo_walking_mirror.yaml")
FRESH = Path("wr2/locomotion/configs/ppo_walking_fresh.yaml")
MIRROR_OFF = Path("wr2/locomotion/configs/ppo_walking_mirror_off.yaml")


class ActorMirrorLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_training_config(TRIAL)
        robot = RobotDescription.load(include_local_calibration=False)
        cls.base_env = SimpleNamespace(
            robot=robot,
            config=cls.config.environment,
            _active_indices=robot.active_actuator_indices(("leg",)),
        )
        cls.mirror = ActorMirror.from_environment(cls.base_env, "cpu")

    def make_trainer(self, *, coefficient=1.0, restore=None):
        config = replace(
            self.config,
            ppo=replace(
                self.config.ppo,
                num_envs=4,
                unroll_length=2,
                num_minibatches=1,
                num_updates_per_batch=1,
                value_loss_coef=0.0,
                entropy_cost=0.0,
                learning_rate=0.001,
                learning_rate_schedule="fixed",
                mirror_loss_coeff=coefficient,
            ),
            network=replace(
                self.config.network,
                policy_hidden_layer_sizes=(8, 4),
                value_hidden_layer_sizes=(8, 4),
            ),
        )
        env = SimpleNamespace(
            env=SimpleNamespace(unwrapped=self.base_env),
            device=torch.device("cpu"),
            num_envs=4,
            num_actions=10,
            get_observations=lambda: (
                torch.zeros(4, 855),
                {"observations": {"critic": torch.zeros(4, 1470)}},
            ),
        )
        return RSLPPOTrainer(
            env,
            config,
            output=Path("unused"),
            restore_checkpoint=restore,
            progress_fn=lambda *args: None,
            policy_params_fn=lambda *args: None,
            evaluator=None,
        )

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                self.assert_nested_equal(actual[name], expected[name])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected, strict=True):
                self.assert_nested_equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def test_trial_changes_only_the_loss_and_explicit_control_overrides(self):
        base = load_training_config()
        expected = replace(
            base,
            version=self.config.version,
            version_name=self.config.version_name,
            environment=replace(base.environment, solver_iterations=10),
            ppo=replace(
                base.ppo, num_timesteps=20_000_000, num_evals=6, mirror_loss_coeff=1.0
            ),
            output=self.config.output,
        )
        self.assertEqual(self.config, expected)
        self.assertEqual(base.ppo.mirror_loss_coeff, 0.0)

    def test_fresh_config_changes_only_budget_cadence_and_run_labels(self):
        from wr2.locomotion.train import _apply_cli_overrides, parse_args

        fresh = load_training_config(FRESH)
        expected = replace(
            self.config,
            version=fresh.version,
            version_name=fresh.version_name,
            ppo=replace(self.config.ppo, num_timesteps=1_000_000_000, num_evals=51),
            output=replace(self.config.output, run_prefix="wr2_fresh"),
        )
        self.assertEqual(fresh, expected)
        with patch("sys.argv", ["train", "--config", str(FRESH)]):
            args = parse_args()
        self.assertIsNone(args.restore_checkpoint)
        self.assertIsNone(args.run_id)
        self.assertEqual(_apply_cli_overrides(fresh, args), fresh)

    def test_fresh_config_starts_native_policy_and_optimizer_from_zero(self):
        fresh = load_training_config(FRESH)
        env = SimpleNamespace(
            env=SimpleNamespace(unwrapped=self.base_env),
            device=torch.device("cpu"),
            num_envs=4,
            num_actions=10,
            get_observations=lambda: (
                torch.zeros(4, 855),
                {"observations": {"critic": torch.zeros(4, 1470)}},
            ),
        )
        with patch.object(RSLPPOTrainer, "load", side_effect=AssertionError):
            trainer = RSLPPOTrainer(
                env,
                fresh,
                output=Path("unused"),
                restore_checkpoint=None,
                progress_fn=lambda *args: None,
                policy_params_fn=lambda *args: None,
                evaluator=None,
            )
        state = trainer.checkpoint_state()
        self.assertEqual(state["iteration"], 0)
        self.assertEqual(state["total_steps"], 0)
        self.assertEqual(state["optimizer_state_dict"]["state"], {})
        self.assertEqual(state["learning_rate"], fresh.ppo.learning_rate)
        self.assertEqual(state["mirror_loss_coeff"], 1.0)
        torch.testing.assert_close(
            trainer.policy.log_std.exp(), torch.full((10,), 0.5), rtol=0, atol=0
        )

    def test_mirror_off_config_changes_only_loss_budget_cadence_and_labels(self):
        from wr2.locomotion.train import _apply_cli_overrides, parse_args

        fresh = load_training_config(FRESH)
        trial = load_training_config(MIRROR_OFF)
        self.assertEqual(
            trial,
            replace(
                fresh,
                version=trial.version,
                version_name=trial.version_name,
                ppo=replace(
                    fresh.ppo,
                    mirror_loss_coeff=0.0,
                    num_timesteps=200_000_000,
                    num_evals=11,
                ),
                output=replace(fresh.output, run_prefix="wr2_mirror_off"),
            ),
        )
        source = "results/wr2_walking/wr2_fresh_20261009_104213_seed0/checkpoints/000920043520.pt"
        with patch("sys.argv", [
            "train", "--config", str(MIRROR_OFF), "--restore-checkpoint", source
        ]):
            args = parse_args()
        self.assertEqual(args.restore_checkpoint, Path(source))
        self.assertIsNone(args.run_id)
        self.assertEqual(_apply_cli_overrides(trial, args), trial)

    def test_mirror_off_saves_11_evaluations_across_200m_additional_steps(self):
        from wr2.locomotion.rsl_rl_training import (
            evaluation_iterations,
            learning_iterations,
        )

        ppo = load_training_config(MIRROR_OFF).ppo
        iterations = learning_iterations(ppo.num_timesteps, ppo.num_envs, ppo.unroll_length)
        evaluations = evaluation_iterations(iterations, ppo.num_evals)
        self.assertEqual(iterations, 4883)
        self.assertEqual(len(evaluations) + 1, 11)
        self.assertEqual(max(evaluations), iterations)
        self.assertEqual(iterations * ppo.num_envs * ppo.unroll_length, 200_007_680)

    def test_mirror_on_checkpoint_restores_all_state_with_loss_disabled(self):
        source = self.make_trainer(coefficient=1.0)
        source.algorithm.optimizer.zero_grad()
        sum(parameter.sum() for parameter in source.policy.parameters()).backward()
        source.algorithm.optimizer.step()
        source.completed_iterations = 22462
        source.total_steps = 920_043_520
        source.algorithm.learning_rate = 1e-5
        for group in source.algorithm.optimizer.param_groups:
            group["lr"] = 1e-5
        with torch.no_grad():
            source.policy.log_std.fill_(np.log(0.0292))
        state = source.checkpoint_state()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.pt"
            torch.save(state, path)
            restored = self.make_trainer(coefficient=0.0, restore=path)
        self.assertIsNone(restored.algorithm.symmetry)
        self.assert_nested_equal(restored.policy.state_dict(), state["model_state_dict"])
        self.assert_nested_equal(
            restored.algorithm.optimizer.state_dict(), state["optimizer_state_dict"]
        )
        self.assertEqual(restored.completed_iterations, state["iteration"])
        self.assertEqual(restored.total_steps, state["total_steps"])
        self.assertEqual(restored.algorithm.learning_rate, state["learning_rate"])
        torch.testing.assert_close(torch.get_rng_state(), state["torch_rng_state"], rtol=0, atol=0)
        self.assertEqual(restored.checkpoint_state()["mirror_loss_coeff"], 0.0)

    def test_fresh_config_saves_51_evaluations_across_one_billion_steps(self):
        from wr2.locomotion.rsl_rl_training import (
            evaluation_iterations,
            learning_iterations,
        )

        ppo = load_training_config(FRESH).ppo
        iterations = learning_iterations(
            ppo.num_timesteps, ppo.num_envs, ppo.unroll_length
        )
        evaluations = sorted(evaluation_iterations(iterations, ppo.num_evals))
        self.assertEqual(len(evaluations) + 1, 51)
        self.assertEqual(evaluations[-1], iterations)
        steps_per_iteration = ppo.num_envs * ppo.unroll_length
        self.assertEqual(evaluations[0] * steps_per_iteration, 20_029_440)
        self.assertEqual(iterations * steps_per_iteration, 1_000_038_400)
        self.assertLess(
            iterations * steps_per_iteration - ppo.num_timesteps,
            steps_per_iteration,
        )

    def test_legacy_configs_default_to_disabled_and_invalid_coefficients_fail(self):
        raw = training_config_to_dict(load_training_config())
        del raw["ppo"]["mirror_loss_coeff"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(yaml.safe_dump(raw))
            self.assertEqual(load_training_config(path).ppo.mirror_loss_coeff, 0.0)
            for coefficient in (-1.0, float("inf"), float("nan")):
                raw["ppo"]["mirror_loss_coeff"] = coefficient
                path.write_text(yaml.safe_dump(raw))
                with (
                    self.subTest(coefficient=coefficient),
                    self.assertRaises(ValueError),
                ):
                    load_training_config(path)
            raw["ppo"].update(backend="brax", mirror_loss_coeff=1.0)
            path.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError, "requires the rsl_rl"):
                load_training_config(path)

    def test_torch_reflection_matches_jax_in_every_history_frame(self):
        import mujoco
        from wr2.locomotion.symmetry import joint_reflection

        model = mujoco.MjModel.from_xml_path(
            str(self.base_env.robot.directory / "scene_mjx.xml")
        )
        source, signs = joint_reflection(model, self.base_env.robot.actuator_names)
        for heading in (False, True):
            mirror = ActorMirror(
                source,
                signs,
                self.base_env._active_indices,
                heading=heading,
                device="cpu",
            )
            values = torch.arange(
                2 * 15 * mirror.frame_size, dtype=torch.float32
            ).reshape(2, -1)
            actual = mirror.observations(values)
            expected = mirror_actor(
                jp.asarray(values.numpy()),
                jp.asarray(source),
                jp.asarray(signs),
                self.base_env._active_indices,
                heading=heading,
            )
            np.testing.assert_array_equal(actual.numpy(), np.asarray(expected))
            torch.testing.assert_close(
                mirror.observations(actual), values, rtol=0, atol=0
            )
            actions = torch.arange(20, dtype=torch.float32).reshape(2, 10)
            torch.testing.assert_close(
                mirror.actions(mirror.actions(actions)), actions, rtol=0, atol=0
            )

    def test_callback_is_original_first_and_does_not_augment_the_critic(self):
        obs, actions = torch.ones(3, 855), torch.arange(30.0).reshape(3, 10)
        augmented_obs, augmented_actions = self.mirror(obs=obs, actions=actions)
        torch.testing.assert_close(augmented_obs[:3], obs)
        torch.testing.assert_close(augmented_obs[3:], self.mirror.observations(obs))
        torch.testing.assert_close(augmented_actions[:3], actions)
        torch.testing.assert_close(augmented_actions[3:], self.mirror.actions(actions))
        self.assertIsNone(self.mirror(obs=obs)[1])
        self.assertIsNone(self.mirror(actions=actions)[0])
        with self.assertRaisesRegex(ValueError, "critic"):
            self.mirror(obs=torch.ones(3, 1470), obs_type="critic")
        with self.assertRaisesRegex(ValueError, "15 raw"):
            self.mirror.observations(torch.ones(3, 57))

    def test_old_checkpoint_restores_actions_adam_lr_counters_and_rng_unchanged(self):
        torch.manual_seed(17)
        source = ActorCritic(
            855,
            1470,
            10,
            actor_hidden_dims=[8, 4],
            critic_hidden_dims=[8, 4],
            noise_std_type="log",
        )
        optimizer = torch.optim.Adam(source.parameters(), lr=2.25e-5)
        sum(p.sum() for p in source.parameters()).backward()
        optimizer.step()
        rng = torch.get_rng_state()
        checkpoint = {
            "backend": "rsl_rl",
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": 5862,
            "total_steps": 240107520,
            "learning_rate": 2.25e-5,
            "torch_rng_state": rng,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.pt"
            torch.save(checkpoint, path)
            control = self.make_trainer(coefficient=0.0, restore=path)
            treatment = self.make_trainer(restore=path)
        self.assert_nested_equal(treatment.policy.state_dict(), source.state_dict())
        self.assert_nested_equal(
            treatment.algorithm.optimizer.state_dict(), optimizer.state_dict()
        )
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertIsNone(control.algorithm.symmetry)
        self.assertFalse(treatment.algorithm.symmetry["use_data_augmentation"])
        self.assertEqual(treatment.algorithm.symmetry["mirror_loss_coeff"], 1.0)
        self.assertEqual(treatment.completed_iterations, 5862)
        self.assertEqual(treatment.total_steps, 240107520)
        self.assertEqual(treatment.algorithm.learning_rate, 2.25e-5)
        obs = torch.arange(855.0).repeat(4, 1) / 855
        torch.testing.assert_close(
            treatment.policy.act_inference(obs),
            control.policy.act_inference(obs),
            rtol=0,
            atol=0,
        )
        self.assertEqual(treatment.checkpoint_state()["mirror_loss_coeff"], 1.0)

    def test_native_rsl_loss_detaches_target_updates_actor_and_leaves_data_unchanged(
        self,
    ):
        torch.manual_seed(31)
        trainer = self.make_trainer()
        algorithm, policy = trainer.algorithm, trainer.policy
        obs, critic = torch.randn(4, 855), torch.randn(4, 1470)
        with torch.inference_mode():
            for _ in range(2):
                algorithm.act(obs, critic)
                algorithm.process_env_step(torch.zeros(4), torch.zeros(4), {})
            algorithm.compute_returns(critic)
            # Isolate the mirror term; returns/advantages are inference tensors.
            algorithm.storage.advantages.zero_()
        stored_actions = algorithm.storage.actions.clone()
        critic_before = {
            name: value.clone() for name, value in policy.critic.state_dict().items()
        }
        batch = algorithm.storage.observations.flatten(0, 1)

        def mse():
            return torch.nn.functional.mse_loss(
                policy.actor(self.mirror.observations(batch)),
                self.mirror.actions(policy.actor(batch)).detach(),
            )

        before = float(mse().detach())
        captured = []

        def hook(module, inputs, result):
            if result.requires_grad:
                result.retain_grad()
                captured.append(result)

        handle = policy.actor.register_forward_hook(hook)
        losses = algorithm.update()
        handle.remove()
        self.assertGreater(losses["symmetry"], 0)
        self.assertLess(float(mse().detach()), before)
        mirrored_batch = captured[-1]
        torch.testing.assert_close(mirrored_batch.grad[:8], torch.zeros(8, 10))
        self.assertGreater(float(mirrored_batch.grad[8:].abs().sum()), 0)
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in policy.parameters()
                if p.grad is not None
            )
        )
        self.assert_nested_equal(policy.critic.state_dict(), critic_before)
        torch.testing.assert_close(algorithm.storage.actions, stored_actions)
        metrics = trainer._training_metrics(losses)
        self.assertEqual(metrics["training/mirror_loss"], losses["symmetry"])
        self.assertAlmostEqual(
            metrics["training/total_loss"], losses["symmetry"], places=6
        )

    def test_normalized_observations_are_rejected_before_mirroring(self):
        trainer = self.make_trainer()
        config = replace(
            trainer.config, ppo=replace(trainer.config.ppo, normalize_observations=True)
        )
        with self.assertRaisesRegex(ValueError, "raw, unnormalized"):
            RSLPPOTrainer(
                trainer.env,
                config,
                output=Path("unused"),
                restore_checkpoint=None,
                progress_fn=lambda *args: None,
                policy_params_fn=lambda *args: None,
                evaluator=None,
            )

    def test_loss_logging_includes_the_weighted_native_mirror_term(self):
        trainer = self.make_trainer(coefficient=0.25)
        metrics = trainer._training_metrics(
            {"surrogate": 1.0, "value_function": 2.0, "entropy": 3.0, "symmetry": 4.0}
        )
        self.assertEqual(metrics["training/mirror_loss"], 4.0)
        self.assertEqual(metrics["training/mirror_loss_weighted"], 1.0)
        self.assertEqual(metrics["training/total_loss"], 2.0)

    def test_mirrored_residuals_produce_mirrored_clipped_physical_targets(self):
        from wr2.locomotion.symmetry import joint_reflection
        import mujoco

        robot = self.base_env.robot
        model = mujoco.MjModel.from_xml_path(str(robot.directory / "scene_mjx.xml"))
        source, signs = joint_reflection(model, robot.actuator_names)
        active = self.base_env._active_indices
        action = torch.linspace(-5, 5, 10)[None, :]

        def targets(values):
            target = robot.home_position_rad.copy()
            target[active] += 0.25 * values.numpy()[0]
            return np.clip(target, robot.lower_limit_rad, robot.upper_limit_rad)

        np.testing.assert_allclose(
            targets(self.mirror.actions(action)),
            targets(action)[source] * signs,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
