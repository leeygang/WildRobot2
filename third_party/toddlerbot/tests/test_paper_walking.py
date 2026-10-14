"""Paper configuration and measurement contracts; no hardware access."""

import os

os.environ["USE_JAX"] = "true"

import unittest
import copy
import contextlib
import io
import json
import tempfile
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gin
import jax
import jax.numpy as jnp
import numpy as np

from toddlerbot.locomotion.evaluate_xy import (
    CommandVmapWrapper,
    GROUPS,
    MEASUREMENTS,
    command_and_groups,
    set_command,
    summarize,
)
from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.ppo_config import PPOConfig


class PaperWalkingTest(unittest.TestCase):
    def tearDown(self):
        gin.clear_config()

    def test_paper_configuration(self):
        gin.parse_config_file("toddlerbot/locomotion/walk.gin")
        gin.parse_config_file("toddlerbot/locomotion/paper.gin")
        ppo = PPOConfig()
        cfg = MJXConfig()
        self.assertEqual((ppo.num_timesteps, ppo.num_envs), (300_000_000, 1024))
        self.assertEqual((ppo.batch_size, ppo.num_minibatches), (256, 4))
        self.assertEqual((ppo.unroll_length, ppo.num_updates_per_batch), (20, 4))
        self.assertEqual((ppo.learning_rate, ppo.discounting), (1e-4, 0.97))
        self.assertEqual((ppo.entropy_cost, ppo.clipping_epsilon), (5e-4, 0.2))
        self.assertEqual(ppo.policy_hidden_layer_sizes, (512, 256, 128))
        self.assertEqual(ppo.value_hidden_layer_sizes, (512, 256, 128))
        self.assertEqual((ppo.num_evals, ppo.num_eval_envs), (16, 128))
        self.assertTrue(ppo.evaluate_xy)
        expected = {
            "torso_quat": 1,
            "lin_vel_xy": 5,
            "lin_vel_z": 1,
            "ang_vel_xy": 2,
            "ang_vel_z": 5,
            "leg_motor_pos": 5,
            "feet_contact": 1,
            "feet_air_time": 500,
            "feet_clearance": 0.05,
            "feet_distance": 1,
            "feet_slip": 0.05,
            "align_ground": 1,
            "stand_still": 1,
            "torso_roll": 0.5,
            "torso_pitch": 0.5,
            "collision": 0.1,
            "leg_action_rate": 0.05,
            "leg_action_acc": 0.05,
            "motor_torque": 0.01,
            "energy": 0.05,
            "survival": 10,
        }
        actual = {
            key: value for key, value in vars(cfg.reward_scales).items() if value != 0
        }
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(cfg.sim.timestep * cfg.action.n_frames, 0.02)
        self.assertEqual(cfg.action.cycle_time, 0.72)

    def test_native_motor_imitation_and_terminal_penalty(self):
        env = SimpleNamespace(
            q_start_idx=0,
            leg_motor_indices=jnp.array([0, 1]),
            ref_start_idx=0,
            leg_ref_indices=jnp.array([0, 1]),
            reset_steps=1000,
        )
        state = SimpleNamespace(q=jnp.array([0.1, -0.2]))
        info = {"state_ref": jnp.zeros(2), "done": jnp.asarray(False), "step": 0}
        self.assertAlmostEqual(
            float(MJXEnv._reward_leg_motor_pos(env, state, info, None)), -0.025
        )
        self.assertEqual(float(MJXEnv._reward_survival(env, state, info, None)), 0)
        info["done"] = jnp.asarray(True)
        self.assertEqual(float(MJXEnv._reward_survival(env, state, info, None)), -1)

    def test_numpy_and_jax_foot_interpolation(self):
        from toddlerbot.algorithms import zmp_walk
        from toddlerbot.utils import array_utils

        env = SimpleNamespace(control_dt=0.02, foot_to_com_y=0.05, footstep_height=0.05)
        for backend in (np, jnp):
            with (
                self.subTest(backend=backend.__name__),
                patch.object(zmp_walk, "np", backend),
                patch.object(array_utils, "USE_JAX", backend is jnp),
            ):
                times = backend.asarray([0, 0.12, 0.36, 0.48], dtype=backend.float32)
                footsteps = backend.asarray(
                    [
                        [0, 0, 0, 2],
                        [0, 0, 0, 2],
                        [0.06, -0.05, 0.3, 1],
                    ],
                    dtype=backend.float32,
                )
                _, _, position, orientation, _, stance = (
                    zmp_walk.ZMPWalk.compute_foot_trajectories(env, times, footsteps)
                )
                np.testing.assert_allclose(
                    position[[6, 12, 18], 0], [0, 0.03, 0.06], atol=1e-6
                )
                np.testing.assert_allclose(
                    orientation[[6, 12, 18], 2], [0, 0.15, 0.3], atol=1e-6
                )
                np.testing.assert_array_equal(stance[6:18, 1], np.zeros(12))

    def test_exponential_polynomial_under_jit(self):
        from toddlerbot.algorithms.zmp_planner import ExpPlusPPoly, PPoly

        polynomial = PPoly(jnp.array([[[1.0], [10.0]]]), jnp.array([0.0, 1.0, 2.0]))
        curve = ExpPlusPPoly(
            jnp.ones((1, 1)), jnp.zeros((1, 1)), jnp.array([[2.0, 20.0]]), polynomial
        )
        values = jax.jit(jax.vmap(curve.value))(
            jnp.array([-0.1, 0.5, 0.999, 1.0, 1.5, 2.1])
        )
        np.testing.assert_allclose(values[:, 0], [3, 3, 3, 30, 30, 30])

    def test_transition_sample_counts(self):
        command, masks = jax.jit(jax.vmap(command_and_groups, in_axes=(0, None, None)))(
            jnp.arange(1000), jnp.array([0.1, 0.0]), jnp.asarray(True)
        )
        np.testing.assert_array_equal(
            np.asarray(masks).sum(axis=0), [1000, 450, 550, 50, 50]
        )
        np.testing.assert_allclose(command[149, 5:7], [0, 0])
        np.testing.assert_allclose(command[150, 5:7], [0.1, 0])
        np.testing.assert_allclose(command[450, 5:7], [0, 0])
        np.testing.assert_allclose(command[600, 5:7], [0.1, 0])
        np.testing.assert_allclose(command[850, 5:7], [0, 0])
        _, fixed = jax.vmap(command_and_groups, in_axes=(0, None, None))(
            jnp.arange(1000), jnp.array([0, -0.05]), jnp.asarray(False)
        )
        np.testing.assert_array_equal(
            np.asarray(fixed).sum(axis=0), [1000, 0, 1000, 0, 0]
        )

    def test_unequal_episode_lengths_and_empty_categories(self):
        counts = np.zeros((2, len(GROUPS)), dtype=np.int32)
        sums = np.zeros((2, len(GROUPS), len(MEASUREMENTS)), dtype=np.float32)
        counts[:, 0] = [4, 2]
        sums[:, 0, 2] = [0.4, 0.8]
        sums[:, 0, 4] = [4 * 0.2**2, 2 * 0.1**2]
        groups, episodes = summarize(sums, counts)
        self.assertAlmostEqual(groups["all"]["xy_error_m_s"]["mean"], 0.25)
        self.assertAlmostEqual(groups["all"]["leg_motor_rmse_rad"]["mean"], 0.15)
        self.assertEqual(groups["all"]["sample_count"], 6)
        self.assertEqual(groups["start"]["episode_count"], 0)
        self.assertIsNone(groups["start"]["xy_error_m_s"]["mean"])
        self.assertEqual(episodes["start"]["xy_error_m_s"], [None, None])

    def test_command_changes_preserve_history_and_reference(self):
        from brax.envs.base import State

        obs = jnp.arange(30, dtype=jnp.float32).reshape(2, 15)
        reference = jnp.array([8.0])
        state = State(
            None,
            obs,
            obs + 100,
            jnp.zeros(2),
            jnp.zeros(2),
            info={"state_ref": reference},
        )
        command = jnp.zeros((2, 8)).at[:, 5:8].set(jnp.array([0.1, -0.05, 0]))
        updated = set_command(state, command)
        np.testing.assert_allclose(updated.obs[:, 2:5], command[:, 5:8])
        np.testing.assert_array_equal(updated.obs[:, 5:], state.obs[:, 5:])
        np.testing.assert_array_equal(updated.obs[:, :2], state.obs[:, :2])
        np.testing.assert_array_equal(
            updated.privileged_obs[:, 5:], state.privileged_obs[:, 5:]
        )
        self.assertIs(updated.info["state_ref"], reference)

    def test_explicit_commands_match_native_reset_and_step(self):
        from brax.envs.wrappers.training import VmapWrapper
        from toddlerbot.locomotion.walk_env import WalkEnv
        from toddlerbot.sim.robot import Robot

        gin.parse_config_file("toddlerbot/locomotion/walk.gin")
        env = WalkEnv("walk", Robot("toddlerbot"), MJXConfig())
        native_env = copy.copy(env)
        command = jnp.zeros(8).at[6].set(0.05)
        native_env._sample_command = lambda rng, last_command=None: command
        native = VmapWrapper(native_env)
        explicit = CommandVmapWrapper(
            env, lambda sys: (sys, jax.tree.map(lambda _: None, sys), jnp.zeros((2, 0)))
        )
        keys = jax.random.split(jax.random.PRNGKey(12102), 2)
        commands = jnp.broadcast_to(command, (2, 8))
        expected = jax.jit(native.reset)(keys)
        actual = jax.jit(explicit.reset)(keys, commands)
        action = jnp.zeros((2, env.action_size))
        expected = jax.jit(native.step)(expected, action)
        actual = jax.jit(explicit.step)(set_command(actual, commands), action)
        for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)

    def test_nonfinite_checkpoint_is_retained_before_abort(self):
        from toddlerbot.locomotion import train_mjx

        Params = namedtuple("Params", "policy value")
        for bad_parameters in (True, False):
            with (
                self.subTest(bad_parameters=bad_parameters),
                tempfile.TemporaryDirectory() as folder,
            ):
                params = (
                    jnp.zeros(1),
                    Params(
                        {"w": jnp.array([np.nan if bad_parameters else 0.0])},
                        {"w": jnp.zeros(1)},
                    ),
                )

                def fake_train(**kwargs):
                    kwargs["progress_fn"](
                        40,
                        {
                            "eval/episode_reward": 0.0 if bad_parameters else np.nan,
                        },
                    )
                    kwargs["policy_params_fn"](40, None, params)

                env = SimpleNamespace(
                    cfg=MJXConfig(),
                    robot=SimpleNamespace(name="toddlerbot"),
                    add_domain_rand=False,
                )
                with (
                    patch.object(train_mjx.ppo, "train", side_effect=fake_train),
                    patch.object(train_mjx.wandb, "init"),
                    patch.object(train_mjx.wandb, "log"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaises(FloatingPointError):
                        train_mjx.train(env, env, None, PPOConfig(), folder, "")
                checkpoint = Path(folder) / "40"
                self.assertTrue((checkpoint / "policy").is_file())
                self.assertFalse((Path(folder) / "policy").exists())
                finite = json.loads((checkpoint / "finite.json").read_text())
                self.assertEqual(finite["parameters_finite"], not bad_parameters)
                self.assertEqual(finite["metrics_finite"], bad_parameters)
                provenance = json.loads((Path(folder) / "provenance.json").read_text())
                self.assertEqual(
                    provenance["brax_revision"],
                    "14147806f6c25724ba830ba36764df3243889b0d",
                )
                if Path("SOURCE.json").is_file():
                    self.assertEqual(
                        provenance["upstream"], json.loads(Path("SOURCE.json").read_text())
                    )


if __name__ == "__main__":
    unittest.main()
