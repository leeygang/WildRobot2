import unittest

import jax
import jax.numpy as jp
import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.tools.investigate_walking import (
    ablation_config,
    evaluation_keys,
    summarize_trace,
)


class WalkingInvestigationTest(unittest.TestCase):
    def test_replay_keys_match_evaluators_first_split(self):
        model_keys, unroll_key = evaluation_keys(707, 128)
        wrap_key, eval_key = jax.random.split(jax.random.PRNGKey(707))
        _, expected = jax.random.split(eval_key)
        np.testing.assert_array_equal(unroll_key, expected)
        np.testing.assert_array_equal(model_keys, jax.random.split(wrap_key, 128))
        # A selected episode must retain its original index, not key zero.
        self.assertFalse(np.array_equal(model_keys[37], model_keys[0]))

    def test_ablations_separate_noise_from_dynamics_without_changing_timing(self):
        original = load_training_config().environment
        for case, dynamics, noise in (
            ("full", True, True),
            ("dynamics", True, False),
            ("noise", False, True),
            ("nominal", False, False),
        ):
            config, actual_noise = ablation_config(original, case)
            self.assertEqual(config.randomization.enabled, dynamics)
            self.assertEqual(actual_noise, noise)
            self.assertEqual(config.rewards, original.rewards)
            self.assertEqual(config.action_delay_steps, original.action_delay_steps)
            if not dynamics:
                for name in (
                    "friction_scale",
                    "damping_scale",
                    "armature_scale",
                    "frictionloss_scale",
                    "body_mass_scale",
                ):
                    self.assertEqual(getattr(config.randomization, name), (1.0, 1.0))

    def test_first_terminal_step_is_included_but_autoresets_are_not(self):
        trace = {
            "episode_done": np.array([[0, 0], [1, 0], [1, 1]]),
            "fall": np.array([[0, 0], [1, 0], [1, 0]]),
            "nonfinite": np.zeros((3, 2)),
            "command_m_s": np.array([[0, 0], [0.1, 0.1], [0.1, 0.1]]),
            "velocity_torso_m_s": np.zeros((3, 2, 3)),
            "heading_error_deg": np.zeros((3, 2)),
            "phase_rad": np.zeros((3, 2)),
            "position_m": np.full((3, 2, 3), 0.19),
            "clip_lower": np.zeros((3, 2, 2), dtype=bool),
            "clip_upper": np.zeros((3, 2, 2), dtype=bool),
            "reward_components": np.ones((3, 2, 1)),
        }
        # Inactive shoulder clipping must not affect the leg action gate.
        trace["clip_upper"][:, :, 1] = True
        result = summarize_trace(trace, 0.02, np.array([0]))
        self.assertEqual(result["avg_episode_length"], 2.5)
        self.assertEqual(result["fall_count"], 1)
        self.assertEqual(result["failures"][0]["episode_index"], 0)
        self.assertEqual(result["failures"][0]["time_s"], 0.04)
        self.assertEqual(result["active_target_clipping_fraction"], 0)
        self.assertAlmostEqual(result["walking_speed_mae_m_s"], 0.1)
        self.assertEqual(result["reward_component_returns"], [2.5])

    def test_brax_autoreset_keeps_pre_reset_diagnostic_state(self):
        from brax.envs.base import Env, State
        from brax.envs import training

        class TerminalEnv(Env):
            @property
            def observation_size(self):
                return 1

            @property
            def action_size(self):
                return 1

            @property
            def backend(self):
                return "test"

            def reset(self, rng):
                return State(
                    pipeline_state=jp.array([0.3]),
                    obs=jp.zeros(1),
                    reward=jp.array(0.0),
                    done=jp.array(0.0),
                    metrics={"fall": jp.array(0.0)},
                    info={"diagnostic": jp.array([0.3])},
                )

            def step(self, state, action):
                return state.replace(
                    pipeline_state=jp.array([0.19]),
                    done=jp.array(1.0),
                    metrics={"fall": jp.array(1.0)},
                    info={**state.info, "diagnostic": jp.array([0.19])},
                )

        env = training.wrap(TerminalEnv(), episode_length=3, action_repeat=1)
        state = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
        after = env.step(state, jp.zeros((1, 1)))
        np.testing.assert_allclose(after.pipeline_state, [[0.3]])
        np.testing.assert_allclose(after.info["diagnostic"], [[0.19]])
        self.assertEqual(float(after.info["episode_done"][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
