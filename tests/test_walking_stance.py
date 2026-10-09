import unittest

import jax
import jax.numpy as jp
import numpy as np

from brax.envs.base import Env, State
from brax.training import acting

from wr2.tools.check_walking_stance import (
    assert_matching_prefix,
    foot_geometry,
    stance_summary,
    switched_unroll,
)


class HistoryEnv(Env):
    """Small deterministic plant with delayed commands and hidden carry."""

    @property
    def observation_size(self):
        return 3

    @property
    def action_size(self):
        return 1

    @property
    def backend(self):
        return "test"

    def reset(self, rng):
        state = State(
            pipeline_state=jp.array([0.3]),
            obs={"state": jp.zeros(3)},
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics={},
            info={
                "delay": jp.zeros(1),
                "warmstart": jp.ones(1),
                "sensor_rng": rng,
                "phase": jp.array(0.0),
                "episode_done": jp.array(0.0),
            },
        )
        state.info["diagnostic"] = self.trace(state, state, jp.zeros(1))
        return state

    def trace(self, before, after, action):
        return {
            "action": action,
            "position": after.pipeline_state,
            "pre_history": before.obs["state"],
            "pre_warmstart": before.info["warmstart"],
            "pre_delay": before.info["delay"],
            "pre_sensor_rng": before.info["sensor_rng"],
            "pre_phase": before.info["phase"],
        }

    def step(self, before, action):
        key, next_key = jax.random.split(before.info["sensor_rng"])
        position = (
            before.pipeline_state
            + 0.02 * before.info["delay"]
            + 0.01 * before.info["warmstart"]
            + 0.001 * jax.random.uniform(key)
        )
        after = before.replace(
            pipeline_state=position,
            obs={"state": jp.concatenate([position, before.obs["state"][:2]])},
            reward=position[0],
            info={
                **before.info,
                "delay": action,
                "warmstart": position,
                "sensor_rng": next_key,
                "phase": before.info["phase"] + 10.0,
            },
        )
        after.info["diagnostic"] = self.trace(before, after, action)
        return after


def constant_policy(value):
    return lambda obs, key: (jp.array([value]), {})


class WalkingStanceTest(unittest.TestCase):
    def test_signed_geometry_retains_which_foot_is_ahead_after_yaw_rotation(self):
        trace = {
            "qpos": np.array(
                [[[0, 0, 0, 1, 0, 0, 0]], [[0, 0, 0, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]]]
            ),
            "foot_position_m": np.array(
                [[[[-0.024, 0.09, 0], [0, 0, 0]]], [[[-0.09, -0.024, 0], [0, 0, 0]]]]
            ),
        }
        dx, dy = foot_geometry(trace)
        np.testing.assert_allclose(dx, -0.024, atol=1e-10)
        np.testing.assert_allclose(dy, 0.09, atol=1e-10)

    def test_self_handover_matches_brax_unroll_and_key_stream(self):
        env = HistoryEnv()
        key = jax.random.PRNGKey(707)
        actor = constant_policy(1.0)
        reference, trajectory = acting.generate_unroll(
            env,
            env.reset(key),
            actor,
            key,
            unroll_length=6,
            extra_fields=("diagnostic", "episode_done"),
        )
        final, final_key, extras = switched_unroll(
            env, env.reset(key), actor, actor, key, 2, 6
        )
        for expected, actual in zip(
            jax.tree.leaves(reference), jax.tree.leaves(final), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        assert_matching_prefix(
            extras["diagnostic"], trajectory.extras["state_extras"]["diagnostic"], 6
        )
        for _ in range(6):
            key = jax.random.split(key)[1]
        np.testing.assert_array_equal(final_key, key)

    def test_crossed_actor_keeps_history_phase_rng_warmstart_and_delay(self):
        env = HistoryEnv()
        key = jax.random.PRNGKey(707)
        old, new = constant_policy(1.0), constant_policy(2.0)
        _, _, reference = switched_unroll(env, env.reset(key), old, old, key, 2, 6)
        _, _, crossed = switched_unroll(env, env.reset(key), old, new, key, 2, 6)
        before, after = reference["diagnostic"], crossed["diagnostic"]
        assert_matching_prefix(after, before, 2)
        for name in (
            "pre_history",
            "pre_phase",
            "pre_sensor_rng",
            "pre_warmstart",
            "pre_delay",
        ):
            np.testing.assert_array_equal(after[name][2], before[name][2])
        # The new action is emitted now, but the first physical step still
        # applies the delayed donor action. Changing only qpos would fail this.
        np.testing.assert_array_equal(after["position"][2], before["position"][2])
        np.testing.assert_array_equal(after["action"][2], [2.0])
        np.testing.assert_array_equal(after["pre_delay"][2], [1.0])
        np.testing.assert_array_equal(after["pre_delay"][3], [2.0])
        self.assertFalse(np.array_equal(after["position"][3], before["position"][3]))

    def test_prefix_check_rejects_hidden_trace_difference(self):
        with self.assertRaises(AssertionError):
            assert_matching_prefix(
                {"history": np.array([0, 1])}, {"history": np.array([0, 2])}, 2
            )

    def test_restart_window_excludes_prior_falls_and_autoreset_samples(self):
        trace = {
            "qpos": np.tile([0, 0, 0, 1, 0, 0, 0], (1000, 2, 1)),
            "foot_position_m": np.tile([[-0.024, 0.09, 0], [0, 0, 0]], (1000, 2, 1, 1)),
            "foot_force_path_n": np.tile([[0, 0, 4], [0, 0, 6]], (1000, 2, 1, 1)),
            "phase_rad": np.ones((1000, 2)),
            "episode_done": np.zeros((1000, 2)),
            "fall": np.zeros((1000, 2)),
            "nonfinite": np.zeros((1000, 2)),
        }
        trace["episode_done"][160:, 0] = 1
        trace["fall"][160:, 0] = 1
        trace["foot_position_m"][550:, 0, 0, 0] = 100
        first, second = (stance_summary(trace, start, 0.02) for start in (150, 600))
        self.assertEqual(first["alive_at_start_count"], 2)
        self.assertEqual(first["fall_count_next_5s"], 1)
        self.assertEqual(second["alive_at_start_count"], 1)
        self.assertEqual(second["fall_count_next_5s"], 0)
        self.assertAlmostEqual(second["pre_start_signed_left_minus_right_x_mm"], -24)
        self.assertAlmostEqual(second["pre_start_left_load_share"], 0.4)


if __name__ == "__main__":
    unittest.main()
