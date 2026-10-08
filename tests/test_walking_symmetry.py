import unittest
from dataclasses import replace

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.walking_env import WR2WalkingEnv
from wr2.tools.verify_walking_symmetry import (
    joint_reflection,
    ablate_heading,
    conditional_mean,
    contact_impulses,
    mirror_actor,
    replay_control,
    total_momentum,
    verify_model_reference,
)


class WalkingSymmetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_training_config().environment
        config = replace(
            config, randomization=replace(config.randomization, enabled=False)
        )
        cls.env = WR2WalkingEnv(config, add_observation_noise=False)
        cls.model = mujoco.MjModel.from_xml_path(
            str(cls.env.robot.directory / "scene_mjx.xml")
        )
        cls.source, cls.signs = joint_reflection(
            cls.model, cls.env.robot.actuator_names
        )
        cls.active = np.asarray(cls.env._active_indices)

    def test_model_derived_signs_are_not_copied_from_wr1(self):
        np.testing.assert_allclose(self.signs, -1)
        np.testing.assert_array_equal(self.source[self.source], np.arange(17))
        np.testing.assert_allclose(self.signs * self.signs[self.source], 1)
        home = np.asarray(self.env._home_ctrl)
        np.testing.assert_allclose(home[self.source] * self.signs, home)

    def test_model_and_entire_reference_grid_are_nearly_mirrored(self):
        result = verify_model_reference(self.env, self.model, self.source, self.signs)
        self.assertEqual(result["home_error_deg"]["max_abs"], 0)
        self.assertEqual(result["limit_error_deg"]["max_abs"], 0)
        self.assertLess(result["fk_position_error_mm"]["max_abs"], 0.5)
        self.assertLess(result["reference_joint_error_deg"]["max_abs"], 0.1)
        self.assertLess(result["reference_foot_error_mm"]["max_abs"], 0.1)
        self.assertLess(result["leg_mass_matrix_relative_error"]["max_abs"], 0.002)

    def test_actor_transform_matches_a_physically_reflected_state(self):
        env = self.env
        state = env.reset(jax.random.PRNGKey(0))
        q = state.pipeline_state.q.at[env._joint_qpos_indices].add(
            jp.linspace(-0.1, 0.1, 17)
        )
        q = q.at[3:7].set(
            jp.asarray([0.98, 0.08, 0.06, 0.16])
            / jp.linalg.norm(jp.asarray([0.98, 0.08, 0.06, 0.16]))
        )
        qd = jp.linspace(-0.2, 0.2, env.sys.nv)
        original = env.pipeline_init(q, qd)
        reflected_q = q.at[env._joint_qpos_indices].set(
            q[env._joint_qpos_indices][self.source] * self.signs
        )
        reflected_q = reflected_q.at[:3].set(q[:3] * jp.asarray([1, -1, 1]))
        reflected_q = reflected_q.at[3:7].set(q[3:7] * jp.asarray([1, -1, 1, -1]))
        reflected_qd = qd.at[env._joint_qvel_indices].set(
            qd[env._joint_qvel_indices][self.source] * self.signs
        )
        reflected_qd = reflected_qd.at[:3].set(qd[:3] * jp.asarray([1, -1, 1]))
        reflected_qd = reflected_qd.at[3:6].set(qd[3:6] * jp.asarray([-1, 1, -1]))
        reflected = env.pipeline_init(reflected_q, reflected_qd)
        command, action = jp.asarray([0.1, 0.02, 0.03]), jp.linspace(-0.1, 0.1, 10)
        active_source = np.asarray(
            [list(self.active).index(int(self.source[j])) for j in self.active]
        )

        def observe(pipeline, a, c, phase, reference):
            return env._observation(
                pipeline,
                a,
                c,
                phase,
                state.info["rng"],
                state.info["imu_state"],
                state.info["backlash"],
                jp.asarray([1, 0]),
                jp.asarray([1, 0]),
                reference_heading=reference,
            )[0]["state"]

        frame = observe(original, action, command, 0.7, 0.1)
        physical_mirror = observe(
            reflected,
            action[active_source] * self.signs[self.active],
            command * jp.asarray([1, -1, -1]),
            0.7 + np.pi,
            -0.1,
        )
        feature_mirror = mirror_actor(
            frame, jp.asarray(self.source), jp.asarray(self.signs), self.active
        )
        np.testing.assert_allclose(feature_mirror, physical_mirror, atol=1e-6)

    def test_mirror_all_actor_frames_and_involution(self):
        values = jp.asarray(
            np.random.default_rng(0).normal(size=(2, 855)), dtype=jp.float32
        )

        def mirror(x):
            return mirror_actor(
                x, jp.asarray(self.source), jp.asarray(self.signs), self.active
            )

        mirrored = jax.jit(mirror)(values)
        np.testing.assert_allclose(mirror(mirrored), values)
        original, reflected = (
            np.asarray(values).reshape(2, 15, 57),
            np.asarray(mirrored).reshape(2, 15, 57),
        )
        np.testing.assert_allclose(reflected[..., :2], -original[..., :2])
        np.testing.assert_allclose(reflected[..., 2], original[..., 2])
        np.testing.assert_allclose(reflected[..., 3:5], -original[..., 3:5])
        np.testing.assert_allclose(
            reflected[..., 49:52], original[..., 49:52] * [-1, 1, -1]
        )
        np.testing.assert_allclose(
            reflected[..., 52:55], original[..., 52:55] * [1, -1, 1]
        )
        np.testing.assert_allclose(
            reflected[..., 55:57], original[..., 55:57] * [-1, 1]
        )
        for start in (5, 22):
            np.testing.assert_allclose(
                reflected[..., start : start + 17],
                original[..., start + self.source] * self.signs,
            )

    def test_heading_ablation_preserves_every_other_history_feature(self):
        values = jp.arange(2 * 855, dtype=jp.float32).reshape(2, 855)
        hidden = jax.jit(ablate_heading)(values)
        original, changed = values.reshape(2, 15, 57), hidden.reshape(2, 15, 57)
        np.testing.assert_array_equal(changed[..., :55], original[..., :55])
        np.testing.assert_array_equal(changed[..., 55], 0)
        np.testing.assert_array_equal(changed[..., 56], 1)
        np.testing.assert_array_equal(ablate_heading(hidden), hidden)

    def test_substep_replay_preserves_training_physics(self):
        state = jax.jit(self.env.reset)(jax.random.PRNGKey(0))
        action = jp.linspace(-0.2, 0.2, self.env.action_size)
        pipeline, trace = jax.jit(lambda s, a: replay_control(self.env, s, a))(
            state, action
        )
        after = jax.jit(self.env.step)(state, action)
        np.testing.assert_allclose(pipeline.q, after.pipeline_state.q, atol=1e-7)
        np.testing.assert_allclose(pipeline.qd, after.pipeline_state.qd, atol=1e-7)
        self.assertEqual(trace["impulse_world_ns"].shape, (2, 3))
        self.assertLess(float(jp.max(jp.abs(trace["momentum_residual_ns"][:2]))), 0.002)
        # Control has a one-step delay: only the following step applies action.
        second, _ = jax.jit(lambda s, a: replay_control(self.env, s, a))(after, -action)
        expected = jax.jit(self.env.step)(after, -action)
        np.testing.assert_allclose(second.q, expected.pipeline_state.q, atol=1e-7)

    def test_total_momentum_uses_inertial_center_velocity(self):
        from types import SimpleNamespace

        model = SimpleNamespace(
            body_rootid=jp.asarray([0, 1]), body_mass=jp.asarray([0.0, 2.0])
        )
        data = SimpleNamespace(
            xipos=jp.asarray([[0, 0, 0], [0, 1, 0]]),
            subtree_com=jp.zeros((2, 3)),
            cvel=jp.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]),
        )
        np.testing.assert_allclose(total_momentum(model, data), [-2, 0, 0])

    def test_unsupported_foot_has_no_slip_measurement(self):
        self.assertIsNone(
            conditional_mean(np.asarray([2.0, 4.0]), np.asarray([False, False]))
        )
        self.assertEqual(
            conditional_mean(np.asarray([2.0, 4.0]), np.asarray([True, False])), 2.0
        )

    def test_impulses_integrate_all_substeps_without_cancelling_braking_metric(self):
        force = jp.asarray(
            [
                [[2, 0, 10], [0, 0, 0]],
                [[-1, 0, 10], [0, 0, 0]],
                [[2, 0, 10], [0, 0, 0]],
                [[-1, 0, 10], [0, 0, 0]],
            ],
            dtype=jp.float32,
        )
        net, propulsive, braking = contact_impulses(force, 0.005)
        np.testing.assert_allclose(net, [[0.01, 0, 0.2], [0, 0, 0]], atol=1e-8)
        np.testing.assert_allclose(propulsive, [0.02, 0])
        np.testing.assert_allclose(braking, [0.01, 0])


if __name__ == "__main__":
    unittest.main()
