import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jp
import jax
import numpy as np

from wr2.tools.diagnose_walking import (
    contact_point_velocity,
    foot_contact_samples,
    rotate_heading,
    sample_step,
    summarize,
)


class WalkingDiagnosticsTest(unittest.TestCase):
    def test_target_trace_uses_delayed_request_not_current_action(self):
        env = SimpleNamespace(
            config=SimpleNamespace(action_delay_steps=1),
            _ctrl_lower=jp.asarray([-1.0]),
            _ctrl_upper=jp.asarray([0.0]),
            _max_command_step_rad=0.1,
            _root_link_index=0,
            _joint_qpos_indices=jp.asarray([7]),
            _gait_phase_increment=0.1,
            _policy_target=lambda a: a,
            _quantize_target=lambda a: a,
            _kinematic_observation=lambda p: (None, None, jp.zeros(3), None),
        )
        before = SimpleNamespace(
            info={
                "previous_action": jp.asarray([0.2]),
                "actuator_target_bias": jp.asarray([0.05]),
                "previous_target": jp.asarray([-0.2]),
                "reference_heading": 0.0,
                "command": jp.asarray([0.1, 0, 0]),
                "gait_phase": 0.0,
                "command_age_steps": 0,
            }
        )
        pipeline = SimpleNamespace(
            x=SimpleNamespace(pos=jp.zeros((1, 3)), rot=jp.asarray([[1.0, 0, 0, 0]])),
            q=jp.zeros(8),
        )
        after = SimpleNamespace(
            pipeline_state=pipeline,
            metrics={
                "fall": 0.0,
                "nonfinite_state": 0.0,
                "actuator_torque_rms_nm_per_step": 0.0,
            },
        )
        with patch(
            "wr2.tools.diagnose_walking.foot_contact_samples",
            return_value=(jp.zeros((2, 3)), jp.zeros(2), jp.zeros(2)),
        ):
            trace = sample_step(env, before, after, jp.asarray([-0.5]))
        self.assertAlmostEqual(
            float(trace["requested_target_deg"][0]), np.rad2deg(0.25), places=5
        )
        self.assertAlmostEqual(
            float(trace["applied_target_deg"][0]), np.rad2deg(-0.1), places=5
        )
        self.assertTrue(bool(trace["clip_upper"][0]))
        self.assertTrue(bool(trace["slew_limited"][0]))
        self.assertEqual(float(trace["command_m_s"]), float(before.info["command"][0]))

    def test_heading_rotation_keeps_reference_and_joint_controller_state(self):
        from dataclasses import replace
        from wr2.locomotion.configs import load_training_config
        from wr2.locomotion.walking_env import WR2WalkingEnv, torso_heading

        config = load_training_config().environment
        env = WR2WalkingEnv(
            replace(config, randomization=replace(config.randomization, enabled=False)),
            add_observation_noise=False,
        )
        state = env.reset(jax.random.PRNGKey(0))
        # Set a world translation speed and a body-local angular speed.
        qd = state.pipeline_state.qd.at[:6].set(jp.asarray([1.0, 0, 0, 0.1, 0, 0]))
        state = state.replace(
            pipeline_state=env.pipeline_init(state.pipeline_state.q, qd)
        )
        turned = rotate_heading(env, state, np.pi / 2)
        self.assertAlmostEqual(
            float(torso_heading(turned.pipeline_state.x.rot[env._root_link_index])),
            np.pi / 2,
            places=6,
        )
        np.testing.assert_allclose(turned.pipeline_state.qd[:3], [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(turned.pipeline_state.qd[3:], qd[3:])
        np.testing.assert_allclose(
            turned.pipeline_state.q[7:], state.pipeline_state.q[7:]
        )
        np.testing.assert_allclose(turned.obs["state"][57:], state.obs["state"][57:])
        np.testing.assert_allclose(turned.obs["state"][55:57], [1, 0], atol=1e-6)
        for name in (
            "reference_heading",
            "previous_action",
            "previous_target",
            "gait_phase",
        ):
            np.testing.assert_allclose(turned.info[name], state.info[name])
        for old, new in zip(
            jax.tree.leaves(state.info["imu_state"]),
            jax.tree.leaves(turned.info["imu_state"]),
            strict=True,
        ):
            np.testing.assert_allclose(new, old)
        # All body distances and floor clearance are invariant under this yaw.
        before = np.asarray(state.pipeline_state.x.pos)
        after = np.asarray(turned.pipeline_state.x.pos)
        np.testing.assert_allclose(after[:, 2], before[:, 2], atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(after - after[0], axis=1),
            np.linalg.norm(before - before[0], axis=1),
            atol=1e-6,
        )

    def test_contact_velocity_includes_rotation_and_not_swing_origin_motion(self):
        # A pivoting foot origin moves while the stationary contact does not.
        velocity = contact_point_velocity(
            jp.asarray([1.0, 0.0, 0.0]),
            jp.asarray([0.0, 0.0, 1.0]),
            jp.asarray([0.0, 1.0, 0.0]),
            jp.zeros(3),
        )
        np.testing.assert_allclose(velocity, 0)

    def test_force_sign_floor_pairing_and_support_slip_denominator(self):
        env = SimpleNamespace(
            sys=None,
            _floor_geom_id=0,
            _foot_geom_ids=jp.asarray([1, 2]),
            _foot_link_indices=jp.asarray([0, 1]),
            config=SimpleNamespace(contact_force_threshold_n=1.0),
        )
        contact = SimpleNamespace(
            dist=jp.asarray([-0.01, -0.01, -0.01, -0.01]),
            geom1=jp.asarray([0, 2, 0, 1]),
            geom2=jp.asarray([1, 0, 1, 2]),
            efc_address=jp.asarray([0, 3, -1, 6]),
            pos=jp.zeros((4, 3)),
        )
        pipeline = SimpleNamespace(
            _impl=SimpleNamespace(contact=contact),
            x=SimpleNamespace(pos=jp.zeros((2, 3))),
            xd=SimpleNamespace(
                vel=jp.asarray([[0.02, 0, 0], [0.03, 0, 0]]), ang=jp.zeros((2, 3))
            ),
        )
        forces = jp.asarray([[2, 0, 10], [-3, 0, -15], [99, 99, 99], [99, 99, 99]])
        with patch(
            "mujoco.mjx._src.support.contact_force",
            side_effect=lambda m, d, i, w: forces[i],
        ):
            force, slip, counts = foot_contact_samples(env, pipeline)
        np.testing.assert_allclose(force, [[2, 0, 10], [3, 0, 15]])
        np.testing.assert_allclose(slip, [0.02, 0.03])
        np.testing.assert_array_equal(counts, [1, 1])

    def test_summary_uses_applied_commands_and_excludes_terminated_samples(self):
        length = 240
        trace = {
            "valid": np.ones((length, 1), bool),
            "command_m_s": np.full((length, 1), 0.1),
            "command_age_steps": np.full((length, 1), 30),
            "velocity_torso_m_s": np.zeros((length, 1, 3)),
            "heading_error_deg": np.full((length, 1), 5.0),
            "position_m": np.zeros((length, 1, 3)),
            "fall": np.zeros((length, 1)),
            "nonfinite": np.zeros((length, 1)),
            "torque_rms_nm": np.zeros((length, 1)),
            "phase_rad": np.zeros((length, 1)),
            "support_point_count": np.ones((length, 1, 2)),
            "support_point_slip_m_s": np.zeros((length, 1, 2)),
            "foot_force_path_n": np.zeros((length, 1, 2, 3)),
            "clip_lower": np.zeros((length, 1, 1), bool),
            "clip_upper": np.zeros((length, 1, 1), bool),
            "requested_target_deg": np.full((length, 1, 1), -2.0),
            "applied_target_deg": np.zeros((length, 1, 1)),
            "joint_position_deg": np.full((length, 1, 1), 3.0),
        }
        trace["command_m_s"][:50] = 0
        trace["clip_lower"][10] = True
        trace["clip_lower"][55] = True
        trace["command_age_steps"][50:75] = np.arange(25)[:, None]
        trace["heading_error_deg"][160:196] = 1
        trace["fall"][220] = 1
        trace["valid"][221:] = False
        result = summarize(trace, ("knee",), 0.02)
        self.assertEqual(result["mean_episode_length"], 221)
        self.assertEqual(result["fall_rate"], 1)
        self.assertAlmostEqual(result["heading_recovery_s_per_run"][0], 0.2)
        self.assertEqual(result["phase"][0]["samples"], 21)
        self.assertAlmostEqual(
            result["clipping"]["standing"]["joints"]["knee"]["lower_fraction"], 1 / 50
        )
        event = result["clipping"]["start_0p5s"]["joints"]["knee"]["first_event"]
        self.assertEqual(event["requested_deg"], -2)
        self.assertEqual(event["applied_deg"], 0)
        self.assertEqual(event["actual_deg"], 3)
        self.assertAlmostEqual(event["time_s"], 1.1)
        # Early termination cannot be mislabeled as post-injection recovery.
        trace["valid"][80:] = False
        early = summarize(trace, ("knee",), 0.02)
        self.assertEqual(early["heading_recovery_s_per_run"], [None])
        self.assertEqual(early["post_injection_path_xy_m_per_run"], [None])
        self.assertEqual(early["final_3s_signed_heading_deg_per_run"], [None])


if __name__ == "__main__":
    unittest.main()
