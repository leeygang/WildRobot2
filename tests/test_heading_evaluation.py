"""Coordinate, episode accounting and normalization oracles for heading eval."""

import numpy as np
import pytest

from wr2.locomotion.heading_evaluation import heading_rollout, summarize_heading
from wr2.tools.evaluate_heading import parse_args


def trace_from_path(xy, heading, dt=.02, initial_heading=0.):
    xy = np.asarray(xy, float)
    heading = np.asarray(heading, float)
    steps = len(xy) - 1
    initial = np.r_[xy[0], .3, np.cos(initial_heading/2), 0, 0, np.sin(initial_heading/2)]
    pose = np.column_stack([xy[1:], np.full(steps, .3), np.cos(heading[1:]/2),
                            np.zeros((steps, 2)), np.sin(heading[1:]/2)])
    velocity = np.zeros((steps, 3))
    velocity[:, 0] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    angular = np.zeros((steps, 3))
    angular[:, 2] = np.diff(heading) / dt
    return {
        "initial_root_pose": initial[None], "root_pose": pose[:, None],
        "physics_quaternion": pose[:, None, 3:7],
        "root_delta_m": np.diff(xy, axis=0)[:, None],
        "body_velocity_m_s": velocity[:, None],
        "body_angular_velocity_rad_s": angular[:, None],
        "command_physical": np.tile([.1, 0, 0], (steps, 1, 1)),
        "reward_terms": np.zeros((steps, 1, 3)),
        "total_reward": np.zeros((steps, 1)),
        "valid": np.ones((steps, 1), bool), "fall": np.zeros((steps, 1), bool),
        "nonfinite": np.zeros((steps, 1), bool),
    }


def test_initial_random_heading_defines_forward_axes():
    xy = np.column_stack([np.zeros(6), np.linspace(0, .01, 6)])
    trace = trace_from_path(xy, np.full(6, np.pi/2), initial_heading=np.pi/2)
    row = summarize_heading(trace, .02, 1., 0)["per_episode"][0]["full"]
    assert row["reference_forward_m"] == pytest.approx(.01)
    assert row["reference_lateral_m"] == pytest.approx(0, abs=1e-12)
    assert row["net_heading_change_deg"] == pytest.approx(0)
    assert row["body_forward_TB_m_s"] == pytest.approx(.1)


def test_quarter_circle_separates_path_forward_and_lateral_progress():
    angle = np.linspace(0, np.pi/2, 201)
    trace = trace_from_path(np.column_stack([np.sin(angle), 1-np.cos(angle)]), angle)
    row = summarize_heading(trace, .02, 1., 0)["per_episode"][0]["full"]
    assert row["reference_forward_m"] == pytest.approx(1.)
    assert row["reference_lateral_m"] == pytest.approx(1.)
    assert row["root_net_distance_m"] == pytest.approx(np.sqrt(2))
    assert row["root_path_m"] == pytest.approx(np.pi/2, rel=1e-5)
    assert row["net_heading_change_deg"] == pytest.approx(90)
    assert row["mean_abs_heading_error_deg"] == pytest.approx(45.225)


def test_unwrap_preserves_turn_through_pi():
    angle = np.linspace(np.deg2rad(170), np.deg2rad(200), 6)
    trace = trace_from_path(np.column_stack([np.arange(6)*.002, np.zeros(6)]), angle,
                           initial_heading=angle[0])
    row = summarize_heading(trace, .02, 1., 0)["per_episode"][0]["full"]
    assert row["final_heading_error_deg"] == pytest.approx(30)


def test_size_time_normalization_and_exact_warmup_boundary():
    xy = np.column_stack([np.arange(101)*.001, np.zeros(101)])
    trace = trace_from_path(xy, np.zeros(101))
    report = summarize_heading(trace, .02, .5, .1)
    full = report["per_episode"][0]["full"]
    steady = report["per_episode"][0]["post_warmup"]
    assert full["duration_TB_s"] == pytest.approx(4)
    assert full["body_forward_TB_m_s"] == pytest.approx(.1)
    assert full["reference_forward_TB_m"] == pytest.approx(.4)
    assert steady["duration_physical_s"] == pytest.approx(1.94)
    assert steady["reference_forward_m"] == pytest.approx(.097)


def test_finite_terminal_is_retained_and_invalid_padding_is_excluded():
    xy = np.column_stack([np.arange(6)*.002, np.zeros(6)])
    trace = trace_from_path(xy, np.zeros(6))
    trace["valid"][3:] = False
    trace["fall"][2] = True
    for name in ["root_pose", "root_delta_m", "body_velocity_m_s", "reward_terms"]:
        trace[name][3:] = np.nan
    report = summarize_heading(trace, .02, 1., 0)
    assert report["fall_count"] == 1
    assert report["completed_count"] == 0
    assert report["per_episode"][0]["full"]["reference_forward_m"] == pytest.approx(.006)
    assert report["per_episode"][0]["valid_steps"] == 3


def test_cli_has_retained_checkpoints_and_native_tb_default():
    common = ["--run-dir", "run", "--output", "out"]
    assert parse_args(common).checkpoint_steps == [120053760, 200089600, 300134400]
    assert parse_args(common + ["--tb-root", "paper"]).checkpoint_steps == [200089600]
    with pytest.raises(SystemExit):
        parse_args(common + ["--num-envs", "0"])


def test_rollout_counts_one_fall_and_one_invalid_transition_without_reset():
    import jax
    import jax.numpy as jp
    from brax import base
    from brax.envs.base import State

    class Episodes:
        def reset(self, keys, command):
            q = jp.zeros((2, 7)).at[:, 3].set(1.)
            p = base.State(q, jp.zeros((2, 6)),
                           base.Transform.zero((2, 1)), base.Motion.zero((2, 1)), None)
            return State(p, {"state": jp.zeros((2, 1))}, jp.zeros(2), jp.zeros(2),
                         {"fall": jp.zeros(2)}, {"step": jp.zeros(2, int), "command": command})

        def step(self, state, action):
            tick = state.info["step"] + 1
            fall = (jp.arange(2) == 0) & (tick == 2)
            invalid = (jp.arange(2) == 1) & (tick == 3)
            q = state.pipeline_state.q.at[:, 0].add(.002)
            q = q.at[:, 0].set(jp.where(invalid, jp.nan, q[:, 0]))
            return state.replace(pipeline_state=state.pipeline_state.replace(q=q),
                                 done=fall.astype(float), metrics={"fall": fall.astype(float)},
                                 info={**state.info, "step": tick})

    def policy(obs, key):
        return jp.zeros((2, 1)), {}

    def rewards(before, after):
        return jp.zeros((2, 3)), after.metrics["fall"] > 0, before.info["command"]

    raw = heading_rollout(Episodes(), policy, jax.random.split(jax.random.PRNGKey(0), 2),
                          jp.tile(jp.asarray([.1, 0, 0]), (2, 1)), 6, 0, rewards)
    trace = jax.device_get(raw)
    np.testing.assert_array_equal(trace["valid"].sum(0), [2, 2])
    np.testing.assert_array_equal(trace["fall"].sum(0), [1, 0])
    np.testing.assert_array_equal(trace["nonfinite"].sum(0), [0, 1])
    report = summarize_heading(trace, .02, 1., 0)
    assert report["fall_count"] == report["nonfinite_count"] == 1
    assert report["completed_count"] == 0
