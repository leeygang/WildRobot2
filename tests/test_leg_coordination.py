"""Physical constraints for the matched sagittal-leg diagnostic."""

from pathlib import Path
import json

import mujoco
import numpy as np
import pytest

from wr2.tools.diagnose_leg_coordination import (
    SagittalCompensator,
    centered_reference_wave,
    floor_wrenches,
    physics_trace,
    placement_summary,
    reference_hip_targets,
)


NAMES = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee_pitch",
        "ankle_pitch",
        "ankle_roll",
    )
]


def test_waveform_has_zero_applied_mean_despite_reference_offsets():
    strength = np.asarray([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    reference = np.asarray(
        [
            [1000.0, -1000.0],
            [10.0, 22.0],
            [12.0, 21.0],
            [11.0, 19.0],
            [9.0, 20.0],
            [8.0, 18.0],
            [-1000.0, 1000.0],
            [500.0, 500.0],
        ]
    )
    wave = centered_reference_wave(reference, strength)
    np.testing.assert_allclose(wave.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.abs(wave).max(axis=0), 1)
    np.testing.assert_array_equal(wave[strength == 0], 0)
    np.testing.assert_allclose(
        centered_reference_wave(reference + [3.0, 7.0], strength), wave, atol=1e-12
    )


@pytest.fixture
def setup():
    path = Path(__file__).resolve().parents[1] / "wr2/descriptions/wr2/scene_mjx.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    qpos = model.key("walk_home").qpos.copy()
    ids = [model.actuator(name).trnid[0] for name in NAMES]
    home = qpos[model.jnt_qposadr[ids]].copy()
    # Paper walking uses the IK standing home, not the raw export keyframe.
    home[[0, 3, 4]] = np.deg2rad([-16.6, -40.2, -23.6])
    solver = SagittalCompensator(model, NAMES, home, 0.25, 0.1373)
    # Operating values observed in the selected policy, not the reference pose.
    target = home.copy()
    target[[0, 3, 4]] = np.deg2rad([-4.0, -33.0, -31.0])
    qpos[solver.qpos_indices] = target
    return model, qpos, target, solver


def pose(model, qpos, solver):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    w, x, y, z = data.xquat[solver.torso]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    heading = np.asarray([np.cos(yaw), np.sin(yaw)])
    normal = (
        data.geom_xmat[solver.geom].reshape(3, 3)[:, solver.normal_axis]
        * solver.normal_sign
    )
    return (
        data.site_xpos[solver.site].copy(),
        np.arctan2(normal[:2] @ heading, normal[2]),
        heading,
    )


def test_identity_preserves_targets(setup):
    _, qpos, target, solver = setup
    result, error = solver.solve(qpos, target, target[0])
    np.testing.assert_array_equal(result, target[solver.ka])
    assert error["height_error_m"] < 1e-12
    assert error["pitch_error_rad"] < 1e-12


@pytest.mark.parametrize("yaw", [0.0, 0.35])
def test_forward_hip_compensation_preserves_height_and_pitch(setup, yaw):
    model, qpos, target, solver = setup
    qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    original_position, original_pitch, heading = pose(model, qpos, solver)
    hip = target[0] - np.deg2rad(6)
    ka, _ = solver.solve(qpos, target, hip)
    compensated = qpos.copy()
    compensated[solver.qpos_indices[0]] = hip
    compensated[solver.qpos_indices[solver.ka]] = ka
    position, pitch, _ = pose(model, compensated, solver)
    assert abs(position[2] - original_position[2]) < 1e-6
    assert abs(pitch - original_pitch) < 1e-6
    assert (position[:2] - original_position[:2]) @ heading > 0.005
    assert np.all(ka >= solver.lower[solver.ka])
    assert np.all(ka <= solver.upper[solver.ka])
    unchanged = np.setdiff1d(np.arange(model.nq), solver.qpos_indices[[0, 3, 4]])
    np.testing.assert_array_equal(compensated[unchanged], qpos[unchanged])


def test_unreachable_compensation_is_rejected(setup):
    _, qpos, target, solver = setup
    solver.lower[solver.ka] = target[solver.ka] - 1e-8
    solver.upper[solver.ka] = target[solver.ka] + 1e-8
    with pytest.raises(ValueError, match="No reachable compensation"):
        solver.solve(qpos, target, target[0] - np.deg2rad(6))


@pytest.mark.parametrize("yaw", [0.0, 0.35])
def test_full_compensation_preserves_height_and_orientation(setup, yaw):
    model, qpos, target, solver = setup
    qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    position, _, heading = pose(model, qpos, solver)
    solver.data.qpos[:] = qpos
    mujoco.mj_kinematics(model, solver.data)
    original_rotation = solver.data.geom_xmat[solver.geom].copy()
    hip = target[0] - np.deg2rad(6)
    unknown, result, _ = solver.solve_full(qpos, target, hip)
    compensated = qpos.copy()
    compensated[solver.qpos_indices[0]] = hip
    compensated[solver.qpos_indices[unknown]] = result
    new_position, _, _ = pose(model, compensated, solver)
    assert abs(new_position[2] - position[2]) < 1e-6
    data = mujoco.MjData(model)
    data.qpos[:] = compensated
    mujoco.mj_kinematics(model, data)
    np.testing.assert_allclose(
        data.geom_xmat[solver.geom], original_rotation, atol=1e-6
    )
    assert (new_position[:2] - position[:2]) @ heading > 0.005
    assert np.all(result >= solver.lower[unknown])
    assert np.all(result <= solver.upper[unknown])
    unchanged = np.setdiff1d(
        np.arange(model.nq), solver.qpos_indices[np.r_[solver.hip, unknown]]
    )
    np.testing.assert_array_equal(compensated[unchanged], qpos[unchanged])


def test_full_reference_uses_controller_joint_limits(setup):
    model, qpos, target, solver = setup
    target[1] = solver.upper[1] + np.deg2rad(0.24)
    unknown, result, _ = solver.solve_full(qpos, target, target[0])
    expected = np.clip(target, solver.lower, solver.upper)
    np.testing.assert_allclose(result, expected[unknown], atol=1e-8)
    assert np.all(result <= solver.upper[unknown])


@pytest.mark.parametrize("yaw", [0.0, 0.35])
def test_right_compensation_moves_back_and_preserves_sole_pose(setup, yaw):
    model, qpos, target, _ = setup
    home = target.copy()
    home[[0, 3, 4, 6, 9, 10]] = np.deg2rad([-16.6, -40.2, -23.6, 16.6, 40.2, 23.6])
    target[[6, 9, 10]] = np.deg2rad([28, 37, 14])
    solver = SagittalCompensator(model, NAMES, home, 0.25, 0.1373, side="right")
    qpos[solver.qpos_indices] = target
    qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    position, _, heading = pose(model, qpos, solver)
    # pose() uses a separate MjData; evaluate the original orientation explicitly.
    solver.data.qpos[:] = qpos
    mujoco.mj_kinematics(model, solver.data)
    rotation = solver.data.geom_xmat[solver.geom].copy()
    hip = np.deg2rad(14)
    unknown, angles, _ = solver.solve_full(qpos, target, hip)
    assert unknown.tolist() == [8, 9, 10, 11]
    changed = qpos.copy()
    changed[solver.qpos_indices[solver.hip]] = hip
    changed[solver.qpos_indices[unknown]] = angles
    new_position, _, _ = pose(model, changed, solver)
    solver.data.qpos[:] = changed
    mujoco.mj_kinematics(model, solver.data)
    np.testing.assert_allclose(solver.data.geom_xmat[solver.geom], rotation, atol=1e-6)
    assert abs(new_position[2] - position[2]) < 1e-6
    assert (new_position[:2] - position[:2]) @ heading < -0.005
    np.testing.assert_array_equal(changed[solver.qpos_indices[:6]], qpos[solver.qpos_indices[:6]])


def test_full_leg_preserves_lateral_pose_at_recorded_fixed_roll_failure(setup):
    model, _, _, _ = setup
    path = Path(__file__).parent / "fixtures/bilateral_ik_limited_pose.json"
    case = json.loads(path.read_text())
    qpos, targets = np.asarray(case["qpos"]), np.asarray(case["targets"])
    fixed = SagittalCompensator(model, NAMES, case["home"], .25, case["height_scale_m"])
    with pytest.raises(ValueError, match="No reachable full-pose"):
        fixed.solve_full(qpos, targets, case["hip_target"])
    full = SagittalCompensator(
        model, NAMES, case["home"], .25, case["height_scale_m"], free_hip_roll=True
    )
    actual_hip, unknown, angles, error = full.solve_reachable_full(
        qpos, targets, case["hip_target"]
    )
    assert unknown.tolist() == [1, 2, 3, 4, 5]
    assert error["position_error_m"] < 1e-5
    assert error["orientation_error_rad"] < 1e-5
    assert abs(error["lateral_change_m"]) < 1e-5
    assert error["projection_fraction"] == 1
    assert actual_hip == case["hip_target"]
    assert np.all(angles >= full.lower[unknown])
    assert np.all(angles <= full.upper[unknown])


def test_projection_returns_original_pose_when_other_joint_travel_is_exhausted(setup):
    model, _, _, _ = setup
    case = json.loads((Path(__file__).parent / "fixtures/bilateral_ik_limited_pose.json").read_text())
    solver = SagittalCompensator(
        model, NAMES, case["home"], .25, case["height_scale_m"], free_hip_roll=True
    )
    targets = np.asarray(case["targets"])
    original = np.clip(targets, solver.lower, solver.upper)
    unknown = solver.full_unknown
    solver.lower[unknown] = np.maximum(solver.lower[unknown], original[unknown] - 1e-6)
    solver.upper[unknown] = np.minimum(solver.upper[unknown], original[unknown] + 1e-6)
    hip, _, angles, error = solver.solve_reachable_full(np.asarray(case["qpos"]), targets, case["hip_target"])
    assert error["projection_fraction"] < 1
    assert hip == original[solver.hip]
    assert error["position_error_m"] < 1e-5
    assert error["orientation_error_rad"] < 1e-5
    assert np.all(angles >= solver.lower[unknown])
    assert np.all(angles <= solver.upper[unknown])


def test_reference_targets_stay_bounded_and_preserve_inactive_tape():
    strength = np.asarray([0, 0.5, 1, 1, 1, 0.5, 0], dtype=float)
    original = np.full((7, 2), 0.22)
    reference = np.asarray([[10, -10], [.1, -.1], [.2, -.2], [.1, -.1], [0, 0], [.1, -.1], [10, -10]])
    targets, _, peaks = reference_hip_targets(original, reference, strength, -.25, .25, .2, [.1, -.1])
    assert (targets >= -.25).all() and (targets <= .25).all()
    assert (peaks > 0).all() and (peaks <= .2).all()
    np.testing.assert_array_equal(targets[strength == 0], original[strength == 0])


def test_bilateral_targets_preserve_combined_forward_hip_mean():
    strength = np.asarray([0, .5, 1, 1, 1, .5, 0], dtype=float)
    left = np.asarray([-.04, -.03, -.02, -.05, -.03, -.04, -.04])[:, None]
    right = np.asarray([.22, .21, .20, .24, .22, .21, .22])[:, None]
    reference = np.asarray([0, .1, .2, .1, 0, -.1, 0])[:, None]
    mean = ((right - left) * strength[:, None]).sum(axis=0) / (2 * strength.sum())
    new_left, _, _ = reference_hip_targets(left, reference, strength, -.25, .25, .05, -mean)
    new_right, _, _ = reference_hip_targets(right, -reference, strength, -.25, .25, .05, mean)
    np.testing.assert_allclose((new_right - new_left).sum(axis=0), (right - left).sum(axis=0), atol=1e-12)


def test_floor_wrenches_respect_endpoint_sign_and_yaw_balance():
    # Both feet push forward. Their lateral offsets give opposite yaw moments.
    geom1, geom2 = np.asarray([0, 2, 3]), np.asarray([1, 0, 4])
    positions = np.asarray([[0, .1, 0], [0, -.1, 0], [10, 10, 10]])
    forces = np.asarray([[10, 0, 20, 0, 0, .2], [-10, 0, -20, 0, 0, .2], [99] * 6], dtype=float)
    result = floor_wrenches(geom1, geom2, positions, forces, [1, 2], 0, np.zeros(3))
    np.testing.assert_allclose(result[:, :3], [[10, 0, 20], [10, 0, 20]])
    np.testing.assert_allclose(result[:, 5], [-.8, .8])
    np.testing.assert_allclose(result[:, 5].sum(), 0, atol=1e-6)
    shifted = floor_wrenches(geom1, geom2, positions + [1, 2, 3], forces, [1, 2], 0, np.asarray([1, 2, 3]))
    np.testing.assert_allclose(shifted, result, atol=2e-6)


@pytest.mark.parametrize("truncated", [False, True])
def test_placement_filters_short_and_truncated_landings(truncated):
    contact = np.zeros((18, 2), dtype=bool)
    contact[3:6, 0] = contact[10:14, 0] = contact[17:, 0] = True
    contact[7:10, 1] = contact[15:18, 1] = True
    previous = np.vstack([np.ones((1, 2), bool), contact[:-1]])
    positions = np.zeros((18, 2, 3))
    positions[:, 0, 0] = np.linspace(0, .05, 18)
    positions[:, 1, 0] = positions[:, 0, 0] + .01
    valid = np.ones(18, bool)
    valid[-1] = not truncated
    trace = {
        "valid": valid, "fall": np.zeros(18, bool), "contact": contact,
        "previous_contact": previous, "foot_geom_center_m": positions,
        "torso_quaternion": np.tile([1, 0, 0, 0], (18, 1)),
    }
    result = placement_summary(trace, 1, 18, 3, .02, .824)
    assert result["left"]["landings"] == 2
    assert result["left"]["lands_ahead"] == 0
    assert result["left"]["mean_lead_physical_mm"] == pytest.approx(-10)
    assert result["right"]["landings"] == (1 if truncated else 2)
    assert result["right"]["lands_ahead"] == result["right"]["landings"]
    assert result["left"]["mean_world_x_stride_physical_mm"] == pytest.approx(7 * .05 / 17 * 1000)


def test_substep_logging_reproduces_native_step_and_solver_reactions():
    from dataclasses import replace
    from functools import partial
    from unittest.mock import patch
    import jax
    import jax.numpy as jp
    from wr2.locomotion.configs import load_training_config
    from wr2.locomotion.walking_env import WR2WalkingEnv
    from wr2.reference.paper_walk import PaperWalkReference

    root = Path(__file__).resolve().parents[1]
    cfg = load_training_config(root / "wr2/locomotion/configs/ppo_walking_step.yaml")
    nominal = replace(cfg.environment, randomization=replace(cfg.environment.randomization, enabled=False))
    reference = partial(PaperWalkReference, command_vectors=[[.1 * nominal.normalization_scale, 0, 0]])
    with patch("wr2.locomotion.walking_env.PaperWalkReference", reference):
        env = WR2WalkingEnv(nominal, add_observation_noise=False, reward_discounting=cfg.ppo.discounting)
    state = jax.jit(env.reset)(jax.random.PRNGKey(7), jp.asarray([.1 * nominal.normalization_scale, 0, 0]))
    action = jp.full(env.action_size, .05)
    actual = jax.jit(env.step)(state, action)
    replay, trace = jax.jit(partial(physics_trace, env))(state, action)
    np.testing.assert_allclose(replay.q, actual.pipeline_state.q, atol=1e-6)
    np.testing.assert_allclose(replay.qd, actual.pipeline_state.qd, atol=1e-6)
    assert trace["torque_nm"].shape == (4, 12)
    assert np.max(trace["actuator_error_nm"]) < 1e-6
    assert np.max(trace["floor_vertical_error_n"]) < 1e-5
    assert np.all(trace["constraint_error"] <= 2e-5 * (1 + trace["constraint_scale"]))
    bound = np.asarray(env.sys.dof_frictionloss)[np.asarray(env._joint_qvel_indices)[np.asarray(env._active_indices)]]
    assert np.all(np.abs(trace["friction_torque_nm"]) <= bound + 1e-5)
