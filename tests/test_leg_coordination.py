"""Physical constraints for the matched sagittal-leg diagnostic."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from wr2.tools.diagnose_leg_coordination import (
    SagittalCompensator,
    centered_reference_wave,
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
