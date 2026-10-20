"""Check reference replay against training motors, the original cache and MuJoCo."""

import hashlib
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from wr2.locomotion.configs import load_training_config
from wr2.reference.paper_walk import PaperWalkReference, paper_command_grid
from wr2.sim.robot import RobotDescription
from wr2.tools.view_paper_reference import (
    PAPER_CONFIG,
    TB_PAPER_ROOT,
    draw_overlay,
    load_tb_reference,
    load_wr2_reference,
)


COMMANDS = [(0.1, 0, 0), (0, 0.05, 0), (0, 0, 0.5), (0, 0, 0)]


def test_training_command_grid_is_byte_identical_to_previous_implementation():
    scale = load_training_config(PAPER_CONFIG).environment.normalization_scale
    keys = paper_command_grid(scale)
    assert keys.shape == (227, 3)
    # Recorded from the original constructor before extracting its grid.
    assert hashlib.sha256(keys.tobytes()).hexdigest() == (
        "6a5d54bab1ba7ccb17adc0f58cbd08b664778430c3f58d86fc1969f7a51d79a5"
    )


@pytest.mark.parametrize("command", COMMANDS)
def test_wr2_replay_matches_training_reference_at_every_tick(command):
    replay = load_wr2_reference(PAPER_CONFIG, np.asarray(command))
    robot = RobotDescription.load(include_local_calibration=False)
    config = load_training_config(PAPER_CONFIG).environment
    reference = PaperWalkReference(
        replay.model,
        keyframe_id=replay.model.key("walk_home").id,
        actuator_names=robot.actuator_names,
        config=config,
        control_dt=replay.dt,
        normalization_scale=replay.scale,
        command_vectors=[replay.selected_command],
    )
    motors, stance, _ = jax.vmap(
        lambda t: reference.get_reference(
            t, jp.asarray(replay.command, dtype=jp.float32)
        )
    )(jp.arange(len(replay.qpos)) * replay.dt)
    addresses = [replay.model.joint(name).qposadr[0] for name in robot.actuator_names]
    np.testing.assert_allclose(replay.qpos[:, addresses], motors, atol=1e-6)
    np.testing.assert_array_equal(replay.stance, stance)
    verify_mujoco_frames(replay)


@pytest.mark.skipif(
    not TB_PAPER_ROOT.exists(), reason="Local frozen TB Paper assets required"
)
@pytest.mark.parametrize("command", COMMANDS)
def test_tb_replay_matches_original_joint_cache_including_wrap(command, monkeypatch):
    replay = load_tb_reference(TB_PAPER_ROOT, np.asarray(command))
    cache = Path(replay.source)
    with cache.open("rb") as stream:
        keys, _, masks, legs = pickle.load(stream)
    nearest = int(
        np.argmin(
            np.linalg.norm(np.asarray(keys, dtype=np.float32) - replay.command, axis=1)
        )
    )
    # Robot preserves config insertion order, with motors followed by driven joints.
    from toddlerbot.sim.robot import Robot

    monkeypatch.chdir(TB_PAPER_ROOT)
    robot = Robot("toddlerbot")
    addresses = [
        replay.model.joint(name).qposadr[0]
        for name in robot.joint_ordering
        if robot.joint_groups[name] == "leg"
    ]
    if np.linalg.norm(command) > 1e-6:
        indices = np.arange(1, 1000) % replay.lookup_length
        np.testing.assert_allclose(
            replay.qpos[1:, addresses], legs[nearest][indices], atol=1e-6
        )
        np.testing.assert_array_equal(replay.stance[1:], masks[nearest][indices])
    else:
        np.testing.assert_allclose(
            replay.qpos, np.broadcast_to(replay.qpos[0], replay.qpos.shape)
        )
        assert replay.stance.all()
    verify_mujoco_frames(replay)


def verify_mujoco_frames(replay):
    data = mujoco.MjData(replay.model)
    scene = mujoco.MjvScene(replay.model, maxgeom=100)
    for index in (0, 9, 18, 35, 959, 960, 999):
        data.qpos[:] = replay.qpos[index]
        mujoco.mj_forward(replay.model, data)
        assert np.isfinite(data.subtree_com).all()
        draw_overlay(scene, replay, data, index)
        assert 10 <= scene.ngeom <= scene.maxgeom
        assert np.isfinite(replay.optimal_cop[index]).all()
