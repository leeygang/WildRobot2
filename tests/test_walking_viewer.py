"""Keep replay meshes at contact-time poses for WR2 and TB's two foot pairs."""

import json

import mujoco
import numpy as np
import pytest

from wr2.tools import view_walking


@pytest.mark.parametrize(
    "robot,pair_count,complete_trace",
    [("wr2", 1, True), ("wr2", 1, False), ("toddlerbot", 2, True)],
)
def test_replay_preserves_cached_bodies_pair_colors_and_missing_fields(
    tmp_path, monkeypatch, robot, pair_count, complete_trace,
):
    feet = 2 * pair_count
    bodies = []
    for index in range(feet):
        y = .2 if index % 2 == 0 else -.2
        name = (
            f"{'left' if index == 0 else 'right'}_foot_collision"
            if robot == "wr2" else f"foot_{index}_collision"
        )
        bodies.append(
            f'<body pos="{.03 * index} {y} .05">'
            f'<joint name="foot_{index}_slide" type="slide" axis="0 1 0"/>'
            f'<geom name="{name}" type="box" size=".04 .03 .02" '
            'pos=".01 .015 0" euler="0 0 .2" group="3"/>'
            f'<geom name="foot_{index}_visual" type="sphere" '
            'size=".01" pos=".03 -.02 .01" group="2"/></body>'
        )
    xml = (
        '<mujoco><compiler angle="radian"/><worldbody><body>'
        '<freejoint/><geom name="torso_visual" type="sphere" size=".05"/>'
        + ''.join(bodies) + '</body></worldbody></mujoco>'
    )
    model_path = tmp_path / 'toddlerbot/descriptions/toddlerbot/toddlerbot_scene.xml'
    model_path.parent.mkdir(parents=True)
    model_path.write_text(xml)
    monkeypatch.setattr(view_walking, 'MODEL_PATH', model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    foot_ids = [model.geom(
        f"{'left' if i == 0 else 'right'}_foot_collision"
        if robot == 'wr2' else f'foot_{i}_collision'
    ).id for i in range(feet)]

    integrated = mujoco.MjData(model)
    integrated.qpos[2] = .3
    mujoco.mj_kinematics(model, integrated)
    centers, rotations, gaps, expected = [], [], [], []
    for frame in range(2):
        recorded = mujoco.MjData(model)
        recorded.qpos[:] = integrated.qpos
        recorded.qpos[0] += .01 * (frame + 1)
        recorded.qpos[3:7] = [np.cos(.1), 0., 0., np.sin(.1)]
        # A different foot position makes an integrated-pose distance incorrect.
        recorded.qpos[7] += .05 + .01 * frame
        mujoco.mj_kinematics(model, recorded)
        centers.append(recorded.geom_xpos[foot_ids].copy())
        rotations.append(recorded.geom_xmat[foot_ids].reshape(feet, 3, 3).copy())
        gaps.append([
            mujoco.mj_geomDistance(model, recorded, *foot_ids[i:i+2], 1., None)
            for i in range(0, feet, 2)
        ])
        expected.append((recorded.geom_xpos.copy(), recorded.geom_xmat.copy()))
    centers, rotations, gaps = map(np.asarray, (centers, rotations, gaps))
    trace = {
        'qpos': np.tile(integrated.qpos, (2, 1)), 'valid': np.ones(2, bool),
        'foot_geom_center_m': centers if robot == 'wr2' else centers.reshape(2, pair_count, 2, 3),
        'foot_geom_rotation': rotations if robot == 'wr2' else rotations.reshape(2, pair_count, 2, 3, 3),
        'foot_self_force_n': np.asarray([0., 2.]) if robot == 'wr2' else np.asarray([[0., 0.], [0., 2.]]),
        'delivered_rad': np.zeros((2, 2)), 'actual_rad': np.zeros((2, 2)),
        'contact': np.asarray([[True, False], [True, True]]),
        'foot_vertical_force_n': np.asarray([[12., 0.], [9., 3.]]),
    }
    scale = .8 if robot == 'wr2' else 1.
    command = np.asarray([.1 * scale, -.05 * scale, 0.])
    if complete_trace:
        trace['command_m_s'] = np.tile(command, (2, 1))
        trace['foot_gap_m'] = gaps[:, 0] if robot == 'wr2' else gaps
    metadata = {
        'active_joint_names': ['left_hip_roll', 'right_hip_roll'],
        'control_period_s': .02, 'normalization_scale': scale,
        'collision_pairs': [[model.geom(i).name for i in foot_ids[j:j+2]] for j in range(0, feet, 2)],
        'results': [{'policy': 'synthetic', 'mode': 'learned', 'command_TB_m_s': [.1, -.05]}],
    }
    trace_path = tmp_path / 'synthetic_learned.npz'
    np.savez(trace_path, **trace)
    trace_path.with_name('summary.json').write_text(json.dumps(metadata))
    _, replay, replay_model, replay_feet = view_walking.load_recording(
        trace_path, robot, tmp_path
    )
    np.testing.assert_allclose(replay['command_m_s'], np.tile(command, (2, 1)))
    np.testing.assert_allclose(replay['foot_gap_m'], gaps.min(axis=1), atol=1e-10)
    np.testing.assert_allclose(replay['support_age_s'], [[.02, 0.], [.04, .02]])
    np.testing.assert_array_equal(replay['foot_vertical_force_n'], trace['foot_vertical_force_n'])
    data = mujoco.MjData(replay_model)
    scene = mujoco.MjvScene(replay_model, maxgeom=100)
    for frame in range(2):
        view_walking.apply_trace_frame(replay_model, data, replay, frame, replay_feet)
        np.testing.assert_array_equal(data.qpos, integrated.qpos)
        for i, foot in enumerate(replay_feet):
            visual = replay_model.geom(f'foot_{i}_visual').id
            for geom in (foot, visual):
                np.testing.assert_allclose(data.geom_xpos[geom], expected[frame][0][geom], atol=1e-10)
                np.testing.assert_allclose(data.geom_xmat[geom], expected[frame][1][geom], atol=1e-10)
        np.testing.assert_allclose(data.geom_xpos[0], integrated.geom_xpos[0])
        scene.ngeom = 0
        view_walking.draw_foot_boxes(scene, replay_model, replay, frame, replay_feet)
        assert scene.ngeom == 12 * feet
        for i in range(feet):
            red = frame == 1 and (robot == 'wr2' or i >= 2)
            color = [1., .15, .1, 1.] if red else [.1, .7, 1., 1.]
            np.testing.assert_allclose(scene.geoms[12 * i].rgba, color)


@pytest.mark.parametrize('robot', ['wr2', 'toddlerbot'])
def test_native_hip_directions_are_outward_positive(robot):
    path = (
        view_walking.MODEL_PATH if robot == 'wr2' else
        view_walking.TB_PAPER_ROOT / 'toddlerbot/descriptions/toddlerbot/toddlerbot_scene.xml'
    )
    if not path.is_file():
        pytest.skip('Local TB Paper model assets required')
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    feet = (
        ['left_foot_collision', 'right_foot_collision'] if robot == 'wr2' else
        ['ank_roll_link_collision', 'ank_roll_link_2_collision']
    )
    signs = [-1, 1 if robot == 'wr2' else -1]
    for side, geom_name, sign, outward_y in zip(('left', 'right'), feet, signs, (1, -1), strict=True):
        data.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, data)
        before = data.geom_xpos[model.geom(geom_name).id, 1]
        address = model.joint(f'{side}_hip_roll').qposadr[0]
        data.qpos[address] += sign * 1e-5
        mujoco.mj_kinematics(model, data)
        assert outward_y * (data.geom_xpos[model.geom(geom_name).id, 1] - before) > 0
