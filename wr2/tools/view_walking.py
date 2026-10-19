"""Replay a recorded WR2 stance diagnostic in the interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "descriptions/wr2/scene_mjx.xml"
BOX_CORNERS = np.asarray(list(itertools.product((-1, 1), repeat=3)))
BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if np.count_nonzero(BOX_CORNERS[i] != BOX_CORNERS[j]) == 1
]


def apply_trace_frame(model, data, trace, index, foot_ids) -> None:
    """Show integrated body pose and feet at their recorded contact sampling instant."""
    data.qpos[:] = trace["qpos"][index]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    for side, box in enumerate(foot_ids):
        local_rotation = np.empty(9)
        mujoco.mju_quat2Mat(local_rotation, model.geom_quat[box])
        body_rotation = (
            trace["foot_geom_rotation"][index, side] @ local_rotation.reshape(3, 3).T
        )
        body_position = (
            trace["foot_geom_center_m"][index, side]
            - body_rotation @ model.geom_pos[box]
        )
        for geom in np.flatnonzero(model.geom_bodyid == model.geom_bodyid[box]):
            mujoco.mju_quat2Mat(local_rotation, model.geom_quat[geom])
            data.geom_xpos[geom] = body_position + body_rotation @ model.geom_pos[geom]
            data.geom_xmat[geom] = (
                body_rotation @ local_rotation.reshape(3, 3)
            ).reshape(9)


def draw_foot_boxes(scene, model, trace, index, foot_ids) -> None:
    """Outline cached collision boxes; red means recorded model self-contact."""
    color = (
        [1.0, 0.15, 0.1, 1.0]
        if trace["foot_self_force_n"][index] > 0.1
        else [0.1, 0.7, 1.0, 1.0]
    )
    for side, geom in enumerate(foot_ids):
        vertices = (BOX_CORNERS * model.geom_size[geom]) @ trace["foot_geom_rotation"][
            index, side
        ].T
        vertices += trace["foot_geom_center_m"][index, side]
        for first, second in BOX_EDGES:
            line = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.asarray(color),
            )
            mujoco.mjv_connector(
                line, mujoco.mjtGeom.mjGEOM_LINE, 2.0, vertices[first], vertices[second]
            )
            scene.ngeom += 1


def view_trace(args: argparse.Namespace) -> int:
    metadata = json.loads(args.trace.with_name("summary.json").read_text())
    with np.load(args.trace) as recording:
        trace = {name: recording[name] for name in recording.files}
    count = int(trace["valid"].sum())
    if count == 0 or not np.array_equal(
        np.flatnonzero(trace["valid"]), np.arange(count)
    ):
        raise ValueError("The recording must have a nonempty valid sample prefix")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if (
        trace["qpos"].shape[1] != model.nq
        or not np.isfinite(trace["qpos"][:count]).all()
    ):
        raise ValueError("Recorded poses must be finite and match the WR2 model")
    data = mujoco.MjData(model)
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        for side in ("left", "right")
    ]
    hips = [
        metadata["active_joint_names"].index(f"{side}_hip_roll")
        for side in ("left", "right")
    ]
    dt = metadata["control_period_s"]
    playback = {"time": 0.0, "paused": args.paused, "boxes": True, "view": 1}

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            playback["paused"] = not playback["paused"]
        elif keycode in (ord("["), ord("]")):
            playback["paused"] = True
            step = -1 if keycode == ord("[") else 1
            playback["time"] = (
                np.clip(round(playback["time"] / dt) + step, 0, count - 1) * dt
            )
        elif keycode in (ord("r"), ord("R")):
            playback["time"] = 0.0
        elif keycode in (ord("c"), ord("C")):
            playback["boxes"] = not playback["boxes"]
        elif keycode in (ord("1"), ord("2"), ord("3"), ord("4")):
            playback["view"] = keycode - ord("0")

    print(
        f"Replaying {args.trace.name}: {count} ticks, {count * dt:.2f} physical seconds"
    )
    print(
        "Space pause; [ / ] step; R restart; C collision outlines; 1 front / 2 side / 3 top / 4 perspective"
    )
    print(
        "Blue outlines = clear boxes; red = recorded box self-contact. CAD feet use recorded box/body frames."
    )
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = (
                0  # Cached outlines replace the solid collision boxes.
            )
            viewer.opt.geomgroup[2] = 1  # Keep the CAD surfaces visible.
            viewer.opt.sitegroup[:] = 0
            viewer.cam.distance = 0.8
            viewer.cam.lookat[2] = 0.22
        previous_time = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            if not playback["paused"]:
                playback["time"] = (
                    playback["time"]
                    + min(now - previous_time, 0.1) * args.playback_speed
                ) % (count * dt)
            previous_time = now
            index = min(int(playback["time"] / dt), count - 1)
            with viewer.lock():
                data.time = (index + 1) * dt
                apply_trace_frame(model, data, trace, index, feet)
                viewer.cam.lookat[:2] = data.qpos[:2]
                if playback["view"]:
                    viewer.cam.azimuth, viewer.cam.elevation = {
                        1: (0, -10),
                        2: (90, -10),
                        3: (90, -90),
                        4: (135, -20),
                    }[playback["view"]]
                    playback["view"] = 0
                viewer.user_scn.ngeom = 0
                if playback["boxes"]:
                    draw_foot_boxes(viewer.user_scn, model, trace, index, feet)
            target = np.rad2deg(trace["delivered_rad"][index, hips])
            actual = np.rad2deg(trace["actual_rad"][index, hips])
            force = float(trace["foot_self_force_n"][index])
            command = trace["command_m_s"][index]
            support = trace["contact"][index].astype(int)
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Recorded rollout\nWR2 time (s)\nCommand XY (m/s)\nFloor support L/R\nBox self-force (N)\nBox gap (mm)\nHip target L/R (deg)\nHip actual L/R (deg)",
                    f"{args.trace.stem}\n{(index + 1) * dt:.2f}\n{command[0]:+.4f}, {command[1]:+.4f}\n{support[0]} / {support[1]}\n{force:.3f}\n{trace['foot_gap_m'][index] * 1000:+.3f}\n{target[0]:+.2f} / {target[1]:+.2f}\n{actual[0]:+.2f} / {actual[1]:+.2f}",
                )
            )
            viewer.sync()
            time.sleep(0.005)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="NPZ trace with its diagnostic summary.json alongside",
    )
    parser.add_argument("--playback-speed", type=float, default=0.25)
    parser.add_argument("--paused", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("--playback-speed must be finite and positive")
    return view_trace(args)


if __name__ == "__main__":
    raise SystemExit(main())
