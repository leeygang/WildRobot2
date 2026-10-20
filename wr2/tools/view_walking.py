"""Replay WR2 or ToddlerBot Paper policy recordings in the MuJoCo viewer."""

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
TB_PAPER_ROOT = Path(__file__).resolve().parents[2] / ".worktrees/tb-paper-stance"
BOX_CORNERS = np.asarray(list(itertools.product((-1, 1), repeat=3)))
BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if np.count_nonzero(BOX_CORNERS[i] != BOX_CORNERS[j]) == 1
]


def load_recording(trace_path: Path, robot: str, tb_root: Path):
    """Normalize recorded foot pairs while preserving their cached world frames."""
    metadata = json.loads(trace_path.with_name("summary.json").read_text())
    fields = (
        "qpos", "valid", "foot_geom_center_m", "foot_geom_rotation",
        "foot_self_force_n", "foot_gap_m", "command_m_s", "delivered_rad",
        "actual_rad", "contact", "foot_vertical_force_n",
    )
    with np.load(trace_path) as recording:
        trace = {name: recording[name] for name in fields if name in recording.files}
    count = int(trace["valid"].sum())
    if count == 0 or not np.array_equal(
        np.flatnonzero(trace["valid"]), np.arange(count)
    ):
        raise ValueError("The recording must have a nonempty valid sample prefix")
    if robot == "wr2":
        model_path = MODEL_PATH
        pair_names = [[f"{side}_foot_collision" for side in ("left", "right")]]
    else:
        model_path = tb_root.expanduser() / "toddlerbot/descriptions/toddlerbot/toddlerbot_scene.xml"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing TB Paper model: {model_path}. Set --tb-root to the "
                "frozen Paper checkout with its original model/mesh assets."
            )
        pair_names = metadata["collision_pairs"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if trace["qpos"].shape != (len(trace["valid"]), model.nq):
        raise ValueError(f"Recorded poses do not match the {robot} model")
    pairs = np.asarray([[model.geom(name).id for name in pair] for pair in pair_names])
    feet = pairs.flatten()
    ticks = len(trace["valid"])
    trace["foot_geom_center_m"] = trace["foot_geom_center_m"].reshape(ticks, len(feet), 3)
    trace["foot_geom_rotation"] = trace["foot_geom_rotation"].reshape(ticks, len(feet), 3, 3)
    forces = trace["foot_self_force_n"].reshape(ticks, len(pairs))
    trace["foot_self_force_per_foot_n"] = np.repeat(forces, 2, axis=1)
    trace["foot_self_force_n"] = forces.max(axis=1)
    if "command_m_s" not in trace:
        result = next(
            r for r in metadata["results"]
            if f"{r['policy']}_{r['mode']}" == trace_path.stem
        )
        command = np.append(
            np.asarray(result["command_TB_m_s"]) * metadata["normalization_scale"], 0.
        )
        trace["command_m_s"] = np.broadcast_to(command, (ticks, 3))
    age = np.zeros(2)
    trace["support_age_s"] = np.zeros((ticks, 2))
    for index in range(count):
        age = (age + metadata["control_period_s"]) * trace["contact"][index]
        trace["support_age_s"][index] = age
    for name, value in trace.items():
        if not np.isfinite(value[:count]).all():
            raise ValueError(f"Nonfinite recorded {name}")
    if "foot_gap_m" in trace:
        trace["foot_gap_m"] = trace["foot_gap_m"].reshape(ticks, -1).min(axis=1)
    else:
        # The latest CPU probes retain geometry but omit signed distances.
        # Use their cached contact-time frames, not integrated-qpos foot poses.
        data = mujoco.MjData(model)
        gaps = np.zeros(ticks)
        for index in range(count):
            apply_trace_frame(model, data, trace, index, feet)
            gaps[index] = min(
                mujoco.mj_geomDistance(model, data, *pair, 1.0, None) for pair in pairs
            )
        trace["foot_gap_m"] = gaps
    return metadata, trace, model, feet


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
    for side, geom in enumerate(foot_ids):
        color = (
            [1.0, 0.15, 0.1, 1.0]
            if trace["foot_self_force_per_foot_n"][index, side] > 0.1
            else [0.1, 0.7, 1.0, 1.0]
        )
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
    metadata, trace, model, feet = load_recording(args.trace, args.robot, args.tb_root)
    count = int(trace["valid"].sum())
    data = mujoco.MjData(model)
    hips = [
        metadata["active_joint_names"].index(f"{side}_hip_roll")
        for side in ("left", "right")
    ]
    dt = metadata["control_period_s"]
    scale = metadata["normalization_scale"]
    hip_signs = np.asarray([-1, 1 if args.robot == "wr2" else -1])
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
        f"Replaying {args.robot} / {args.trace.name}: {count} ticks, "
        f"{count * dt:.2f} physical / {count * dt / scale:.2f} TB seconds"
    )
    print("Hip angles in the overlay use outward-positive directions for both legs.")
    if args.check_only:
        scene = mujoco.MjvScene(model, maxgeom=100)
        for index in (0, count // 2, count - 1):
            apply_trace_frame(model, data, trace, index, feet)
            scene.ngeom = 0
            draw_foot_boxes(scene, model, trace, index, feet)
        print(f"Checked recorded poses and {len(feet)} foot collision boxes.")
        return 0
    print(
        "Space pause; [ / ] step; R restart; C collision outlines; 1 front / 2 side / 3 top / 4 perspective"
    )
    print(
        "Blue = no recorded self-contact force; red = force above 0.1 N on the pair. CAD feet use recorded body frames."
    )
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = (
                0  # Cached outlines replace the solid collision boxes.
            )
            viewer.opt.geomgroup[2] = 1  # Keep the CAD surfaces visible.
            viewer.opt.sitegroup[:] = 0
            viewer.cam.distance = max(0.8, 1.3 * model.stat.extent)
            viewer.cam.lookat[:] = model.stat.center
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
            target = np.rad2deg(trace["delivered_rad"][index, hips]) * hip_signs
            actual = np.rad2deg(trace["actual_rad"][index, hips]) * hip_signs
            force = float(trace["foot_self_force_n"][index])
            command = trace["command_m_s"][index] / scale
            support = trace["contact"][index].astype(int)
            age = trace["support_age_s"][index] / scale
            floor_force = (
                f"{trace['foot_vertical_force_n'][index, 0]:.2f} / {trace['foot_vertical_force_n'][index, 1]:.2f}"
                if "foot_vertical_force_n" in trace else "n/a"
            )
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Recorded rollout\nPhysical / TB time (s)\nCommand XY (TB m/s)\nBox floor contact L/R\nLoaded duration L/R (TB s)\nFloor force L/R (N)\nBox self-force (N)\nFoot-to-foot box gap (mm)\nHip target outward L/R (deg)\nHip actual outward L/R (deg)",
                    f"{args.robot}: {args.trace.stem}\n{(index + 1) * dt:.2f} / {(index + 1) * dt / scale:.2f}\n{command[0]:+.4f}, {command[1]:+.4f}\n{support[0]} / {support[1]}\n{age[0]:.3f} / {age[1]:.3f}\n{floor_force}\n{force:.3f}\n{trace['foot_gap_m'][index] * 1000:+.3f}\n{target[0]:+.2f} / {target[1]:+.2f}\n{actual[0]:+.2f} / {actual[1]:+.2f}",
                )
            )
            viewer.sync()
            time.sleep(0.005)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("wr2", "toddlerbot"), default="wr2")
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="NPZ trace with its diagnostic summary.json alongside",
    )
    parser.add_argument("--playback-speed", type=float, default=0.25)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument(
        "--tb-root", type=Path, default=TB_PAPER_ROOT,
        help="Local ToddlerBot Paper checkout with original model/mesh assets",
    )
    parser.add_argument("--check-only", action="store_true", help="Check replay without opening a window")
    args = parser.parse_args()
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("--playback-speed must be finite and positive")
    return view_trace(args)


if __name__ == "__main__":
    raise SystemExit(main())
