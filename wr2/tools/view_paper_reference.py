"""Render WR2 or the local ToddlerBot Paper ZMP reference in MuJoCo.

Commands use TB-equivalent units; WR2 uses its training speed/time scale.
References: https://arxiv.org/abs/2502.00893 and the local TB Paper source.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.reference.paper_walk import PaperWalkReference, paper_command_grid
from wr2.reference.zmp_validation import box_projected_corners_xy, convex_hull_xy
from wr2.sim.robot import RobotDescription
from wr2.tools.view_zmp_reference import _add_line, _add_sphere


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_CONFIG = PROJECT_ROOT / "wr2/locomotion/configs/ppo_walking_paper.yaml"
TB_PAPER_ROOT = PROJECT_ROOT / ".worktrees/tb-paper-stance"


@dataclass
class ReferenceReplay:
    model: mujoco.MjModel
    qpos: np.ndarray
    stance: np.ndarray
    planned_com: np.ndarray
    desired_zmp: np.ndarray
    optimal_cop: np.ndarray
    foot_geoms: tuple[tuple[int, ...], tuple[int, ...]]
    dt: float
    cycle: float
    scale: float
    command: np.ndarray
    selected_command: np.ndarray
    lookup_length: int
    source: str


def load_wr2_reference(config_path: Path, command_tb: np.ndarray) -> ReferenceReplay:
    config = load_training_config(config_path).environment
    if config.recipe != "tb_paper":
        raise ValueError("--config must use recipe: tb_paper")
    robot = RobotDescription.load(include_local_calibration=False)
    model = mujoco.MjModel.from_xml_path(str(robot.directory / "scene_mjx.xml"))
    scale = config.normalization_scale
    command = np.asarray(command_tb) * [scale, scale, 1 / scale]
    keys = paper_command_grid(scale)
    # Select with the same float32 JAX operation as the training reference.
    nearest = int(
        jp.argmin(
            jp.linalg.norm(
                jp.asarray(keys, dtype=jp.float32)
                - jp.asarray(command, dtype=jp.float32),
                axis=1,
            )
        )
    )
    selected = keys[nearest]
    print("Generating WR2's selected training lookup...", flush=True)
    reference = PaperWalkReference(
        model,
        keyframe_id=model.key("walk_home").id,
        actuator_names=robot.actuator_names,
        config=config,
        control_dt=robot.control_period_s,
        normalization_scale=scale,
        command_vectors=[selected],
    )
    count = config.episode_length
    frames = [
        reference.sample_time(i * robot.control_period_s, command) for i in range(count)
    ]
    qpos, stance, optimal, desired = map(np.asarray, zip(*frames, strict=True))
    data = mujoco.MjData(model)
    data.qpos[:] = reference.standing_qpos
    mujoco.mj_forward(model, data)
    offset = data.subtree_com[0, :2] - reference.standing_qpos[:2]
    yaw = 2 * np.arctan2(qpos[:, 6], qpos[:, 3])
    com = np.column_stack(
        [
            qpos[:, 0] + np.cos(yaw) * offset[0] - np.sin(yaw) * offset[1],
            qpos[:, 1] + np.sin(yaw) * offset[0] + np.cos(yaw) * offset[1],
            np.full(count, reference.diagnostics.com_height_m),
        ]
    )
    return ReferenceReplay(
        model,
        qpos,
        stance,
        com,
        desired,
        optimal,
        tuple((model.geom(f"{side}_foot_collision").id,) for side in ("left", "right")),
        robot.control_period_s,
        config.gait_cycle_s,
        scale,
        command,
        selected,
        reference.diagnostics.lookup_length,
        str(config_path),
    )


def load_tb_reference(root: Path, command_tb: np.ndarray) -> ReferenceReplay:
    root = root.expanduser().resolve()
    cache_path = root / "toddlerbot/descriptions/toddlerbot/walk_zmp_lookup_table.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing Paper cache: {cache_path}. Point --tb-root at the frozen "
            "paper-walking-baseline checkout with its original model/cache assets."
        )
    # TB's native loaders resolve assets relative to the checkout.
    previous_directory = Path.cwd()
    sys.path.insert(0, str(root))
    os.environ["USE_JAX"] = "false"
    try:
        os.chdir(root)
        from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
        from toddlerbot.sim.robot import Robot

        reference = WalkZMPReference(Robot("toddlerbot"), 0.02, 0.72, 0.0)
        with cache_path.open("rb") as stream:
            _, cached_com, _, _ = pickle.load(stream)
        command = np.asarray(command_tb, dtype=np.float32)
        nearest = int(
            np.argmin(np.linalg.norm(reference.lookup_keys - command, axis=1))
        )
        selected = reference.lookup_keys[nearest]
        length = int(reference.lookup_length[nearest])
        planner = reference.zmp_walk
        first = int(np.ceil(reference.cycle_time / reference.dt))
        moving = np.linalg.norm(command) >= 1e-6
        if moving:
            desired, regenerated_com, legs, masks = planner.plan(
                np.zeros(3, dtype=np.float32),
                np.asarray([1, 0, 0, 0], dtype=np.float32),
                selected,
            )
            # The overlays must describe the original cached joint reference.
            np.testing.assert_allclose(
                regenerated_com[first : first + length],
                cached_com[nearest],
                atol=2e-5,
                rtol=2e-5,
            )
            np.testing.assert_allclose(
                legs[first : first + length],
                reference.leg_joint_pos_lookup[nearest, :length],
                atol=2e-5,
                rtol=2e-5,
            )
            np.testing.assert_array_equal(
                masks[first : first + length],
                reference.stance_mask_lookup[nearest, :length],
            )
            times = planner.zmp_planner.time_steps
            steps = len(desired) // 2
            yaws = np.arange(steps) * selected[2] * 20 / (steps - 1)
            if np.linalg.norm(selected[:2]) > 1e-6:
                yaws[:] = 0
            footsteps = np.column_stack(
                [desired, np.repeat(yaws, 2), np.repeat(np.arange(steps) % 2, 2)]
            )
            left, _, right, _, torso, _ = planner.compute_foot_trajectories(
                times, footsteps
            )
            feet = np.stack([left, right], axis=1)

        qposes, stance, com, desired_zmp, optimal = [], [], [], [], []
        state = np.zeros(13 + 2 * reference.robot.nu + 2, dtype=np.float32)
        state[3] = 1
        full_command = np.concatenate([np.zeros(5, dtype=np.float32), command])
        for i in range(1000):
            state = reference.get_state_ref(state, i * reference.dt, full_command)
            render_state = state.copy()
            render_state[:3] = 0
            yaw = torso[first + i % length, 2] if moving and i > 0 else 0.0
            render_state[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            qpos = reference.get_qpos_ref(render_state, path_frame=False)
            if moving and i > 0:
                frame = i % length
                plan_frame = first + frame
                data = reference.forward(qpos)
                side = int(
                    data.site_xpos[reference.left_foot_site_id, 2]
                    >= data.site_xpos[reference.right_foot_site_id, 2]
                )
                origin = (reference.left_foot_center, reference.right_foot_center)[side]
                # Undo get_qpos_ref's fixed XY support anchor with the native
                # finite footstep trajectory. Motor/joint pose and Z stay native.
                qpos[:2] += feet[plan_frame, side, :2] - origin[:2]
                planned = cached_com[nearest][frame]
                acceleration = planner.zmp_planner.get_optim_com_acc(
                    plan_frame * reference.dt, planned
                )
                cop = planned[:2] - planner.com_z / 9.81 * acceleration
                target = planner.zmp_planner.get_desired_zmp(plan_frame * reference.dt)
                planned_xy = planned[:2]
            else:
                planned_xy = reference.desired_com[:2]
                target = cop = planned_xy
            qposes.append(qpos)
            stance.append(state[-2:])
            com.append([*planned_xy, planner.com_z])
            desired_zmp.append(target)
            optimal.append(cop)
        model = reference.model
        geoms = (
            (
                model.geom("left_ank_pitch_link_collision").id,
                model.geom("ank_roll_link_collision").id,
            ),
            (
                model.geom("right_ank_pitch_link_collision").id,
                model.geom("ank_roll_link_2_collision").id,
            ),
        )
        return ReferenceReplay(
            model,
            np.asarray(qposes),
            np.asarray(stance, dtype=bool),
            np.asarray(com),
            np.asarray(desired_zmp),
            np.asarray(optimal),
            geoms,
            reference.dt,
            reference.cycle_time,
            1.0,
            command,
            selected,
            length,
            str(cache_path),
        )
    finally:
        os.chdir(previous_directory)
        sys.path.remove(str(root))


def draw_overlay(scene, replay, data, index):
    scene.ngeom = 0
    _add_sphere(scene, data.subtree_com[0], 0.008, [1, 0.1, 0.1, 1])
    _add_sphere(scene, replay.planned_com[index], 0.007, [1, 0.8, 0.1, 1])
    _add_sphere(scene, [*replay.desired_zmp[index], 0.006], 0.006, [0.1, 1, 0.1, 1])
    _add_sphere(scene, [*replay.optimal_cop[index], 0.012], 0.004, [0.9, 0.2, 1, 1])
    _add_line(
        scene,
        data.subtree_com[0],
        [*data.subtree_com[0, :2], 0.002],
        [1, 0.1, 0.1, 0.8],
        2,
    )
    active = []
    for side, ids in enumerate(replay.foot_geoms):
        polygon = convex_hull_xy(
            np.concatenate(
                [box_projected_corners_xy(replay.model, data, geom) for geom in ids]
            )
        )
        support = replay.stance[index, side]
        color = [0.2, 0.7, 1, 1] if support else [0.5, 0.5, 0.5, 0.7]
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            _add_line(scene, [*start, 0.002], [*end, 0.002], color, 2)
        if support:
            active.append(polygon)
    hull = convex_hull_xy(np.concatenate(active))
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        _add_line(scene, [*start, 0.004], [*end, 0.004], [0.1, 1, 0.1, 1], 3)


def view_reference(replay, args):
    data = mujoco.MjData(replay.model)
    count = len(replay.qpos)
    playback = {"time": 0.0, "paused": args.paused, "view": 1}

    def key_callback(key):
        if key == ord(" "):
            playback["paused"] = not playback["paused"]
        elif key in (ord("["), ord("]")):
            playback["paused"] = True
            playback["time"] = (
                np.clip(
                    round(playback["time"] / replay.dt)
                    + (-1 if key == ord("[") else 1),
                    0,
                    count - 1,
                )
                * replay.dt
            )
        elif key in (ord("r"), ord("R")):
            playback["time"] = 0.0
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            playback["view"] = key - ord("0")

    print(
        "Space pause; [ / ] step; R restart; 1 front / 2 side / 3 top / 4 perspective"
    )
    print(
        "Red=actual COM; yellow=planner COM/proxy; green=desired ZMP/support; magenta=optimal COP"
    )
    print(
        "Blue=planned stance foot; gray=planned swing foot. Kinematic reference replay."
    )
    with mujoco.viewer.launch_passive(
        replay.model, data, key_callback=key_callback
    ) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[2] = 1
            viewer.opt.sitegroup[:] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            viewer.cam.distance = 0.85 * replay.scale
        previous = time.monotonic()
        loops = 0
        while viewer.is_running():
            now = time.monotonic()
            if not playback["paused"]:
                playback["time"] += min(now - previous, 0.1) * args.playback_speed
                if playback["time"] >= count * replay.dt:
                    loops += 1
                    if args.loops and loops >= args.loops:
                        break
                    playback["time"] %= count * replay.dt
            previous = now
            index = min(round(playback["time"] / replay.dt), count - 1)
            with viewer.lock():
                data.qpos[:] = replay.qpos[index]
                data.time = index * replay.dt
                mujoco.mj_forward(replay.model, data)
                viewer.cam.lookat[:] = data.subtree_com[0]
                if playback["view"]:
                    viewer.cam.azimuth, viewer.cam.elevation = {
                        1: (0, -10),
                        2: (90, -10),
                        3: (90, -90),
                        4: (135, -20),
                    }[playback["view"]]
                    playback["view"] = 0
                draw_overlay(viewer.user_scn, replay, data, index)
            mask = replay.stance[index].astype(int)
            viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "ZMP reference\nPhysical / TB time (s)\nCommand XY / yaw\nLookup tick\nPlanned stance L/R",
                        f"{args.robot}\n{data.time:.2f} / {data.time / replay.scale:.2f}\n"
                        f"{replay.command[0]:+.4f}, {replay.command[1]:+.4f} / {replay.command[2]:+.3f}\n"
                        f"{index % replay.lookup_length} / {replay.lookup_length}\n{mask[0]} / {mask[1]}",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        "Red: actual COM\nYellow: planned COM/proxy\nGreen: desired ZMP + support\n"
                        "Magenta: optimal COP\nBlue/gray: stance/swing\n"
                        "Space: pause   [ / ]: step\n1/2/3/4: camera   R: restart",
                        "",
                    ),
                ]
            )
            viewer.sync()
            time.sleep(0.005)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("wr2", "tb"), required=True)
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=(0.1, 0.0, 0.0),
        metavar=("VX", "VY", "YAW"),
        help="TB-equivalent m/s, m/s, rad/s",
    )
    parser.add_argument(
        "--config", type=Path, default=PAPER_CONFIG, help="WR2 paper training YAML"
    )
    parser.add_argument(
        "--tb-root",
        type=Path,
        default=TB_PAPER_ROOT
        if TB_PAPER_ROOT.exists()
        else Path.home() / "projects/toddlerbot",
        help="Frozen ToddlerBot Paper checkout with original cache/assets",
    )
    parser.add_argument("--playback-speed", type=float, default=0.25)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument(
        "--loops", type=int, default=0, help="Episode replays; zero loops forever"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load/check all frames without opening a window",
    )
    args = parser.parse_args()
    if not np.isfinite(args.command).all():
        parser.error("--command must be finite")
    if np.linalg.norm(args.command[:2]) > 1e-6 and abs(args.command[2]) > 1e-6:
        parser.error(
            "The Paper lookup supports XY walking or pure yaw; use zero yaw for XY"
        )
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("--playback-speed must be finite and positive")
    if args.loops < 0:
        parser.error("--loops must be nonnegative")
    command = np.asarray(args.command)
    replay = (
        load_wr2_reference(args.config, command)
        if args.robot == "wr2"
        else load_tb_reference(args.tb_root, command)
    )
    for name in ("qpos", "planned_com", "desired_zmp", "optimal_cop"):
        if not np.isfinite(getattr(replay, name)).all():
            raise ValueError(f"Nonfinite {name} in reference")
    print(f"Source: {replay.source}")
    print(
        f"Physical command: {replay.command}; selected lookup: {replay.selected_command}"
    )
    print(
        f"{len(replay.qpos)} frames at {replay.dt:.3f}s; gait {replay.cycle:.3f}s; "
        f"lookup wraps after {replay.lookup_length * replay.dt:.2f}s",
        flush=True,
    )
    return 0 if args.check_only else view_reference(replay, args)


if __name__ == "__main__":
    raise SystemExit(main())
