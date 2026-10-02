"""Replay and validate WR2's ZMP walking reference in MuJoCo Viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from wr2.locomotion.configs import DEFAULT_TRAINING_CONFIG_PATH
from wr2.reference.zmp_validation import (
    apply_reference_frame,
    box_projected_corners_xy,
    convex_hull_xy,
    create_reference_context,
    validate_zmp_reference,
)


def _parse_commands(value: str) -> tuple[float, ...]:
    try:
        commands = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "commands must be comma-separated numbers"
        ) from exc
    if not commands or any(command <= 0.0 for command in commands):
        raise argparse.ArgumentTypeError("commands must be positive")
    return commands


def _add_sphere(scene, position, radius, rgba) -> None:
    index = scene.ngeom
    mujoco.mjv_initGeom(
        scene.geoms[index],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(position, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_line(scene, start, end, rgba, width=3.0) -> None:
    index = scene.ngeom
    mujoco.mjv_initGeom(
        scene.geoms[index],
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[index],
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _draw_reference_overlay(viewer, context, sample, root_x: float) -> None:
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        actual_com = np.asarray(context.data.subtree_com[0])
        planned_com = np.asarray([root_x, sample.com_lateral_m, sample.root_height_m])
        planned_zmp = np.asarray([root_x, sample.zmp_lateral_m, 0.004])
        _add_sphere(scene, actual_com, 0.008, [1.0, 0.1, 0.1, 1.0])
        _add_sphere(scene, planned_com, 0.007, [1.0, 0.8, 0.1, 1.0])
        _add_sphere(scene, planned_zmp, 0.006, [0.1, 1.0, 0.1, 1.0])
        _add_line(
            scene,
            actual_com,
            [actual_com[0], actual_com[1], 0.0],
            [1.0, 0.2, 0.2, 0.8],
            2.0,
        )

        active_points = []
        for geom_id, stance in zip(
            context.foot_geom_ids, sample.stance_mask, strict=True
        ):
            polygon = convex_hull_xy(
                box_projected_corners_xy(context.model, context.data, int(geom_id))
            )
            color = [0.2, 0.7, 1.0, 1.0] if stance else [0.4, 0.4, 0.4, 0.5]
            for index, start_xy in enumerate(polygon):
                end_xy = polygon[(index + 1) % len(polygon)]
                _add_line(
                    scene,
                    [start_xy[0], start_xy[1], 0.002],
                    [end_xy[0], end_xy[1], 0.002],
                    color,
                    2.0,
                )
            if stance:
                active_points.append(polygon)

        support = convex_hull_xy(np.concatenate(active_points))
        for index, start_xy in enumerate(support):
            end_xy = support[(index + 1) % len(support)]
            _add_line(
                scene,
                [start_xy[0], start_xy[1], 0.004],
                [end_xy[0], end_xy[1], 0.004],
                [0.1, 1.0, 0.1, 1.0],
                4.0,
            )


def _print_validation(result) -> None:
    verdict = "PASS" if result.passed else "FAIL"
    print(
        f"vx={result.command_forward_m_s:.2f} {verdict}: "
        f"stance_z={result.worst_stance_bottom_z_m:+.4f}m "
        f"swing_z={result.worst_swing_bottom_z_m:+.4f}m "
        f"zmp_margin={result.minimum_zmp_support_margin_m * 1000.0:+.2f}mm "
        f"lipm={result.maximum_lipm_residual_m:.2e}m "
        f"max_dq={result.maximum_joint_step_rad:.3f}rad"
    )
    for failure in result.failures[:10]:
        print(f"  - {failure}")
    if len(result.failures) > 10:
        print(f"  ... and {len(result.failures) - 10} more")


def validate_only(args: argparse.Namespace) -> int:
    context = create_reference_context(args.commands, args.config)
    failed = False
    for command in args.commands:
        result = validate_zmp_reference(context, command, samples=args.samples)
        _print_validation(result)
        failed |= not result.passed
    return int(failed)


def view_reference(args: argparse.Namespace) -> int:
    command = args.commands[0]
    context = create_reference_context((command,), args.config)
    result = validate_zmp_reference(context, command, samples=args.samples)
    _print_validation(result)
    if not result.passed and not args.show_failed:
        print("Refusing to open the viewer for a failed reference; use --show-failed.")
        return 1

    control_period = context.training_config.environment.gait_cycle_s / (
        context.training_config.environment.zmp_reference.phase_samples
    )
    playback = {"time": 0.0, "paused": bool(args.paused)}

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            playback["paused"] = not playback["paused"]
        elif keycode == ord("["):
            playback["paused"] = True
            playback["time"] = max(0.0, playback["time"] - control_period)
        elif keycode == ord("]"):
            playback["paused"] = True
            playback["time"] += control_period
        elif keycode in (ord("r"), ord("R")):
            playback["time"] = 0.0

    cycle_time = context.training_config.environment.gait_cycle_s
    print(
        "Viewer: red=MuJoCo COM, yellow=planned LIPM COM/root proxy, green=planned ZMP"
    )
    print("        green outline=active support, blue=stance foot, gray=swing foot")
    print("Controls: Space pause; [ / ] step; R reset")
    with mujoco.viewer.launch_passive(
        context.model, context.data, key_callback=key_callback
    ) as viewer:
        viewer.cam.distance = 0.75
        viewer.cam.elevation = -18.0
        previous_wall_time = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            wall_delta = min(now - previous_wall_time, 0.1)
            previous_wall_time = now
            if not playback["paused"]:
                playback["time"] += wall_delta * args.playback_speed
            if args.cycles > 0 and playback["time"] >= args.cycles * cycle_time:
                break

            cycle_index = int(playback["time"] // cycle_time)
            phase_time = playback["time"] - cycle_index * cycle_time
            phase = phase_time / cycle_time * 2.0 * np.pi
            sample, root_x = apply_reference_frame(
                context,
                phase_rad=phase,
                command_forward_m_s=command,
                cycle_index=cycle_index,
            )
            _draw_reference_overlay(viewer, context, sample, root_x)
            viewer.sync()
            time.sleep(0.005)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument(
        "--commands",
        type=_parse_commands,
        default=(0.10,),
        help="Comma-separated forward speeds; viewer uses the first",
    )
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=0, help="Zero loops forever")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Open the viewer even if the numeric reference gate fails",
    )
    args = parser.parse_args()
    if args.samples < 8:
        parser.error("--samples must be at least eight")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if args.cycles < 0:
        parser.error("--cycles must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    return validate_only(args) if args.validate_only else view_reference(args)


if __name__ == "__main__":
    raise SystemExit(main())
