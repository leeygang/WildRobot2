"""Run a WR2 Paper policy live or replay WR2/TB recordings in MuJoCo."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import glfw
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
    """Show integrated body pose and feet at their cached contact sampling instant."""
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
    """Outline cached collision boxes; red means model foot self-contact."""
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


class LivePolicy:
    """A loaded deterministic PPO policy stepping its native MJX environment."""

    def __init__(self, run_dir, checkpoint_path=None, command=(.1, 0., 0.)):
        from dataclasses import replace
        from functools import partial

        from wr2.locomotion.train import (
            _configure_backend_logging, _filter_optional_backend_import_messages,
        )

        _configure_backend_logging()
        with _filter_optional_backend_import_messages():
            import jax
            import jax.numpy as jp
            from brax.io import model as brax_model
            from brax.training import types
            from brax.training.acme import running_statistics
            from brax.training.agents.ppo import checkpoint, networks
            from wr2.locomotion.configs import load_training_config
            from wr2.locomotion.ppo import make_network_factory
            from wr2.reference.paper_walk import paper_command_grid
            import wr2.locomotion.walking_env as env_module

        self.jax = jax
        config = load_training_config(run_dir / "training_config.yaml")
        if config.environment.recipe != "tb_paper":
            raise ValueError("Live viewing requires a WR2 Paper training run")
        self.path = (checkpoint_path or run_dir / "best_params").resolve()
        params = (
            checkpoint.load(str(self.path)) if self.path.is_dir()
            else brax_model.load_params(self.path)
        )
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
            raise ValueError("Checkpoint has nonfinite parameters")
        scale = config.environment.normalization_scale
        self.command = jp.asarray(np.asarray(command) * [scale, scale, 1 / scale], dtype=jp.float32)
        nominal = replace(
            config.environment,
            command_resample_steps=config.environment.episode_length + 1,
            randomization=replace(config.environment.randomization, enabled=False),
        )
        # Select the exact native float32 lookup rows needed for this command.
        grid = paper_command_grid(scale)
        indices = np.asarray(jax.vmap(lambda c: jp.argmin(
            jp.linalg.norm(jp.asarray(grid, dtype=jp.float32) - c, axis=1)
        ))(jp.stack([jp.zeros(3, dtype=jp.float32), self.command])))
        print(f"Loading {self.path.name}; building nominal reference...", flush=True)
        reference_class = env_module.PaperWalkReference
        env_module.PaperWalkReference = partial(reference_class, command_vectors=grid[indices])
        try:
            self.env = env_module.WR2WalkingEnv(
                nominal, add_observation_noise=False,
                reward_discounting=config.ppo.discounting,
            )
        finally:
            env_module.PaperWalkReference = reference_class
        env = self.env
        self.model = env.sys.mj_model
        self.feet = np.asarray(env._foot_geom_ids)
        self.data = mujoco.MjData(self.model)
        self.metadata = {
            "active_joint_names": [env.robot.actuator_names[int(i)] for i in env._active_indices],
            "control_period_s": env.robot.control_period_s,
            "normalization_scale": scale,
        }
        preprocess = running_statistics.normalize if config.ppo.normalize_observations else types.identity_observation_preprocessor
        network = make_network_factory(config.network)(
            env.observation_size, env.action_size,
            preprocess_observations_fn=preprocess,
        )
        policy = networks.make_inference_fn(network)(params[:2], deterministic=True)
        self._step = jax.jit(lambda state: env.step(
            state, policy(state.obs, state.info["rng"])[0],
        ))
        self._reset = jax.jit(env.reset)
        self._key = jax.random.split(jax.random.split(jax.random.PRNGKey(22102))[1], 8)[0]
        feet = env._foot_geom_ids
        active = env._active_indices

        def sample(state):
            p = state.pipeline_state
            forces = env._contact_forces(p)
            pair = ((p.contact.geom1 == feet[0]) & (p.contact.geom2 == feet[1])) | (
                (p.contact.geom1 == feet[1]) & (p.contact.geom2 == feet[0])
            )
            pair &= (p.contact.dist < 0) & (p._impl.contact.efc_address >= 0)
            return {
                "qpos": p.q,
                "foot_geom_center_m": p.geom_xpos[feet],
                "foot_geom_rotation": p.geom_xmat[feet],
                "foot_self_force_n": jp.max(jp.where(pair, jp.linalg.norm(forces[:, :3], axis=1), 0.)),
                "command_m_s": state.info["command"],
                "delivered_rad": state.info["previous_target"][active],
                "actual_rad": p.q[env._joint_qpos_indices][active],
                "contact": env._foot_contact(p, forces),
                "foot_vertical_force_n": env._foot_vertical_forces(p, forces),
                "time_s": state.info["step_count"] * env.robot.control_period_s,
                "nonfinite": state.metrics["nonfinite_state"],
            }

        self._sample = jax.jit(sample)
        self.reset()
        print(f"Compiling policy and physics on {jax.default_backend()}...", flush=True)
        # Compile without advancing the displayed initial state.
        self._step(self.state).pipeline_state.q.block_until_ready()

    def reset(self):
        self.state = self._reset(self._key, self.command)
        self.age = np.zeros(2)
        self._capture()

    def advance(self):
        self.state = self._step(self.state)
        self._capture(advance=True)

    def _capture(self, advance=False):
        values = self.jax.device_get(self._sample(self.state))
        if values.pop("nonfinite") or any(not np.isfinite(x).all() for x in values.values()):
            raise RuntimeError("Nonfinite live simulation state")
        self.time_s = float(values["time_s"])
        if self.time_s == 0:
            self.age[:] = 0
        elif advance:
            self.age = (self.age + self.metadata["control_period_s"]) * values["contact"]
        self.trace = {name: np.asarray(value)[None] for name, value in values.items()}
        self.trace["support_age_s"] = self.age[None].copy()
        self.trace["foot_self_force_per_foot_n"] = np.repeat(self.trace["foot_self_force_n"][:, None], 2, axis=1)
        apply_trace_frame(self.model, self.data, self.trace, 0, self.feet)
        self.trace["foot_gap_m"] = np.asarray([
            mujoco.mj_geomDistance(self.model, self.data, *self.feet, 1., None),
        ])


def view_trace(args: argparse.Namespace) -> int:
    if not args.check_only and (not glfw.init() or not glfw.get_monitors()):
        raise RuntimeError(
            "MuJoCo viewing requires a desktop display. Use --check-only for a headless simulation check."
        )
    live = None
    if args.run_dir:
        live = LivePolicy(args.run_dir, args.checkpoint, args.command)
        metadata, trace, model, feet = live.metadata, live.trace, live.model, live.feet
        count = 1
    else:
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
    playback = {"time": 0.0, "paused": args.paused, "boxes": True, "view": 1, "step": False, "reset": False}

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            playback["paused"] = not playback["paused"]
        elif keycode in (ord("["), ord("]")):
            playback["paused"] = True
            if live:
                playback["step"] = keycode == ord("]")
                return
            step = -1 if keycode == ord("[") else 1
            playback["time"] = (
                np.clip(round(playback["time"] / dt) + step, 0, count - 1) * dt
            )
        elif keycode in (ord("r"), ord("R")):
            playback["time"] = 0.0
            playback["reset"] = True
        elif keycode in (ord("c"), ord("C")):
            playback["boxes"] = not playback["boxes"]
        elif keycode in (ord("1"), ord("2"), ord("3"), ord("4")):
            playback["view"] = keycode - ord("0")

    label = live.path.name if live else args.trace.stem
    print(f"Running live policy {live.path}" if live else (
        f"Replaying {args.robot} / {args.trace.name}: {count} ticks, "
        f"{count * dt:.2f} physical / {count * dt / scale:.2f} TB seconds"
    ))
    print("Hip angles in the overlay use outward-positive directions for both legs.")
    if args.check_only:
        scene = mujoco.MjvScene(model, maxgeom=100)
        if live:
            for _ in range(20):
                live.advance()
                apply_trace_frame(model, data, live.trace, 0, feet)
                scene.ngeom = 0
                draw_foot_boxes(scene, model, live.trace, 0, feet)
            print(f"Checked 20 live policy/physics steps, time={live.time_s:.2f}s.")
            return 0
        for index in (0, count // 2, count - 1):
            apply_trace_frame(model, data, trace, index, feet)
            scene.ngeom = 0
            draw_foot_boxes(scene, model, trace, index, feet)
        print(f"Checked recorded poses and {len(feet)} foot collision boxes.")
        return 0
    print(
        f"Space pause; {'] step' if live else '[ / ] step'}; R restart; C collision outlines; 1 front / 2 side / 3 top / 4 perspective"
    )
    print(
        "Blue = self-contact force at most 0.1 N; red = force above 0.1 N on the pair. CAD feet use cached contact-time frames."
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
        next_tick = previous_time
        while viewer.is_running():
            now = time.monotonic()
            if live:
                if playback["reset"]:
                    live.reset()
                    playback["reset"] = False
                if playback["step"] or (not playback["paused"] and now >= next_tick):
                    live.advance()
                    playback["step"] = False
                    next_tick = now + dt / args.playback_speed
                trace = live.trace
            elif not playback["paused"]:
                playback["time"] = (
                    playback["time"]
                    + min(now - previous_time, 0.1) * args.playback_speed
                ) % (count * dt)
            previous_time = now
            index = 0 if live else min(int(playback["time"] / dt), count - 1)
            physical_time = live.time_s if live else (index + 1) * dt
            with viewer.lock():
                data.time = physical_time
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
                    f"{'Live policy' if live else 'Recorded rollout'}\nPhysical / TB time (s)\nCommand XY (TB m/s)\nBox floor contact L/R\nLoaded duration L/R (TB s)\nFloor force L/R (N)\nBox self-force (N)\nFoot-to-foot box gap (mm)\nHip target outward L/R (deg)\nHip actual outward L/R (deg)",
                    f"{args.robot}: {label}\n{physical_time:.2f} / {physical_time / scale:.2f}\n{command[0]:+.4f}, {command[1]:+.4f}\n{support[0]} / {support[1]}\n{age[0]:.3f} / {age[1]:.3f}\n{floor_force}\n{force:.3f}\n{trace['foot_gap_m'][index] * 1000:+.3f}\n{target[0]:+.2f} / {target[1]:+.2f}\n{actual[0]:+.2f} / {actual[1]:+.2f}",
                )
            )
            viewer.sync()
            time.sleep(0.005)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("wr2", "toddlerbot"), default="wr2")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trace",
        type=Path,
        help="NPZ trace with its diagnostic summary.json alongside",
    )
    source.add_argument("--run-dir", type=Path, help="Run a WR2 Paper policy live using its saved training_config.yaml")
    parser.add_argument("--checkpoint", type=Path, help="Live policy file or Orbax directory; defaults to run-dir/best_params")
    parser.add_argument("--command", type=float, nargs=3, default=(.1, 0., 0.), metavar=("VX", "VY", "YAW"), help="Live command in TB-equivalent m/s, m/s, rad/s")
    parser.add_argument("--playback-speed", type=float, default=0.25)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument(
        "--tb-root", type=Path, default=TB_PAPER_ROOT,
        help="Local ToddlerBot Paper checkout with original model/mesh assets",
    )
    parser.add_argument("--check-only", action="store_true", help="Check replay or 20 live steps without opening a window")
    args = parser.parse_args(argv)
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("--playback-speed must be finite and positive")
    if not np.isfinite(args.command).all():
        parser.error("--command must be finite")
    if args.checkpoint and not args.run_dir:
        parser.error("--checkpoint requires --run-dir")
    if args.run_dir and args.robot != "wr2":
        parser.error("Live policy viewing supports WR2; use --trace for ToddlerBot")
    return args


def main() -> int:
    args = parse_args()
    return view_trace(args)


if __name__ == "__main__":
    raise SystemExit(main())
