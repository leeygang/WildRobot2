"""Timestamped live-evaluation records with explicit physics and frame units."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import threading
import time

import mujoco
import numpy as np
import yaml


RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results/visual_eval"


def checkpoint_hash(path):
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*")):
        if member.is_file():
            digest.update(str(member.relative_to(path)).encode())
            digest.update(hashlib.sha256(member.read_bytes()).digest())
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class VisualEvaluationLog:
    def __init__(self, live, args):
        self.live, self.args = live, args
        self.directory = RESULTS_ROOT / datetime.now(timezone.utc).strftime("wr2_%Y%m%d_%H%M%S_%f")
        self.lock = threading.RLock()
        self.stream = None
        self.started = time.monotonic()
        self.episode, self.total_steps, self.valid_steps = 0, 0, 0
        self.events = Counter()
        self.previous_contact = np.zeros(2, bool)
        self.stance_steps = np.zeros(2, int)
        self.self_contact = False
        self.clip_active = False
        self.sums = Counter()
        self.minimum_box_gap = float("inf")
        self.minimum_cad_gap = float("inf")
        self.maximum_abs_pitch = 0.0
        self.data = mujoco.MjData(live.model)
        self.meshes = [np.flatnonzero(
            (live.model.geom_bodyid == live.model.geom_bodyid[geom])
            & (live.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        ) for geom in live.feet]

    def __enter__(self):
        env, model = self.live.env, self.live.model
        active = np.asarray(env._active_indices)
        config_path = self.live.run_dir / "training_config.yaml"
        run_manifest = self.live.run_dir / "run_config.json"
        run_data = json.loads(run_manifest.read_text()) if run_manifest.exists() else {}
        root = Path(__file__).resolve().parents[2]
        params = {
            "schema_version": 1,
            "session_id": self.directory.name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "mode": "headless_check" if self.args.check_only else "live",
            "viewer_arguments": vars(self.args),
            "initial_camera_preset": "front",
            "initial_collision_outlines": True,
            "invocation_argv": sys.argv,
            "python_executable": sys.executable,
            "run_dir": self.live.run_dir,
            "checkpoint": self.live.path,
            "checkpoint_sha256": checkpoint_hash(self.live.path),
            "checkpoint_hash_method": "file bytes; directory hashes sorted relative paths and per-file SHA256 digests",
            "training_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "saved_training_config": yaml.safe_load(config_path.read_text()),
            "effective_environment": asdict(env.config),
            "command_TB": np.asarray(self.live.command) * [1 / env.config.normalization_scale, 1 / env.config.normalization_scale, env.config.normalization_scale],
            "command_physical": np.asarray(self.live.command),
            "reference_lookup_command_physical": self.live.metadata["reference_lookup_command_physical"],
            "deterministic_policy": True,
            "observation_noise": env.add_observation_noise,
            "physical_pushes": bool(env.config.randomization.enabled and env.config.push_interval_steps and (env.config.push_linear_velocity_m_s or env.config.push_angular_velocity_rad_s)),
            "reset_seed": self.live.metadata["reset_seed"],
            "control_period_s": env.robot.control_period_s,
            "physics_timestep_s": model.opt.timestep,
            "physics_substeps": env._n_frames,
            "normalization_scale": env.config.normalization_scale,
            "active_joint_names": self.live.metadata["active_joint_names"],
            "action_home_rad": np.asarray(env._home_ctrl)[active],
            "initial_qpos": np.asarray(self.live.state.pipeline_state.q),
            "initial_qvel": np.asarray(self.live.state.pipeline_state.qd),
            "policy_action_scale_rad": env.config.policy_action_scale_rad,
            "controller_joint_limits_rad": np.stack([np.asarray(env._ctrl_lower)[active], np.asarray(env._ctrl_upper)[active]], axis=1),
            "model_dimensions": {name: int(getattr(model, name)) for name in ["nq", "nv", "nu", "ngeom"]},
            "solver_iterations": int(model.opt.iterations),
            "training_provenance": run_data.get("provenance"),
            "training_physics_manifest": run_data.get("physics"),
            "viewer_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "viewer_git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
            "actuator_parameters": {
                name: float(getattr(env, name)) for name in [
                    "_servo_kp", "_servo_kv", "_servo_torque_limit_nm",
                    "_servo_brake_torque_limit_nm", "_servo_max_speed_rad_s",
                    "_servo_peak_torque_speed_rad_s", "_servo_torque_at_max_speed_nm",
                    "_servo_passive_active_ratio", "_command_resolution_rad",
                ]
            },
            "actuator_config": env.robot.config["actuators"]["htd45hServo"],
            "dof_parameters": {name: np.asarray(getattr(model, name))[np.asarray(env._joint_qvel_indices)[active]] for name in ["dof_damping", "dof_frictionloss", "dof_armature"]},
            "runtime": {
                "packages": {name: metadata.version(name) for name in ["jax", "jaxlib", "brax", "mujoco", "mujoco-mjx", "flax", "numpy"]},
                "backend": self.live.jax.default_backend(),
                "matmul_precision": self.live.jax.config.jax_default_matmul_precision,
                "python": sys.version,
                "devices": [str(device) for device in self.live.jax.devices()],
                "jax_enable_x64": self.live.jax.config.jax_enable_x64,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
            },
            "viewer_source_sha256": hashlib.sha256(Path(__file__).with_name("view_walking.py").read_bytes()).hexdigest(),
            "logger_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sampling": {
                "time_utc": "record timestamp after the completed policy/physics step",
                "simulation_time_s": "physical episode time; total_simulation_time_s includes resets",
                "geometry": "cached last-substep geometry/force frames; integrated qpos is one physics timestep later",
                "pose": "raw physics before native reset; episode_reset marks reset of the displayed state",
                "torso_euler_rad": "integrated free-root pose, matching displayed body; positive pitch leans forward",
                "physics_torso_euler_rad": "cached torso quaternion used by native rewards/velocity rotation",
                "velocities": "body_velocity is the native reward point; world_net_velocity is root interval displacement/dt",
                "foot_contact": "native summed upward floor force above the saved threshold",
                "sustained_contact": f"{env._air_time_min_stance_steps} consecutive contact samples; an event, not necessarily paid air credit",
                "foot_gap": "signed collider/CAD convex-hull distance at cached foot frames, metres",
                "reward_terms": "weighted native per-step contributions including control dt",
                "joint_arrays": "active_joint_names order, radians/radians per second/Nm",
                "motor_torque_nm": "final physics-substep force, not a per-control-step peak",
            },
        }
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "params.json").write_text(json.dumps(json_value(params), indent=2, allow_nan=False) + "\n")
        self.stream = (self.directory / "metrics.jsonl").open("w", buffering=1)
        self.event("evaluation_start")
        print(f"Visual evaluation log: {self.directory}", flush=True)
        return self

    def _write(self, record):
        if self.stream is None:
            return
        stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        self.stream.write(json.dumps(json_value({
            "time_utc": stamp, "wall_elapsed_s": time.monotonic() - self.started,
            "episode": self.episode, "total_step": self.total_steps,
            "display_time_s": self.live.time_s,
            "total_simulation_time_s": self.total_steps * self.live.metadata["control_period_s"],
            **record,
        }), allow_nan=False) + "\n")

    def event(self, name, **details):
        with self.lock:
            self.events[name] += 1
            self._write({"type": "event", "event": name, **details})

    def reset(self, reason):
        with self.lock:
            self.episode += 1
            self.previous_contact[:] = False
            self.stance_steps[:] = 0
            self.self_contact = self.clip_active = False
            self.event("reset", reason=reason)

    def step(self):
        values = self.live.metrics
        if values is None:
            return
        with self.lock:
            self.total_steps += 1
            scale = self.live.metadata["normalization_scale"]
            dt = self.live.metadata["control_period_s"]
            finite = not bool(values["nonfinite"]) and all(
                np.isfinite(values[name]).all() for name in ["qpos", "foot_geom_center_m", "foot_geom_rotation", "body_velocity_m_s"]
            )
            events = []
            contact = np.asarray(values["contact"], bool)
            for i, side in enumerate(["left", "right"]):
                if contact[i] and not self.previous_contact[i]:
                    events.append(f"contact_on_{side}")
                elif not contact[i] and self.previous_contact[i]:
                    events.append(f"contact_off_{side}")
                self.stance_steps[i] = self.stance_steps[i] + 1 if contact[i] else 0
                if contact[i] and self.stance_steps[i] == self.live.env._air_time_min_stance_steps:
                    events.append(f"sustained_contact_{side}")
            touch = float(values["foot_self_force_n"]) > .1
            if touch != self.self_contact:
                events.append("self_contact_on" if touch else "self_contact_off")
            clipped = float(values["target_clipping_fraction"]) > 0
            if clipped != self.clip_active:
                events.append("target_clipping_on" if clipped else "target_clipping_off")
            if values["fall"]:
                events.append("fall")
            if values["nonfinite"]:
                events.append("nonfinite")
            self.previous_contact, self.self_contact, self.clip_active = contact.copy(), touch, clipped
            extra = {
                "command_TB": np.asarray(values["command_physical"]) * [1/scale, 1/scale, scale],
                "body_velocity_TB_m_s": np.asarray(values["body_velocity_m_s"]) / scale,
                "body_angular_velocity_TB_rad_s": np.asarray(values["body_angular_velocity_rad_s"]) * scale,
                "world_net_velocity_m_s": np.asarray(values["root_displacement_m"]) / dt,
                "torso_pitch_deg": float(np.rad2deg(values["torso_euler_rad"][1])),
                "joint_tracking_RMS_rad": float(np.sqrt(np.mean((values["actual_rad"] - values["delivered_rad"])**2))),
                "support_duration_s": np.maximum(self.stance_steps - 1, 0) * dt,
                "reference_time_s": max(float(values["simulation_time_s"]) - dt, 0),
                "geometry_time_s": max(float(values["simulation_time_s"]) - self.live.model.opt.timestep, 0),
                "box_gap_m": None, "CAD_convex_gap_m": None,
            }
            if finite:
                from wr2.tools.view_walking import apply_trace_frame
                trace = {name: np.asarray(values[name])[None] for name in ["qpos", "foot_geom_center_m", "foot_geom_rotation"]}
                apply_trace_frame(self.live.model, self.data, trace, 0, self.live.feet)
                extra["box_gap_m"] = mujoco.mj_geomDistance(self.live.model, self.data, *self.live.feet, 1., None)
                extra["CAD_convex_gap_m"] = min(
                    mujoco.mj_geomDistance(self.live.model, self.data, int(left), int(right), 1., None)
                    for left in self.meshes[0] for right in self.meshes[1]
                )
                yaw = float(values["physics_torso_euler_rad"][2])
                delta = values["foot_geom_center_m"][0, :2] - values["foot_geom_center_m"][1, :2]
                lead = float(delta @ [np.cos(yaw), np.sin(yaw)])
                extra["landing_lead_m"] = [lead, -lead]
                self.valid_steps += 1
                self.sums["forward_velocity_TB_m_s"] += float(extra["body_velocity_TB_m_s"][0])
                self.sums["world_forward_velocity_m_s"] += float(extra["world_net_velocity_m_s"][0])
                self.sums["world_lateral_velocity_m_s"] += float(extra["world_net_velocity_m_s"][1])
                self.sums["XY_error_TB_m_s"] += float(np.linalg.norm(extra["body_velocity_TB_m_s"][:2] - extra["command_TB"][:2]))
                self.sums["torso_pitch_deg"] += extra["torso_pitch_deg"]
                self.sums["self_contact_fraction"] += int(touch)
                self.minimum_box_gap = min(self.minimum_box_gap, extra["box_gap_m"])
                self.minimum_cad_gap = min(self.minimum_cad_gap, extra["CAD_convex_gap_m"])
                self.maximum_abs_pitch = max(self.maximum_abs_pitch, abs(extra["torso_pitch_deg"]))
            self.events.update(events)
            self._write({"type": "step", "events": events, "metrics": {**values, **extra}})
            if events:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')} UTC / sim {float(values['simulation_time_s']):.2f}s] {', '.join(events)}; pitch={extra['torso_pitch_deg']:+.2f} deg", flush=True)
            if values["episode_reset"]:
                self.reset("nonfinite" if values["nonfinite"] else "fall" if values["fall"] else "timeout")

    def __exit__(self, kind, error, traceback):
        with self.lock:
            if error is not None:
                self.event("error", message=str(error))
            self.event("evaluation_end")
            summary = {
                "schema_version": 1, "steps": self.total_steps,
                "session_id": self.directory.name,
                "simulated_duration_s": self.total_steps * self.live.metadata["control_period_s"],
                "wall_duration_s": time.monotonic() - self.started,
                "valid_steps": self.valid_steps, "events": dict(self.events),
                "means": {key: value / self.valid_steps for key, value in self.sums.items()} if self.valid_steps else {},
                "maximum_abs_torso_pitch_deg": self.maximum_abs_pitch,
                "minimum_box_gap_m": self.minimum_box_gap,
                "minimum_CAD_convex_gap_m": self.minimum_cad_gap,
                "error": str(error) if error is not None else None,
            }
            (self.directory / "summary.json").write_text(json.dumps(json_value(summary), indent=2, allow_nan=False) + "\n")
            self.stream.close()
            self.stream = None
