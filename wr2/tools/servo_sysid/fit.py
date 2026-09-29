"""Fit the identifiable HTD-45H position-servo dynamics from captures."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from wr2.tools.servo_sysid.core import DEFAULT_FIXTURE, file_sha256


@dataclass(frozen=True)
class ServoDynamics:
    kp: float = 31.902
    effective_velocity_damping: float = 1.60618
    armature: float = 0.024992
    frictionloss: float = 0.324094
    delay_steps: int = 0


@dataclass(frozen=True)
class Trace:
    path: Path
    condition_id: str
    command_rad: np.ndarray
    position_rad: np.ndarray
    timestamps_s: np.ndarray
    fixture_direction: int
    fixture_offset_rad: float
    fixture_sha256: str


def load_trace(path: Path) -> Trace:
    resolved = path.expanduser().resolve()
    manifest_path = resolved.with_suffix(".json")
    if not resolved.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"capture pair is incomplete: {resolved}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("outcome") != "completed":
        raise ValueError(f"capture did not complete: {manifest_path}")
    with np.load(resolved, allow_pickle=False) as values:
        command = np.asarray(values["command_rad"], dtype=np.float64)
        position = np.asarray(values["measured_position_rad"], dtype=np.float64)
        timestamps = np.asarray(values["timestamps_s"], dtype=np.float64)
    if (
        command.size < 10
        or command.shape != position.shape
        or command.shape != timestamps.shape
    ):
        raise ValueError(f"capture has invalid arrays: {resolved}")
    if not np.all(np.isfinite(command)) or not np.all(np.isfinite(position)):
        raise ValueError(f"capture has invalid samples: {resolved}")
    return Trace(
        path=resolved,
        condition_id=str(manifest["condition_id"]),
        command_rad=command,
        position_rad=position,
        timestamps_s=timestamps,
        fixture_direction=int(manifest.get("fixture_direction", 1)),
        fixture_offset_rad=math.radians(
            float(manifest.get("fixture_qpos_offset_deg", 0.0))
        ),
        fixture_sha256=str(manifest["fixture_sha256"]),
    )


class Replay:
    def __init__(
        self, fixture: Path, trace: Trace, *, sim_dt: float, force_limit_nm: float
    ):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(fixture))
        self.data = mujoco.MjData(self.model)
        self.trace = trace
        self.model.opt.timestep = sim_dt
        self.joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pitch"
        )
        self.actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch"
        )
        self.qadr = int(self.model.jnt_qposadr[self.joint_id])
        self.dadr = int(self.model.jnt_dofadr[self.joint_id])
        sample_dt = float(np.median(np.diff(trace.timestamps_s)))
        self.steps_per_sample = max(1, round(sample_dt / sim_dt))
        if abs(self.steps_per_sample * sim_dt - sample_dt) > max(5e-4, 0.1 * sim_dt):
            raise ValueError(
                f"sample period {sample_dt} is incompatible with dt={sim_dt}"
            )
        self.model.actuator_forcerange[self.actuator_id] = (
            -force_limit_nm,
            force_limit_nm,
        )

    def simulate(self, parameters: ServoDynamics) -> np.ndarray:
        model = self.model
        data = self.data
        trace = self.trace
        model.dof_damping[self.dadr] = parameters.effective_velocity_damping
        model.dof_armature[self.dadr] = parameters.armature
        model.dof_frictionloss[self.dadr] = parameters.frictionloss
        model.actuator_gainprm[self.actuator_id, 0] = parameters.kp
        model.actuator_biasprm[self.actuator_id, 1] = -parameters.kp
        model.actuator_biasprm[self.actuator_id, 2] = 0.0
        direction = trace.fixture_direction
        offset = trace.fixture_offset_rad
        self.mujoco.mj_resetData(model, data)
        data.qpos[self.qadr] = offset + direction * trace.position_rad[0]
        self.mujoco.mj_forward(model, data)
        predicted = np.empty_like(trace.position_rad)
        predicted[0] = trace.position_rad[0]
        for index in range(1, predicted.size):
            command_index = max(0, index - 1 - parameters.delay_steps)
            data.ctrl[self.actuator_id] = (
                offset + direction * trace.command_rad[command_index]
            )
            self.mujoco.mj_step(model, data, nstep=self.steps_per_sample)
            predicted[index] = direction * (data.qpos[self.qadr] - offset)
        return predicted


def _parameters(log_values: np.ndarray, delay_steps: int) -> ServoDynamics:
    kp, damping, armature, frictionloss = np.exp(log_values)
    return ServoDynamics(
        float(kp), float(damping), float(armature), float(frictionloss), delay_steps
    )


def _metrics(replays: Sequence[Replay], parameters: ServoDynamics) -> dict[str, Any]:
    captures: dict[str, Any] = {}
    rmses = []
    for replay in replays:
        error = np.degrees(replay.simulate(parameters) - replay.trace.position_rad)
        rmse = float(np.sqrt(np.mean(error**2)))
        rmses.append(rmse)
        captures[replay.trace.condition_id] = {
            "rmse_deg": rmse,
            "abs_p95_deg": float(np.percentile(np.abs(error), 95.0)),
            "bias_deg": float(np.mean(error)),
        }
    return {
        "mean_rmse_deg": float(np.mean(rmses)),
        "max_rmse_deg": float(np.max(rmses)),
        "captures": captures,
    }


def fit(
    replays: Sequence[Replay],
    *,
    delays: Sequence[int],
    max_nfev: int,
) -> tuple[ServoDynamics, dict[str, Any]]:
    initial = np.log([31.902, 1.60618, 0.024992, 0.324094])
    lower = np.log([5.0, 0.01, 0.001, 0.001])
    upper = np.log([100.0, 5.0, 0.15, 1.0])
    best = None
    candidates = []
    for delay in delays:

        def residual(values: np.ndarray) -> np.ndarray:
            parameters = _parameters(values, delay)
            parts = []
            for replay in replays:
                error = replay.simulate(parameters) - replay.trace.position_rad
                parts.append(error / math.sqrt(error.size))
            return np.concatenate(parts)

        result = least_squares(
            residual,
            np.clip(initial, lower, upper),
            bounds=(lower, upper),
            x_scale="jac",
            diff_step=1e-3,
            max_nfev=max_nfev,
        )
        parameters = _parameters(result.x, delay)
        item = {
            "cost": float(result.cost),
            "parameters": asdict(parameters),
            "metrics": _metrics(replays, parameters),
        }
        candidates.append(item)
        if best is None or result.cost < best[0].cost:
            best = (result, parameters)
    assert best is not None
    result, parameters = best
    uncertainty: dict[str, float | None] = {}
    names = ("kp", "effective_velocity_damping", "armature", "frictionloss")
    try:
        degrees_of_freedom = max(1, result.fun.size - result.x.size)
        covariance = np.linalg.inv(result.jac.T @ result.jac) * (
            2.0 * result.cost / degrees_of_freedom
        )
        sigma = np.sqrt(np.diag(covariance))
        uncertainty = {
            name: float(math.exp(1.96 * value)) for name, value in zip(names, sigma)
        }
    except np.linalg.LinAlgError:
        uncertainty = dict.fromkeys(names)
    return parameters, {
        "candidates": candidates,
        "approximate_95pct_factor": uncertainty,
    }


def _delay_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("delays must be non-negative integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_captures", nargs="+", type=Path)
    parser.add_argument("--validation-capture", action="append", type=Path, default=[])
    parser.add_argument("--fixture-mjcf", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--force-limit-nm", type=float, default=4.0)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--delay-steps", type=_delay_list, default=(0, 1, 2, 3))
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixture = args.fixture_mjcf.expanduser().resolve()
    fixture_hash = file_sha256(fixture)
    fit_traces = [load_trace(path) for path in args.fit_captures]
    validation_traces = [load_trace(path) for path in args.validation_capture]
    for trace in fit_traces + validation_traces:
        if trace.fixture_sha256 != fixture_hash:
            raise ValueError(f"fixture hash mismatch for {trace.path}")
    centers = [
        json.loads(trace.path.with_suffix(".json").read_text())["center_deg"]
        for trace in fit_traces
    ]
    if not any(abs(float(value)) < 1e-6 for value in centers) or not (
        min(centers) < 0 < max(centers)
    ):
        raise ValueError("fit captures require zero, positive, and negative load")
    fit_replays = [
        Replay(fixture, trace, sim_dt=args.sim_dt, force_limit_nm=args.force_limit_nm)
        for trace in fit_traces
    ]
    validation_replays = [
        Replay(fixture, trace, sim_dt=args.sim_dt, force_limit_nm=args.force_limit_nm)
        for trace in validation_traces
    ]
    parameters, optimization = fit(
        fit_replays, delays=args.delay_steps, max_nfev=args.max_nfev
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "effective_position_servo",
        "parameters": asdict(parameters),
        "fit_metrics": _metrics(fit_replays, parameters),
        "validation_metrics": (
            _metrics(validation_replays, parameters) if validation_replays else None
        ),
        "optimization": optimization,
        "fixture_mjcf": str(fixture),
        "fixture_sha256": fixture_hash,
        "force_limit_nm_held_fixed": args.force_limit_nm,
        "identified": [
            "effective_position_kp",
            "effective_total_velocity_damping",
            "armature_with_known_fixture_inertia",
            "frictionloss",
            "whole-response_delay_steps",
        ],
        "not_identified": [
            "continuous_torque",
            "voltage_temperature_dependent_torque_speed_envelope",
            "pure_transport_delay_separate_from_servo_response",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["parameters"], indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
