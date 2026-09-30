"""Safely capture one HTD-45H known-load fixture condition.

The command is preflight-only unless both ``--execute`` and
``--confirm-fixture-safe`` are supplied. Hardware execution controls exactly
one servo and always attempts a gravity-neutral return before unloading it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Protocol, Sequence

import numpy as np

from wr2.actuation.htd45h import Htd45hBus, SerialTransport, SerialTransportConfig
from wr2.tools.servo_sysid.core import (
    DEFAULT_FIXTURE,
    FixtureModel,
    ProfileSegment,
    SERVO_CENTER_UNIT,
    SERVO_RANGE_RAD,
    build_profile,
    file_sha256,
    radians_to_units,
    summarize_trace,
    units_to_radians,
)


class ServoBus(Protocol):
    def move(self, servo_id: int, position: int, duration_ms: int) -> None: ...

    def move_stop(self, servo_id: int) -> None: ...

    def set_loaded(self, servo_id: int, loaded: bool) -> None: ...

    def read_position(self, servo_id: int) -> int | None: ...

    def read_voltage_v(self, servo_id: int) -> float | None: ...

    def read_temperature_c(self, servo_id: int) -> int | None: ...

    def read_loaded(self, servo_id: int) -> bool | None: ...

    def read_id(self, servo_id: int) -> int | None: ...

    def read_angle_limits(self, servo_id: int) -> tuple[int, int] | None: ...

    def read_voltage_limits_v(self, servo_id: int) -> tuple[float, float] | None: ...

    def read_temperature_limit_c(self, servo_id: int) -> int | None: ...

    def read_motor_mode(self, servo_id: int) -> tuple[int, int] | None: ...

    def close(self) -> None: ...


@dataclass
class CaptureBuffer:
    timestamp_s: list[float] = field(default_factory=list)
    wall_time_s: list[float] = field(default_factory=list)
    scheduled_time_s: list[float] = field(default_factory=list)
    loop_lateness_s: list[float] = field(default_factory=list)
    command_write_duration_s: list[float] = field(default_factory=list)
    position_read_duration_s: list[float] = field(default_factory=list)
    command_age_at_read_s: list[float] = field(default_factory=list)
    segment_name: list[str] = field(default_factory=list)
    command_rad: list[float] = field(default_factory=list)
    position_rad: list[float] = field(default_factory=list)
    command_written: list[bool] = field(default_factory=list)
    voltage_v: list[float] = field(default_factory=list)
    temperature_c: list[float] = field(default_factory=list)
    loaded: list[float] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        position = np.asarray(self.position_rad, dtype=np.float64)
        timestamp = np.asarray(self.timestamp_s, dtype=np.float64)
        velocity = (
            np.gradient(position, timestamp)
            if position.size >= 3
            else np.zeros_like(position)
        )
        return {
            "timestamps_s": timestamp,
            "host_wall_time_s": np.asarray(self.wall_time_s, dtype=np.float64),
            "scheduled_time_s": np.asarray(self.scheduled_time_s, dtype=np.float64),
            "loop_lateness_s": np.asarray(self.loop_lateness_s, dtype=np.float64),
            "command_write_duration_s": np.asarray(
                self.command_write_duration_s, dtype=np.float64
            ),
            "position_read_duration_s": np.asarray(
                self.position_read_duration_s, dtype=np.float64
            ),
            "command_age_at_read_s": np.asarray(
                self.command_age_at_read_s, dtype=np.float64
            ),
            "segment_name": np.asarray(self.segment_name, dtype="U64"),
            "command_rad": np.asarray(self.command_rad, dtype=np.float64),
            "measured_position_rad": position,
            "measured_velocity_rad_s": velocity,
            "command_written": np.asarray(self.command_written, dtype=bool),
            "voltage_v": np.asarray(self.voltage_v, dtype=np.float64),
            "temperature_c": np.asarray(self.temperature_c, dtype=np.float64),
            "loaded": np.asarray(self.loaded, dtype=np.float64),
        }


def _parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(token.strip()) for token in value.split(",") if token.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _read_required(read, label: str, attempts: int = 3):
    for _ in range(attempts):
        value = read()
        if value is not None:
            return value
        time.sleep(0.002)
    raise RuntimeError(f"servo {label} telemetry is unavailable")


def _health(bus: ServoBus, servo_id: int) -> dict[str, float | bool]:
    return {
        "voltage_v": float(
            _read_required(lambda: bus.read_voltage_v(servo_id), "voltage")
        ),
        "temperature_c": float(
            _read_required(lambda: bus.read_temperature_c(servo_id), "temperature")
        ),
        "loaded": bool(_read_required(lambda: bus.read_loaded(servo_id), "load-state")),
    }


def _check_health(
    health: dict[str, float | bool], *, min_voltage_v: float, max_temperature_c: float
) -> None:
    if float(health["voltage_v"]) < min_voltage_v:
        raise RuntimeError(
            f"voltage {float(health['voltage_v']):.3f} V is below {min_voltage_v:.3f} V"
        )
    if float(health["temperature_c"]) >= max_temperature_c:
        raise RuntimeError(
            f"temperature {float(health['temperature_c']):.1f} C reaches or exceeds "
            f"the {max_temperature_c:.1f} C limit"
        )
    if not bool(health["loaded"]):
        raise RuntimeError("servo unexpectedly disabled torque")


def _wait_for_cooldown(
    bus: ServoBus,
    servo_id: int,
    *,
    target_c: float,
    timeout_s: float,
    poll_s: float,
    min_voltage_v: float,
) -> list[dict[str, float]]:
    started = time.monotonic()
    samples: list[dict[str, float]] = []
    next_progress_s = 0.0
    print(
        f"COOLDOWN: torque is disabled; waiting for temperature <= {target_c:.1f} C "
        f"(timeout {timeout_s:.0f} s).",
        flush=True,
    )
    while True:
        voltage = float(_read_required(lambda: bus.read_voltage_v(servo_id), "voltage"))
        temperature = float(
            _read_required(lambda: bus.read_temperature_c(servo_id), "temperature")
        )
        elapsed = time.monotonic() - started
        samples.append(
            {"elapsed_s": elapsed, "voltage_v": voltage, "temperature_c": temperature}
        )
        if elapsed >= next_progress_s:
            print(
                f"COOLDOWN STATUS: {temperature:.1f} C, {voltage:.3f} V, "
                f"elapsed {elapsed:.0f} s.",
                flush=True,
            )
            next_progress_s = elapsed + 30.0
        if voltage < min_voltage_v:
            raise RuntimeError(f"cooldown voltage {voltage:.3f} V is too low")
        if temperature <= target_c:
            print(
                f"COOLDOWN COMPLETE: {temperature:.1f} C after {elapsed:.0f} s.",
                flush=True,
            )
            return samples
        if elapsed >= timeout_s:
            raise RuntimeError(f"cooldown timed out at {temperature:.1f} C")
        time.sleep(poll_s)


def _move_and_monitor(
    bus: ServoBus,
    servo_id: int,
    target_rad: float,
    *,
    speed_deg_s: float,
    settle_s: float,
    min_voltage_v: float,
    max_temperature_c: float,
    max_position_error_deg: float,
) -> list[dict[str, float | bool]]:
    start_units = int(_read_required(lambda: bus.read_position(servo_id), "position"))
    start_rad = units_to_radians(start_units)
    duration_s = abs(math.degrees(target_rad - start_rad)) / speed_deg_s
    duration_ms = max(20, round(1000.0 * duration_s))
    bus.move(servo_id, start_units, 0)
    bus.set_loaded(servo_id, True)
    if not bool(_read_required(lambda: bus.read_loaded(servo_id), "load-state")):
        raise RuntimeError("servo failed to enable torque")
    bus.move(servo_id, radians_to_units(target_rad), duration_ms)
    samples: list[dict[str, float | bool]] = []
    deadline = time.monotonic() + duration_s + settle_s
    while time.monotonic() < deadline:
        position = units_to_radians(
            int(_read_required(lambda: bus.read_position(servo_id), "position"))
        )
        health = _health(bus, servo_id)
        _check_health(
            health,
            min_voltage_v=min_voltage_v,
            max_temperature_c=max_temperature_c,
        )
        samples.append(
            {
                "elapsed_s": time.monotonic() - (deadline - duration_s - settle_s),
                "position_rad": position,
                **health,
            }
        )
        time.sleep(0.1)
    final_rad = units_to_radians(
        int(_read_required(lambda: bus.read_position(servo_id), "position"))
    )
    if abs(math.degrees(target_rad - final_rad)) > max_position_error_deg:
        raise RuntimeError(
            f"settled position error is {math.degrees(target_rad - final_rad):+.2f} deg"
        )
    return samples


def capture_profile(
    bus: ServoBus,
    servo_id: int,
    segments: Sequence[ProfileSegment],
    *,
    sample_hz: float,
    move_time_ms: int,
    write_deadband_units: int,
    health_poll_hz: float,
    min_voltage_v: float,
    max_temperature_c: float,
    max_position_error_deg: float,
    max_position_error_duration_s: float,
    buffer: CaptureBuffer | None = None,
) -> CaptureBuffer:
    buffer = buffer or CaptureBuffer()
    period_s = 1.0 / sample_hz
    last_health = _health(bus, servo_id)
    _check_health(
        last_health,
        min_voltage_v=min_voltage_v,
        max_temperature_c=max_temperature_c,
    )
    health_fields = (
        ("voltage_v", lambda: bus.read_voltage_v(servo_id)),
        ("temperature_c", lambda: bus.read_temperature_c(servo_id)),
        ("loaded", lambda: bus.read_loaded(servo_id)),
    )
    health_index = 0
    error_violation_count = 0
    error_violation_limit = max(1, math.ceil(max_position_error_duration_s * sample_hz))
    started = time.monotonic()
    next_health_s = 0.0
    last_units: int | None = None
    last_write_completed = started
    sample_index = 0
    for segment in segments:
        for target_rad in segment.targets_rad:
            scheduled_elapsed = sample_index * period_s
            scheduled = started + scheduled_elapsed
            time.sleep(max(0.0, scheduled - time.monotonic()))
            loop_lateness = max(0.0, time.monotonic() - scheduled)
            target_units = radians_to_units(float(target_rad))
            written = (
                last_units is None
                or abs(target_units - last_units) > write_deadband_units
            )
            write_duration = np.nan
            if written:
                write_started = time.monotonic()
                bus.move(servo_id, target_units, move_time_ms)
                last_write_completed = time.monotonic()
                write_duration = last_write_completed - write_started
                last_units = target_units
            read_started = time.monotonic()
            position_units = int(
                _read_required(lambda: bus.read_position(servo_id), "position")
            )
            read_completed = time.monotonic()
            position_rad = units_to_radians(position_units)
            elapsed = read_completed - started
            if health_poll_hz > 0.0 and elapsed >= next_health_s:
                field_name, read = health_fields[health_index % len(health_fields)]
                last_health[field_name] = _read_required(read, field_name)
                health_index += 1
                _check_health(
                    last_health,
                    min_voltage_v=min_voltage_v,
                    max_temperature_c=max_temperature_c,
                )
                next_health_s = elapsed + 1.0 / health_poll_hz
            buffer.timestamp_s.append(elapsed)
            buffer.wall_time_s.append(time.time())
            buffer.scheduled_time_s.append(scheduled_elapsed)
            buffer.loop_lateness_s.append(loop_lateness)
            buffer.command_write_duration_s.append(write_duration)
            buffer.position_read_duration_s.append(read_completed - read_started)
            buffer.command_age_at_read_s.append(read_completed - last_write_completed)
            buffer.segment_name.append(segment.name)
            buffer.command_rad.append(float(target_rad))
            buffer.position_rad.append(position_rad)
            buffer.command_written.append(written)
            buffer.voltage_v.append(float(last_health["voltage_v"]))
            buffer.temperature_c.append(float(last_health["temperature_c"]))
            buffer.loaded.append(float(last_health["loaded"]))
            error_deg = abs(math.degrees(float(target_rad) - position_rad))
            error_violation_count = (
                error_violation_count + 1 if error_deg > max_position_error_deg else 0
            )
            if error_violation_count >= error_violation_limit:
                raise RuntimeError(
                    f"tracking error exceeded {max_position_error_deg:.2f} deg for "
                    f"{error_violation_count / sample_hz:.3f} s in {segment.name}"
                )
            sample_index += 1
    return buffer


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _health_summary(arrays: dict[str, np.ndarray]) -> dict[str, float | None]:
    voltage = _finite(arrays["voltage_v"])
    temperature = _finite(arrays["temperature_c"])
    timestamps = arrays["timestamps_s"][np.isfinite(arrays["temperature_c"])]
    slope = None
    if temperature.size >= 2:
        start = max(0.0, float(timestamps[-1]) - 60.0)
        mask = timestamps >= start
        if np.count_nonzero(mask) >= 2 and np.ptp(timestamps[mask]) > 0.0:
            slope = float(np.polyfit(timestamps[mask], temperature[mask], 1)[0] * 60.0)
    return {
        "min_voltage_v": float(np.min(voltage)) if voltage.size else None,
        "max_temperature_c": float(np.max(temperature)) if temperature.size else None,
        "temperature_rise_c": (
            float(temperature[-1] - temperature[0]) if temperature.size else None
        ),
        "temperature_slope_c_per_min_last_60s": slope,
    }


def _timing_summary(arrays: dict[str, np.ndarray]) -> dict[str, float | None]:
    def milliseconds(name: str, percentile: float | None = None) -> float | None:
        values = _finite(arrays[name])
        if not values.size:
            return None
        value = (
            np.max(values) if percentile is None else np.percentile(values, percentile)
        )
        return 1000.0 * float(value)

    return {
        "loop_lateness_max_ms": milliseconds("loop_lateness_s"),
        "sample_period_p95_ms": (
            1000.0 * float(np.percentile(np.diff(arrays["timestamps_s"]), 95.0))
            if arrays["timestamps_s"].size >= 2
            else None
        ),
        "command_write_p95_ms": milliseconds("command_write_duration_s", 95.0),
        "position_read_p95_ms": milliseconds("position_read_duration_s", 95.0),
        "command_age_at_read_p95_ms": milliseconds("command_age_at_read_s", 95.0),
    }


def _write_artifacts(
    output: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]
) -> None:
    output = output.expanduser().resolve()
    json_path = output.with_suffix(".json")
    if output.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {json_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, schema_version=np.asarray(1), **arrays)
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servo-id", type=int, required=True)
    parser.add_argument("--board-port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fixture-mjcf", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-joint", default="pitch")
    parser.add_argument("--fixture-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--fixture-qpos-offset-deg", type=float, default=0.0)
    parser.add_argument("--center-deg", type=float, required=True)
    parser.add_argument(
        "--amplitudes-deg", type=_parse_float_list, default=(2.0, 5.0, 8.0)
    )
    parser.add_argument("--chirp-start-hz", type=float, default=0.1)
    parser.add_argument("--chirp-end-hz", type=float, default=2.0)
    parser.add_argument("--chirp-duration-s", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--move-time-ms", type=int, default=20)
    parser.add_argument("--write-deadband-units", type=int, default=0)
    parser.add_argument("--constant-hold-s", type=float)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-speed-deg-s", type=float, default=20.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--return-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--cooldown-target-c", type=float, default=35.0)
    parser.add_argument("--cooldown-timeout-s", type=float, default=900.0)
    parser.add_argument("--cooldown-poll-s", type=float, default=5.0)
    parser.add_argument("--profile-health-poll-hz", type=float, default=6.0)
    parser.add_argument("--min-voltage-v", type=float, default=9.6)
    parser.add_argument("--max-temperature-c", type=float, default=80.0)
    parser.add_argument("--max-position-error-deg", type=float, default=5.0)
    parser.add_argument("--max-position-error-duration-s", type=float, default=0.15)
    parser.add_argument("--max-static-torque-nm", type=float, default=3.2)
    parser.add_argument("--max-predicted-torque-nm", type=float, default=3.2)
    parser.add_argument("--required-static-torque-nm", type=float)
    parser.add_argument("--startup-delay-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--servo-label")
    parser.add_argument("--fixture-label", default="wr2-htd45h-2650g")
    parser.add_argument("--external-log-label")
    parser.add_argument("--measured-weight-kg", type=float)
    parser.add_argument("--measured-com-radius-m", type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-fixture-safe", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "baudrate": args.baudrate,
        "sample_hz": args.sample_hz,
        "chirp_duration_s": args.chirp_duration_s,
        "prepare_speed_deg_s": args.prepare_speed_deg_s,
        "return_speed_deg_s": args.return_speed_deg_s,
        "cooldown_target_c": args.cooldown_target_c,
        "cooldown_poll_s": args.cooldown_poll_s,
        "min_voltage_v": args.min_voltage_v,
        "max_temperature_c": args.max_temperature_c,
        "max_position_error_deg": args.max_position_error_deg,
        "max_position_error_duration_s": args.max_position_error_duration_s,
        "max_static_torque_nm": args.max_static_torque_nm,
        "max_predicted_torque_nm": args.max_predicted_torque_nm,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    non_negative = {
        "chirp_start_hz": args.chirp_start_hz,
        "chirp_end_hz": args.chirp_end_hz,
        "settle_s": args.settle_s,
        "cooldown_timeout_s": args.cooldown_timeout_s,
        "profile_health_poll_hz": args.profile_health_poll_hz,
        "startup_delay_s": args.startup_delay_s,
    }
    for name, value in non_negative.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if not 0 <= args.servo_id <= 253:
        raise SystemExit("--servo-id must be between 0 and 253")
    if not math.isfinite(args.center_deg):
        raise SystemExit("--center-deg must be finite")
    if not math.isfinite(args.fixture_qpos_offset_deg):
        raise SystemExit("--fixture-qpos-offset-deg must be finite")
    if any(not math.isfinite(value) or value <= 0.0 for value in args.amplitudes_deg):
        raise SystemExit("--amplitudes-deg values must be finite and positive")
    if not 0 <= args.move_time_ms <= 30000:
        raise SystemExit("--move-time-ms must be between 0 and 30000")
    if args.write_deadband_units < 0:
        raise SystemExit("--write-deadband-units must be non-negative")
    if args.cooldown_target_c > args.max_temperature_c:
        raise SystemExit("--cooldown-target-c must not exceed --max-temperature-c")
    if args.constant_hold_s is not None and (
        not math.isfinite(args.constant_hold_s) or args.constant_hold_s <= 0.0
    ):
        raise SystemExit("--constant-hold-s must be finite and positive")
    if args.required_static_torque_nm is not None and (
        not math.isfinite(args.required_static_torque_nm)
        or args.required_static_torque_nm <= 0.0
    ):
        raise SystemExit("--required-static-torque-nm must be finite and positive")
    if args.execute:
        assert args.measured_weight_kg is not None
        assert args.measured_com_radius_m is not None
        if not math.isfinite(args.measured_weight_kg) or args.measured_weight_kg <= 0.0:
            raise SystemExit("--measured-weight-kg must be finite and positive")
        if (
            not math.isfinite(args.measured_com_radius_m)
            or args.measured_com_radius_m <= 0.0
        ):
            raise SystemExit("--measured-com-radius-m must be finite and positive")


def _profile_unit_range(segments: Sequence[ProfileSegment]) -> tuple[int, int]:
    positions = np.concatenate([segment.targets_rad for segment in segments])
    if (
        not np.all(np.isfinite(positions))
        or float(np.min(positions)) < SERVO_RANGE_RAD[0]
        or float(np.max(positions)) > SERVO_RANGE_RAD[1]
    ):
        raise SystemExit("commanded profile exceeds the servo's +/-120 degree range")
    units = np.asarray([radians_to_units(value) for value in positions])
    # Every hardware condition first visits the gravity-neutral zero position.
    return min(SERVO_CENTER_UNIT, int(np.min(units))), max(
        SERVO_CENTER_UNIT, int(np.max(units))
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and not args.confirm_fixture_safe:
        raise SystemExit("--execute requires --confirm-fixture-safe")
    if args.execute and (
        args.measured_weight_kg is None or args.measured_com_radius_m is None
    ):
        raise SystemExit(
            "--execute requires --measured-weight-kg and --measured-com-radius-m"
        )
    if args.execute and not math.isclose(
        args.measured_weight_kg, 2.650, rel_tol=0.01, abs_tol=0.0
    ):
        raise SystemExit("measured weight differs by >1%; update the fixture MJCF")
    if args.execute and not math.isclose(
        args.measured_com_radius_m, 0.1204, rel_tol=0.01, abs_tol=0.0
    ):
        raise SystemExit("measured COM radius differs by >1%; update the fixture MJCF")
    if args.prepare_only and args.constant_hold_s is not None:
        raise SystemExit("--prepare-only and --constant-hold-s are mutually exclusive")
    _validate_args(args)

    fixture = FixtureModel.load(
        args.fixture_mjcf,
        joint_name=args.fixture_joint,
        direction=args.fixture_direction,
        qpos_offset_rad=math.radians(args.fixture_qpos_offset_deg),
    )
    segments = build_profile(
        center_deg=args.center_deg,
        amplitudes_deg=args.amplitudes_deg,
        sample_hz=args.sample_hz,
        settle_s=args.settle_s,
        chirp_start_hz=args.chirp_start_hz,
        chirp_end_hz=args.chirp_end_hz,
        chirp_duration_s=args.chirp_duration_s,
        constant_hold_s=(
            max(1.0, args.settle_s) if args.prepare_only else args.constant_hold_s
        ),
    )
    required_angle_units = _profile_unit_range(segments)
    center_torque = float(
        fixture.evaluate_static([math.radians(args.center_deg)])[0][0]
    )
    envelope = fixture.dynamic_envelope(segments, sample_hz=args.sample_hz)
    tested_static = (
        abs(center_torque) if args.prepare_only else envelope["peak_static_torque_nm"]
    )
    if envelope["peak_static_torque_nm"] > args.max_static_torque_nm:
        raise SystemExit("static fixture torque exceeds --max-static-torque-nm")
    if envelope["peak_total_torque_nm"] > args.max_predicted_torque_nm:
        raise SystemExit(
            "predicted gravity-plus-inertia torque exceeds --max-predicted-torque-nm"
        )
    if (
        args.required_static_torque_nm
        and tested_static < args.required_static_torque_nm
    ):
        raise SystemExit("fixture does not reach --required-static-torque-nm")

    print(f"condition={args.condition_id} center={args.center_deg:+.1f} deg")
    print(
        f"fixture_mass={fixture.moving_mass_kg:.6f} kg "
        f"center_static={center_torque:+.3f} N m"
    )
    print(
        "preflight: "
        f"static={envelope['peak_static_torque_nm']:.3f} N m, "
        f"inertial={envelope['peak_inertial_torque_nm']:.3f} N m, "
        f"combined={envelope['peak_total_torque_nm']:.3f} N m, "
        f"speed={envelope['peak_speed_rad_s']:.3f} rad/s"
    )
    if not args.execute:
        print(
            "Preflight passed; no serial port was opened. Add --execute and --confirm-fixture-safe to move hardware."
        )
        return 0

    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise SystemExit(f"refusing to overwrite {output}")

    print(f"Hardware starts in {args.startup_delay_s:.1f} s; Ctrl-C to abort.")
    time.sleep(args.startup_delay_s)
    bus: ServoBus = Htd45hBus(
        SerialTransport(SerialTransportConfig(args.board_port, args.baudrate))
    )
    buffer = CaptureBuffer()
    outcome = "failed"
    error: str | None = None
    cooldown: list[dict[str, float]] = []
    preparation: list[dict[str, float | bool]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    servo_eeprom: dict[str, object] = {}
    try:
        if bus.read_id(args.servo_id) != args.servo_id:
            raise RuntimeError(f"servo {args.servo_id} did not respond")
        bus.set_loaded(args.servo_id, False)
        if bool(_read_required(lambda: bus.read_loaded(args.servo_id), "load-state")):
            raise RuntimeError("servo failed to disable torque before cooldown")
        angle_limits = _read_required(
            lambda: bus.read_angle_limits(args.servo_id), "EEPROM angle limits"
        )
        voltage_limits = _read_required(
            lambda: bus.read_voltage_limits_v(args.servo_id), "EEPROM voltage limits"
        )
        temperature_limit = int(
            _read_required(
                lambda: bus.read_temperature_limit_c(args.servo_id),
                "EEPROM temperature limit",
            )
        )
        motor_mode = _read_required(
            lambda: bus.read_motor_mode(args.servo_id), "motor mode"
        )
        servo_eeprom = {
            "angle_limits_units": list(angle_limits),
            "voltage_limits_v": list(voltage_limits),
            "temperature_limit_c": temperature_limit,
            "motor_mode": list(motor_mode),
        }
        if (
            required_angle_units[0] < angle_limits[0]
            or required_angle_units[1] > angle_limits[1]
        ):
            raise RuntimeError(
                f"profile units {required_angle_units} exceed EEPROM angle limits "
                f"{angle_limits}"
            )
        if args.max_temperature_c > temperature_limit:
            raise RuntimeError(
                f"software temperature limit {args.max_temperature_c:.1f} C exceeds "
                f"EEPROM limit {temperature_limit} C"
            )
        if motor_mode[0] != 0:
            raise RuntimeError(
                f"servo is in motor mode {motor_mode}; position mode required"
            )
        cooldown = _wait_for_cooldown(
            bus,
            args.servo_id,
            target_c=args.cooldown_target_c,
            timeout_s=args.cooldown_timeout_s,
            poll_s=args.cooldown_poll_s,
            min_voltage_v=args.min_voltage_v,
        )
        cooldown_voltage = float(cooldown[-1]["voltage_v"])
        if not voltage_limits[0] <= cooldown_voltage <= voltage_limits[1]:
            raise RuntimeError(
                f"servo voltage {cooldown_voltage:.3f} V is outside EEPROM limits "
                f"{voltage_limits}"
            )
        print("PREPARATION: moving servo to the neutral 0.0 deg position.", flush=True)
        preparation.extend(
            _move_and_monitor(
                bus,
                args.servo_id,
                0.0,
                speed_deg_s=5.0,
                settle_s=1.0,
                min_voltage_v=args.min_voltage_v,
                max_temperature_c=args.max_temperature_c,
                max_position_error_deg=args.max_position_error_deg,
            )
        )
        if args.prepare_only:
            print(
                f"TEST START: moving to {args.center_deg:+.1f} deg at "
                f"{args.prepare_speed_deg_s:.1f} deg/s, then holding for "
                f"{args.settle_s:.1f} s. Safety monitoring is active.",
                flush=True,
            )
        else:
            print(
                f"PREPARATION: moving servo to the test center "
                f"{args.center_deg:+.1f} deg.",
                flush=True,
            )
        preparation.extend(
            _move_and_monitor(
                bus,
                args.servo_id,
                math.radians(args.center_deg),
                speed_deg_s=args.prepare_speed_deg_s,
                settle_s=args.settle_s,
                min_voltage_v=args.min_voltage_v,
                max_temperature_c=args.max_temperature_c,
                max_position_error_deg=args.max_position_error_deg,
            )
        )
        if not args.prepare_only:
            profile_duration_s = (
                sum(len(segment.targets_rad) for segment in segments) / args.sample_hz
            )
            profile_description = (
                f"holding {args.center_deg:+.1f} deg"
                if args.constant_hold_s is not None
                else "running the dynamic position profile"
            )
            print(
                f"TEST START: {profile_description} for {profile_duration_s:.1f} s; "
                f"shutdown temperature is {args.max_temperature_c:.1f} C. "
                "Safety monitoring is active; capture may be quiet until completion.",
                flush=True,
            )
            buffer = capture_profile(
                bus,
                args.servo_id,
                segments,
                sample_hz=args.sample_hz,
                move_time_ms=args.move_time_ms,
                write_deadband_units=args.write_deadband_units,
                health_poll_hz=args.profile_health_poll_hz,
                min_voltage_v=args.min_voltage_v,
                max_temperature_c=args.max_temperature_c,
                max_position_error_deg=args.max_position_error_deg,
                max_position_error_duration_s=args.max_position_error_duration_s,
                buffer=buffer,
            )
        print("TEST COMPLETE: commanded test duration finished.", flush=True)
        outcome = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt):
            outcome = "aborted"
        else:
            print(error, file=sys.stderr)
    finally:
        try:
            bus.move_stop(args.servo_id)
        except BaseException as stop_exc:
            error = f"{error or ''}; move-stop failed: {stop_exc}".strip("; ")
            outcome = "failed"
        try:
            if outcome == "completed":
                print(
                    "RETURN: moving servo to neutral 0.0 deg before torque-off.",
                    flush=True,
                )
                _move_and_monitor(
                    bus,
                    args.servo_id,
                    0.0,
                    speed_deg_s=args.return_speed_deg_s,
                    settle_s=1.0,
                    min_voltage_v=args.min_voltage_v,
                    max_temperature_c=args.max_temperature_c,
                    max_position_error_deg=args.max_position_error_deg,
                )
                print("RETURN COMPLETE: neutral position reached.", flush=True)
            else:
                print(
                    "SAFE STOP: test did not complete; skipping controlled return and "
                    "disabling torque.",
                    flush=True,
                )
        except BaseException as unload_exc:
            error = f"{error or ''}; unload return failed: {unload_exc}".strip("; ")
            outcome = "failed"
        finally:
            try:
                bus.set_loaded(args.servo_id, False)
                if bool(
                    _read_required(lambda: bus.read_loaded(args.servo_id), "load-state")
                ):
                    raise RuntimeError("torque-off could not be verified")
                print("TORQUE OFF: servo load state verified disabled.", flush=True)
            except BaseException as unload_exc:
                error = f"{error or ''}; torque-off failed: {unload_exc}".strip("; ")
                outcome = "failed"
            finally:
                bus.close()

    arrays = buffer.arrays()
    trace_summary = (
        summarize_trace(
            arrays["command_rad"],
            arrays["measured_position_rad"],
            arrays["timestamps_s"],
        )
        if arrays["command_rad"].size
        else {}
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "condition_id": args.condition_id,
        "outcome": outcome,
        "error": error,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "servo_id": args.servo_id,
        "servo_label": args.servo_label or f"htd45h-{args.servo_id}",
        "servo_eeprom": servo_eeprom,
        "board_port": args.board_port,
        "fixture_label": args.fixture_label,
        "fixture_mjcf": str(fixture.path),
        "fixture_sha256": file_sha256(fixture.path),
        "fixture_direction": args.fixture_direction,
        "fixture_qpos_offset_deg": args.fixture_qpos_offset_deg,
        "fixture_moving_mass_kg": fixture.moving_mass_kg,
        "center_deg": args.center_deg,
        "center_static_torque_nm": center_torque,
        "tested_static_torque_nm": tested_static,
        "predicted_envelope": envelope,
        "sample_hz": args.sample_hz,
        "write_deadband_units": args.write_deadband_units,
        "constant_hold_s": args.constant_hold_s,
        "cooldown": cooldown,
        "preparation": preparation,
        "trace_summary": trace_summary,
        "health_summary": _health_summary(arrays),
        "limits": {
            "min_voltage_v": args.min_voltage_v,
            "max_temperature_c": args.max_temperature_c,
            "max_position_error_deg": args.max_position_error_deg,
            "max_position_error_duration_s": args.max_position_error_duration_s,
            "max_static_torque_nm": args.max_static_torque_nm,
            "max_predicted_torque_nm": args.max_predicted_torque_nm,
        },
        "external_log_label": args.external_log_label,
        "measured_weight_kg": args.measured_weight_kg,
        "measured_com_radius_m": args.measured_com_radius_m,
        "timing_summary": _timing_summary(arrays),
    }
    _write_artifacts(args.output, arrays, metadata)
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")
    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
