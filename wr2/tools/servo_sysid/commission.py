"""Run repeated low-load HTD-45H commissioning captures."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from wr2.tools.servo_sysid.core import DEFAULT_FIXTURE, file_sha256


@dataclass(frozen=True)
class RepeatMetrics:
    repeat_index: int
    outcome: str
    zero_position_deg: float | None
    loaded_position_deg: float | None
    absolute_error_deg: float | None
    incremental_travel_deg: float | None
    incremental_error_deg: float | None
    minimum_voltage_v: float | None
    initial_temperature_c: float | None
    final_temperature_c: float | None
    temperature_rise_c: float | None


def _split_preparation(samples: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not samples:
        return []
    phases: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(1, len(samples)):
        if float(samples[index]["elapsed_s"]) < float(
            samples[index - 1]["elapsed_s"]
        ):
            phases.append(list(samples[start:index]))
            start = index
    phases.append(list(samples[start:]))
    return phases


def _tail_mean(phase: Sequence[dict[str, Any]], duration_s: float) -> float:
    final_time = float(phase[-1]["elapsed_s"])
    values = [
        math.degrees(float(sample["position_rad"]))
        for sample in phase
        if float(sample["elapsed_s"]) >= final_time - duration_s
    ]
    return float(np.mean(values))


def summarize_capture(path: Path, repeat_index: int) -> RepeatMetrics:
    metadata = json.loads(path.read_text())
    phases = _split_preparation(metadata.get("preparation", []))
    if metadata.get("outcome") != "completed" or len(phases) < 2:
        return RepeatMetrics(
            repeat_index,
            str(metadata.get("outcome")),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    zero = _tail_mean(phases[0], 0.5)
    loaded = _tail_mean(phases[1], 1.0)
    center = float(metadata["center_deg"])
    travel = loaded - zero
    voltage = [
        float(sample["voltage_v"])
        for phase in phases
        for sample in phase
        if sample.get("voltage_v") is not None
    ]
    temperature = [
        float(sample["temperature_c"])
        for phase in phases
        for sample in phase
        if sample.get("temperature_c") is not None
    ]
    return RepeatMetrics(
        repeat_index=repeat_index,
        outcome="completed",
        zero_position_deg=zero,
        loaded_position_deg=loaded,
        absolute_error_deg=center - loaded,
        incremental_travel_deg=travel,
        incremental_error_deg=center - travel,
        minimum_voltage_v=min(voltage) if voltage else None,
        initial_temperature_c=temperature[0] if temperature else None,
        final_temperature_c=temperature[-1] if temperature else None,
        temperature_rise_c=(
            temperature[-1] - temperature[0] if temperature else None
        ),
    )


def summarize_series(
    captures: Sequence[Path],
    *,
    expected_repeats: int | None = None,
    center_deg: float,
    max_repeatability_std_deg: float,
    max_position_error_deg: float,
    min_voltage_v: float,
    max_temperature_c: float,
) -> dict[str, Any]:
    repeats = [
        summarize_capture(path, index)
        for index, path in enumerate(captures, start=1)
    ]
    completed = [item for item in repeats if item.outcome == "completed"]
    failures: list[str] = []
    if expected_repeats is not None and len(repeats) != expected_repeats:
        failures.append(
            f"found {len(repeats)} captures; expected {expected_repeats}"
        )
    if len(completed) != len(repeats):
        failures.append("one or more captures did not complete")

    def values(name: str) -> np.ndarray:
        return np.asarray(
            [getattr(item, name) for item in completed], dtype=np.float64
        )

    loaded = values("loaded_position_deg")
    zero = values("zero_position_deg")
    absolute_error = np.abs(values("absolute_error_deg"))
    voltage = values("minimum_voltage_v")
    final_temperature = values("final_temperature_c")
    loaded_std = float(np.std(loaded)) if loaded.size else None
    zero_std = float(np.std(zero)) if zero.size else None
    if (
        loaded_std is None
        or not math.isfinite(loaded_std)
        or loaded_std > max_repeatability_std_deg
    ):
        failures.append("loaded-position repeatability exceeds limit")
    if (
        zero_std is None
        or not math.isfinite(zero_std)
        or zero_std > max_repeatability_std_deg
    ):
        failures.append("zero-position repeatability exceeds limit")
    if (
        not absolute_error.size
        or not np.all(np.isfinite(absolute_error))
        or float(np.max(absolute_error)) > max_position_error_deg
    ):
        failures.append("position error exceeds limit")
    if (
        not voltage.size
        or not np.all(np.isfinite(voltage))
        or float(np.min(voltage)) < min_voltage_v
    ):
        failures.append("voltage is below limit")
    if (
        not final_temperature.size
        or not np.all(np.isfinite(final_temperature))
        or float(np.max(final_temperature)) >= max_temperature_c
    ):
        failures.append("temperature reaches or exceeds limit")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stable" if not failures else "unstable_or_incomplete",
        "failures": failures,
        "center_deg": center_deg,
        "limits": {
            "max_repeatability_std_deg": max_repeatability_std_deg,
            "max_position_error_deg": max_position_error_deg,
            "min_voltage_v": min_voltage_v,
            "max_temperature_c": max_temperature_c,
        },
        "aggregate": {
            "completed_repeats": len(completed),
            "loaded_position_mean_deg": (
                float(np.mean(loaded)) if loaded.size else None
            ),
            "loaded_position_std_deg": loaded_std,
            "loaded_position_range_deg": (
                float(np.ptp(loaded)) if loaded.size else None
            ),
            "zero_position_mean_deg": float(np.mean(zero)) if zero.size else None,
            "zero_position_std_deg": zero_std,
            "maximum_absolute_error_deg": (
                float(np.max(absolute_error)) if absolute_error.size else None
            ),
            "minimum_voltage_v": float(np.min(voltage)) if voltage.size else None,
            "maximum_final_temperature_c": (
                float(np.max(final_temperature))
                if final_temperature.size
                else None
            ),
        },
        "repeats": [asdict(item) for item in repeats],
    }


def _angle_label(value: float) -> str:
    sign = "plus" if value >= 0.0 else "minus"
    magnitude = f"{abs(value):g}".replace(".", "p")
    return f"{sign}{magnitude}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servo-id", type=int, required=True)
    parser.add_argument("--servo-label", required=True)
    parser.add_argument("--board-port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--center-deg", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fixture-mjcf", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--fixture-qpos-offset-deg", type=float, default=0.0)
    parser.add_argument("--fixture-label", default="wr2-htd45h-2650g")
    parser.add_argument("--measured-weight-kg", type=float)
    parser.add_argument("--measured-com-radius-m", type=float)
    parser.add_argument("--prepare-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--cooldown-target-c", type=float, default=35.0)
    parser.add_argument("--min-voltage-v", type=float, default=9.6)
    parser.add_argument("--max-temperature-c", type=float, default=55.0)
    parser.add_argument("--max-position-error-deg", type=float, default=5.0)
    parser.add_argument("--max-repeatability-std-deg", type=float, default=0.5)
    parser.add_argument("--max-static-torque-nm", type=float, default=0.7)
    parser.add_argument("--max-predicted-torque-nm", type=float, default=0.7)
    parser.add_argument("--external-log-label")
    parser.add_argument("--output-dir", type=Path, default=Path("results/servo_sysid"))
    parser.add_argument("--series-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-fixture-safe", action="store_true")
    return parser


def _capture_command(
    args: argparse.Namespace, output: Path, condition_id: str, *, execute: bool
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "wr2.tools.servo_sysid.capture",
        "--servo-id",
        str(args.servo_id),
        "--servo-label",
        args.servo_label,
        "--board-port",
        args.board_port,
        "--baudrate",
        str(args.baudrate),
        "--condition-id",
        condition_id,
        "--center-deg",
        str(args.center_deg),
        "--prepare-only",
        "--prepare-speed-deg-s",
        str(args.prepare_speed_deg_s),
        "--settle-s",
        str(args.settle_s),
        "--cooldown-target-c",
        str(args.cooldown_target_c),
        "--min-voltage-v",
        str(args.min_voltage_v),
        "--max-temperature-c",
        str(args.max_temperature_c),
        "--max-position-error-deg",
        str(args.max_position_error_deg),
        "--max-static-torque-nm",
        str(args.max_static_torque_nm),
        "--max-predicted-torque-nm",
        str(args.max_predicted_torque_nm),
        "--fixture-mjcf",
        str(args.fixture_mjcf),
        "--fixture-direction",
        str(args.fixture_direction),
        "--fixture-qpos-offset-deg",
        str(args.fixture_qpos_offset_deg),
        "--fixture-label",
        args.fixture_label,
        "--output",
        str(output),
    ]
    if args.external_log_label:
        command.extend(("--external-log-label", args.external_log_label))
    if args.measured_weight_kg is not None:
        command.extend(("--measured-weight-kg", str(args.measured_weight_kg)))
    if args.measured_com_radius_m is not None:
        command.extend(
            ("--measured-com-radius-m", str(args.measured_com_radius_m))
        )
    if execute:
        command.extend(("--execute", "--confirm-fixture-safe"))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.repeats <= 20:
        raise SystemExit("--repeats must be between 1 and 20")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", args.servo_label):
        raise SystemExit("--servo-label must be filesystem-safe")
    if (
        not math.isfinite(args.max_repeatability_std_deg)
        or args.max_repeatability_std_deg <= 0.0
    ):
        raise SystemExit("--max-repeatability-std-deg must be finite and positive")
    if args.execute and not args.confirm_fixture_safe:
        raise SystemExit("--execute requires --confirm-fixture-safe")
    if args.execute and (
        args.measured_weight_kg is None or args.measured_com_radius_m is None
    ):
        raise SystemExit(
            "--execute requires --measured-weight-kg and --measured-com-radius-m"
        )

    label = _angle_label(args.center_deg)
    condition_base = f"C0_commission_{label}deg"
    preflight_output = Path("/tmp") / f"wr2_{condition_base}_preflight.npz"
    print(
        f"Commissioning preflight: center={args.center_deg:+g} deg, "
        f"repeats={args.repeats}",
        flush=True,
    )
    result = subprocess.run(
        _capture_command(
            args, preflight_output, f"{condition_base}_preflight", execute=False
        ),
        check=False,
    )
    if result.returncode or not args.execute:
        return int(result.returncode)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    series_dir = (
        args.series_dir.expanduser().resolve()
        if args.series_dir is not None
        else args.output_dir.expanduser().resolve()
        / f"{condition_base}_{args.servo_label}_{timestamp}"
    )
    series_dir.mkdir(parents=True, exist_ok=False)
    fixture = args.fixture_mjcf.expanduser().resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "servo_id": args.servo_id,
        "servo_label": args.servo_label,
        "center_deg": args.center_deg,
        "requested_repeats": args.repeats,
        "fixture_mjcf": str(fixture),
        "fixture_sha256": file_sha256(fixture),
        "fixture_label": args.fixture_label,
        "fixture_direction": args.fixture_direction,
        "fixture_qpos_offset_deg": args.fixture_qpos_offset_deg,
        "measured_weight_kg": args.measured_weight_kg,
        "measured_com_radius_m": args.measured_com_radius_m,
        "limits": {
            "min_voltage_v": args.min_voltage_v,
            "max_temperature_c": args.max_temperature_c,
            "max_position_error_deg": args.max_position_error_deg,
            "max_repeatability_std_deg": args.max_repeatability_std_deg,
            "max_static_torque_nm": args.max_static_torque_nm,
            "max_predicted_torque_nm": args.max_predicted_torque_nm,
        },
        "captures": [],
    }
    manifest_path = series_dir / "commission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    captures: list[Path] = []
    for repeat_index in range(1, args.repeats + 1):
        condition_id = f"{condition_base}_repeat{repeat_index:02d}"
        output = series_dir / f"repeat_{repeat_index:02d}.npz"
        print(f"\n[{repeat_index}/{args.repeats}] {condition_id}", flush=True)
        result = subprocess.run(
            _capture_command(args, output, condition_id, execute=True), check=False
        )
        metadata_path = output.with_suffix(".json")
        record: dict[str, Any] = {
            "repeat_index": repeat_index,
            "condition_id": condition_id,
            "returncode": int(result.returncode),
            "npz_path": str(output),
            "json_path": str(metadata_path),
        }
        if metadata_path.is_file():
            capture = json.loads(metadata_path.read_text())
            record["outcome"] = capture.get("outcome")
            record["error"] = capture.get("error")
            captures.append(metadata_path)
        manifest["captures"].append(record)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if result.returncode:
            manifest["status"] = "failed"
            manifest["failed_repeat"] = repeat_index
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            return int(result.returncode)

    summary = summarize_series(
        captures,
        center_deg=args.center_deg,
        expected_repeats=args.repeats,
        max_repeatability_std_deg=args.max_repeatability_std_deg,
        max_position_error_deg=args.max_position_error_deg,
        min_voltage_v=args.min_voltage_v,
        max_temperature_c=args.max_temperature_c,
    )
    summary_path = series_dir / "commission_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest["status"] = "completed"
    manifest["stability_status"] = summary["status"]
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nRepeatability status={summary['status']}")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"Wrote {series_dir}")
    return 0 if summary["status"] == "stable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
