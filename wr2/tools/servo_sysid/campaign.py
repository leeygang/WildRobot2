"""Run the staged WR2 HTD-45H deployment-envelope campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from wr2.tools.servo_sysid.core import DEFAULT_FIXTURE, file_sha256


@dataclass(frozen=True)
class CampaignCondition:
    condition_id: str
    description: str
    center_deg: float
    amplitudes_deg: str = "2,5,8"
    chirp_end_hz: float = 2.0
    prepare_only: bool = False
    prepare_speed_deg_s: float = 20.0
    settle_s: float = 1.0
    write_deadband_units: int = 0
    constant_hold_s: float | None = None
    required_static_torque_nm: float | None = None


CONDITIONS = (
    CampaignCondition(
        "E1_static_plus72",
        "slow positive ramp to the 3 N m deployment margin",
        72.0,
        amplitudes_deg="2",
        prepare_only=True,
        prepare_speed_deg_s=5.0,
        settle_s=10.0,
        required_static_torque_nm=3.0,
    ),
    CampaignCondition(
        "E2_static_minus72",
        "slow negative ramp to the 3 N m deployment margin",
        -72.0,
        amplitudes_deg="2",
        prepare_only=True,
        prepare_speed_deg_s=5.0,
        settle_s=10.0,
        required_static_torque_nm=3.0,
    ),
    CampaignCondition(
        "E3_inertial_bandwidth",
        "gravity-neutral 0.1-4 Hz inertial response",
        0.0,
        amplitudes_deg="2",
        chirp_end_hz=4.0,
    ),
    CampaignCondition(
        "E4_loaded_plus60",
        "positive 2.75 N m loaded response",
        60.0,
        amplitudes_deg="2,5",
    ),
    CampaignCondition(
        "E5_loaded_minus60",
        "negative 2.75 N m loaded response",
        -60.0,
        amplitudes_deg="2,5",
    ),
    CampaignCondition(
        "E6_deployment_deadband_plus60",
        "loaded response with three-unit deployment deadband",
        60.0,
        amplitudes_deg="2,5",
        write_deadband_units=3,
    ),
    CampaignCondition(
        "E7_thermal_hold_plus60",
        "bounded ten-minute hold at the deployment RMS target",
        60.0,
        amplitudes_deg="2",
        prepare_speed_deg_s=5.0,
        constant_hold_s=600.0,
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servo-id", type=int, required=True)
    parser.add_argument("--board-port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fixture-mjcf", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--fixture-qpos-offset-deg", type=float, default=0.0)
    parser.add_argument("--fixture-label", default="wr2-htd45h-2650g")
    parser.add_argument("--servo-label")
    parser.add_argument("--external-log-label")
    parser.add_argument("--measured-weight-kg", type=float)
    parser.add_argument("--measured-com-radius-m", type=float)
    parser.add_argument(
        "--start-at", choices=[item.condition_id for item in CONDITIONS]
    )
    parser.add_argument(
        "--stop-after", choices=[item.condition_id for item in CONDITIONS]
    )
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-fixture-safe", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/servo_sysid"))
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        help="Reuse this directory across staged condition invocations.",
    )
    parser.add_argument("--cooldown-target-c", type=float, default=35.0)
    parser.add_argument("--min-voltage-v", type=float, default=9.6)
    parser.add_argument("--max-temperature-c", type=float, default=55.0)
    parser.add_argument("--max-position-error-deg", type=float, default=5.0)
    parser.add_argument("--max-position-error-duration-s", type=float, default=0.15)
    parser.add_argument("--max-static-torque-nm", type=float, default=3.2)
    parser.add_argument("--max-predicted-torque-nm", type=float, default=3.2)
    return parser


def selected_conditions(args: argparse.Namespace) -> tuple[CampaignCondition, ...]:
    first = (
        0
        if args.start_at is None
        else next(
            index
            for index, item in enumerate(CONDITIONS)
            if item.condition_id == args.start_at
        )
    )
    last = (
        len(CONDITIONS) - 1
        if args.stop_after is None
        else next(
            index
            for index, item in enumerate(CONDITIONS)
            if item.condition_id == args.stop_after
        )
    )
    if last < first:
        raise ValueError("--stop-after precedes --start-at")
    return CONDITIONS[first : last + 1]


def capture_command(
    args: argparse.Namespace,
    condition: CampaignCondition,
    output: Path,
    *,
    execute: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "wr2.tools.servo_sysid.capture",
        "--condition-id",
        condition.condition_id,
        "--servo-id",
        str(args.servo_id),
        "--board-port",
        args.board_port,
        "--baudrate",
        str(args.baudrate),
        "--fixture-mjcf",
        str(args.fixture_mjcf),
        "--fixture-direction",
        str(args.fixture_direction),
        "--fixture-qpos-offset-deg",
        str(args.fixture_qpos_offset_deg),
        "--fixture-label",
        args.fixture_label,
        "--center-deg",
        str(condition.center_deg),
        "--amplitudes-deg",
        condition.amplitudes_deg,
        "--chirp-end-hz",
        str(condition.chirp_end_hz),
        "--prepare-speed-deg-s",
        str(condition.prepare_speed_deg_s),
        "--settle-s",
        str(condition.settle_s),
        "--write-deadband-units",
        str(condition.write_deadband_units),
        "--cooldown-target-c",
        str(args.cooldown_target_c),
        "--min-voltage-v",
        str(args.min_voltage_v),
        "--max-temperature-c",
        str(args.max_temperature_c),
        "--max-position-error-deg",
        str(args.max_position_error_deg),
        "--max-position-error-duration-s",
        str(args.max_position_error_duration_s),
        "--max-static-torque-nm",
        str(args.max_static_torque_nm),
        "--max-predicted-torque-nm",
        str(args.max_predicted_torque_nm),
        "--output",
        str(output),
    ]
    if condition.prepare_only:
        command.append("--prepare-only")
    if condition.constant_hold_s is not None:
        command.extend(("--constant-hold-s", str(condition.constant_hold_s)))
    if condition.required_static_torque_nm is not None:
        command.extend(
            ("--required-static-torque-nm", str(condition.required_static_torque_nm))
        )
    if args.servo_label:
        command.extend(("--servo-label", args.servo_label))
    if args.external_log_label:
        command.extend(("--external-log-label", args.external_log_label))
    if args.measured_weight_kg is not None:
        command.extend(("--measured-weight-kg", str(args.measured_weight_kg)))
    if args.measured_com_radius_m is not None:
        command.extend(("--measured-com-radius-m", str(args.measured_com_radius_m)))
    if execute:
        command.extend(("--execute", "--confirm-fixture-safe"))
    return command


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _same_number(left: object, right: float) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = selected_conditions(args)
    if args.execute and not args.confirm_fixture_safe:
        raise SystemExit("--execute requires --confirm-fixture-safe")
    if args.execute and not args.run_all and args.stop_after is None:
        raise SystemExit(
            "Hardware mode requires --stop-after for staged review, or explicit --run-all"
        )
    if args.execute and (
        args.measured_weight_kg is None or args.measured_com_radius_m is None
    ):
        raise SystemExit(
            "hardware mode requires measured fixture weight and COM radius"
        )

    print("WR2 HTD-45H deployment-envelope campaign", flush=True)
    print(f"mode={'HARDWARE' if args.execute else 'PREFLIGHT'}", flush=True)
    for index, condition in enumerate(selected, start=1):
        output = Path("/tmp") / f"wr2_preflight_{condition.condition_id}.npz"
        print(
            f"\n[{index}/{len(selected)}] {condition.condition_id}: "
            f"{condition.description}",
            flush=True,
        )
        result = subprocess.run(
            capture_command(args, condition, output, execute=False), check=False
        )
        if result.returncode:
            return int(result.returncode)
    if not args.execute:
        print("\nCampaign preflight passed; no hardware was opened.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = (
        args.campaign_dir.expanduser().resolve()
        if args.campaign_dir is not None
        else args.output_dir.expanduser().resolve()
        / f"htd45h_{args.servo_id}_deployment_{timestamp}"
    )
    manifest_path = campaign_dir / "campaign_manifest.json"
    fixture_path = args.fixture_mjcf.expanduser().resolve()
    fixture_hash = file_sha256(fixture_path)
    safety_limits = {
        "min_voltage_v": args.min_voltage_v,
        "max_temperature_c": args.max_temperature_c,
        "max_position_error_deg": args.max_position_error_deg,
        "max_position_error_duration_s": args.max_position_error_duration_s,
        "max_static_torque_nm": args.max_static_torque_nm,
        "max_predicted_torque_nm": args.max_predicted_torque_nm,
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("servo_id", -1)) != args.servo_id:
            raise SystemExit("existing campaign directory uses a different servo ID")
        if manifest.get("servo_label") != args.servo_label:
            raise SystemExit("existing campaign directory uses a different servo label")
        if manifest.get("fixture_sha256") != fixture_hash:
            raise SystemExit("existing campaign directory uses a different fixture model")
        expected_numbers = {
            "fixture_direction": args.fixture_direction,
            "fixture_qpos_offset_deg": args.fixture_qpos_offset_deg,
            "measured_weight_kg": args.measured_weight_kg,
            "measured_com_radius_m": args.measured_com_radius_m,
        }
        for name, value in expected_numbers.items():
            assert value is not None
            if not _same_number(manifest.get(name), float(value)):
                raise SystemExit(
                    f"existing campaign directory has a different {name}"
                )
        if manifest.get("fixture_label") != args.fixture_label:
            raise SystemExit("existing campaign directory uses a different fixture label")
        if manifest.get("safety_limits") != safety_limits:
            raise SystemExit("existing campaign directory uses different safety limits")
    else:
        campaign_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "servo_id": args.servo_id,
            "servo_label": args.servo_label,
            "fixture_mjcf": str(fixture_path),
            "fixture_sha256": fixture_hash,
            "fixture_label": args.fixture_label,
            "fixture_direction": args.fixture_direction,
            "fixture_qpos_offset_deg": args.fixture_qpos_offset_deg,
            "measured_weight_kg": args.measured_weight_kg,
            "measured_com_radius_m": args.measured_com_radius_m,
            "safety_limits": safety_limits,
            "conditions": [],
        }
    for condition in selected:
        index = CONDITIONS.index(condition) + 1
        output = campaign_dir / f"{index:02d}_{condition.condition_id}.npz"
        if output.exists() or output.with_suffix(".json").exists():
            raise SystemExit(f"condition output already exists: {output}")
    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)
    for condition in selected:
        index = CONDITIONS.index(condition) + 1
        output = campaign_dir / f"{index:02d}_{condition.condition_id}.npz"
        result = subprocess.run(
            capture_command(args, condition, output, execute=True), check=False
        )
        record: dict[str, object] = {
            **asdict(condition),
            "returncode": int(result.returncode),
            "npz_path": str(output),
            "json_path": str(output.with_suffix(".json")),
        }
        if output.with_suffix(".json").is_file():
            capture = json.loads(output.with_suffix(".json").read_text())
            record.update(
                {
                    "outcome": capture.get("outcome"),
                    "error": capture.get("error"),
                    "tested_static_torque_nm": capture.get("tested_static_torque_nm"),
                    "predicted_envelope": capture.get("predicted_envelope"),
                    "trace_summary": capture.get("trace_summary"),
                    "health_summary": capture.get("health_summary"),
                }
            )
        conditions = manifest["conditions"]
        assert isinstance(conditions, list)
        conditions.append(record)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        if result.returncode:
            manifest["status"] = "failed"
            manifest["failed_condition"] = condition.condition_id
            _write_manifest(manifest_path, manifest)
            return int(result.returncode)
    completed = {
        item.get("condition_id")
        for item in manifest["conditions"]
        if item.get("outcome") == "completed"
    }
    if completed == {item.condition_id for item in CONDITIONS}:
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        manifest["status"] = "partial"
    _write_manifest(manifest_path, manifest)
    print(f"Campaign status={manifest['status']}: {campaign_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
