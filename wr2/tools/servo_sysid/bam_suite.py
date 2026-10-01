"""Run the bounded low-load HTD-45H BAM characterization suite.

The suite is preflight-only unless both ``--execute`` and
``--confirm-fixture-safe`` are supplied.  It deliberately contains only the
currently supported +10 degree commissioning, -10 degree commissioning, and
gravity-neutral E3 conditions.  It never selects the high-load deployment
campaign conditions.
"""

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

from wr2.tools.servo_sysid.core import DEFAULT_FIXTURE, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMISSION_MAX_TORQUE_NM = 0.7
E3_MAX_STATIC_TORQUE_NM = 0.2
E3_MAX_PREDICTED_TORQUE_NM = 0.6


@dataclass(frozen=True)
class BamStage:
    name: str
    description: str
    kind: str
    center_deg: float | None = None


STAGES = (
    BamStage(
        "commission_plus10",
        "five low-load repeatability captures at +10 degrees",
        "commission",
        10.0,
    ),
    BamStage(
        "commission_minus10",
        "five low-load repeatability captures at -10 degrees",
        "commission",
        -10.0,
    ),
    BamStage(
        "e3_inertial_bandwidth",
        "gravity-neutral 2-degree, 0.1-4 Hz response",
        "e3",
    ),
)

BAM_DATA_GOALS = (
    "signed zero and loaded position repeatability",
    "direction-dependent low-load position error",
    "gravity-neutral position-loop response and whole-response delay",
    "command and measured position with derived velocity",
    "servo voltage and temperature during each bounded condition",
    "clock-aligned external supply voltage and current",
    "fixture-geometry-inferred torque, explicitly not measured shaft torque",
)


class BamSuiteError(RuntimeError):
    """An actionable BAM suite orchestration failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servo-id", type=int, required=True)
    parser.add_argument("--servo-label", required=True)
    parser.add_argument("--board-port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fixture-mjcf", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--fixture-qpos-offset-deg", type=float, default=0.0)
    parser.add_argument("--fixture-label", default="wr2-htd45h-2650g")
    parser.add_argument("--measured-weight-kg", type=float)
    parser.add_argument("--measured-com-radius-m", type=float)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--start-at", choices=[stage.name for stage in STAGES]
    )
    parser.add_argument(
        "--stop-after", choices=[stage.name for stage in STAGES]
    )
    parser.add_argument(
        "--run-all-safe",
        action="store_true",
        help="Run all three low-load stages without a manual review pause.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/servo_sysid"))
    parser.add_argument(
        "--suite-dir",
        type=Path,
        help="Stable suite directory; completed stages are skipped on resume.",
    )
    parser.add_argument("--external-log-label-prefix")
    parser.add_argument("--cooldown-target-c", type=float, default=35.0)
    parser.add_argument("--min-voltage-v", type=float, default=9.6)
    parser.add_argument("--max-temperature-c", type=float, default=55.0)
    parser.add_argument("--max-position-error-deg", type=float, default=5.0)
    parser.add_argument("--max-position-error-duration-s", type=float, default=0.15)
    parser.add_argument("--max-repeatability-std-deg", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-fixture-safe", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.servo_id <= 253:
        raise SystemExit("--servo-id must be between 0 and 253")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", args.servo_label):
        raise SystemExit("--servo-label must be filesystem-safe")
    if not 1 <= args.repeats <= 20:
        raise SystemExit("--repeats must be between 1 and 20")
    positive = {
        "baudrate": args.baudrate,
        "cooldown_target_c": args.cooldown_target_c,
        "min_voltage_v": args.min_voltage_v,
        "max_temperature_c": args.max_temperature_c,
        "max_position_error_deg": args.max_position_error_deg,
        "max_position_error_duration_s": args.max_position_error_duration_s,
        "max_repeatability_std_deg": args.max_repeatability_std_deg,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    if args.cooldown_target_c > args.max_temperature_c:
        raise SystemExit("--cooldown-target-c must not exceed --max-temperature-c")
    if args.run_all_safe and (args.start_at is not None or args.stop_after is not None):
        raise SystemExit("--run-all-safe cannot be combined with --start-at/--stop-after")
    if args.execute and not args.confirm_fixture_safe:
        raise SystemExit("--execute requires --confirm-fixture-safe")
    if args.execute and not args.run_all_safe and args.stop_after is None:
        raise SystemExit(
            "hardware mode requires --stop-after for a bounded stage range, "
            "or explicit --run-all-safe"
        )
    if args.execute and (
        args.measured_weight_kg is None or args.measured_com_radius_m is None
    ):
        raise SystemExit(
            "hardware mode requires --measured-weight-kg and "
            "--measured-com-radius-m"
        )
    if args.execute and not args.external_log_label_prefix:
        raise SystemExit(
            "hardware mode requires --external-log-label-prefix for the "
            "clock-aligned supply log"
        )
    if args.measured_weight_kg is not None and not math.isclose(
        args.measured_weight_kg, 2.650, rel_tol=0.01, abs_tol=0.0
    ):
        raise SystemExit("measured weight differs by >1%; update the fixture model")
    if args.measured_com_radius_m is not None and not math.isclose(
        args.measured_com_radius_m, 0.1204, rel_tol=0.01, abs_tol=0.0
    ):
        raise SystemExit("measured COM radius differs by >1%; update the fixture model")


def selected_stages(args: argparse.Namespace) -> tuple[BamStage, ...]:
    if args.run_all_safe:
        return STAGES
    first = (
        0
        if args.start_at is None
        else next(index for index, stage in enumerate(STAGES) if stage.name == args.start_at)
    )
    last = (
        len(STAGES) - 1
        if args.stop_after is None
        else next(index for index, stage in enumerate(STAGES) if stage.name == args.stop_after)
    )
    if last < first:
        raise SystemExit("--stop-after precedes --start-at")
    return STAGES[first : last + 1]


def _append_common_fixture_args(
    command: list[str], args: argparse.Namespace, *, execute: bool
) -> None:
    command.extend(
        (
            "--fixture-mjcf",
            str(args.fixture_mjcf),
            "--fixture-direction",
            str(args.fixture_direction),
            "--fixture-qpos-offset-deg",
            str(args.fixture_qpos_offset_deg),
            "--fixture-label",
            args.fixture_label,
            "--cooldown-target-c",
            str(args.cooldown_target_c),
            "--min-voltage-v",
            str(args.min_voltage_v),
            "--max-temperature-c",
            str(args.max_temperature_c),
            "--max-position-error-deg",
            str(args.max_position_error_deg),
        )
    )
    if args.measured_weight_kg is not None:
        command.extend(("--measured-weight-kg", str(args.measured_weight_kg)))
    if args.measured_com_radius_m is not None:
        command.extend(("--measured-com-radius-m", str(args.measured_com_radius_m)))
    if execute:
        command.extend(("--execute", "--confirm-fixture-safe"))


def stage_directory(suite_dir: Path, stage: BamStage) -> Path:
    index = STAGES.index(stage) + 1
    return suite_dir / f"{index:02d}_{stage.name}"


def stage_command(
    args: argparse.Namespace,
    stage: BamStage,
    suite_dir: Path,
    *,
    execute: bool,
) -> list[str]:
    external_label = (
        f"{args.external_log_label_prefix}-{stage.name}"
        if args.external_log_label_prefix
        else None
    )
    output = stage_directory(suite_dir, stage)
    if stage.kind == "commission":
        assert stage.center_deg is not None
        command = [
            sys.executable,
            "-m",
            "wr2.tools.servo_sysid.commission",
            "--servo-id",
            str(args.servo_id),
            "--servo-label",
            args.servo_label,
            "--board-port",
            args.board_port,
            "--baudrate",
            str(args.baudrate),
            "--center-deg",
            str(stage.center_deg),
            "--repeats",
            str(args.repeats),
            "--series-dir",
            str(output),
            "--max-repeatability-std-deg",
            str(args.max_repeatability_std_deg),
            "--max-static-torque-nm",
            str(COMMISSION_MAX_TORQUE_NM),
            "--max-predicted-torque-nm",
            str(COMMISSION_MAX_TORQUE_NM),
        ]
    elif stage.kind == "e3":
        command = [
            sys.executable,
            "-m",
            "wr2.tools.servo_sysid.campaign",
            "--servo-id",
            str(args.servo_id),
            "--servo-label",
            args.servo_label,
            "--board-port",
            args.board_port,
            "--baudrate",
            str(args.baudrate),
            "--campaign-dir",
            str(output),
            "--start-at",
            "E3_inertial_bandwidth",
            "--stop-after",
            "E3_inertial_bandwidth",
            "--max-position-error-duration-s",
            str(args.max_position_error_duration_s),
            "--max-static-torque-nm",
            str(E3_MAX_STATIC_TORQUE_NM),
            "--max-predicted-torque-nm",
            str(E3_MAX_PREDICTED_TORQUE_NM),
        ]
    else:
        raise AssertionError(f"unknown BAM stage kind: {stage.kind}")
    if external_label:
        command.extend(("--external-log-label", external_label))
    _append_common_fixture_args(command, args, execute=execute)
    return command


def _run(command: Sequence[str]) -> int:
    return int(subprocess.run(list(command), check=False).returncode)


def _git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "worktree_clean": not bool(status.strip())}


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    fixture = args.fixture_mjcf.expanduser().resolve()
    return {
        "servo_id": args.servo_id,
        "servo_label": args.servo_label,
        "baudrate": args.baudrate,
        "fixture_mjcf": str(fixture),
        "fixture_sha256": file_sha256(fixture),
        "fixture_direction": args.fixture_direction,
        "fixture_qpos_offset_deg": args.fixture_qpos_offset_deg,
        "fixture_label": args.fixture_label,
        "measured_weight_kg": args.measured_weight_kg,
        "measured_com_radius_m": args.measured_com_radius_m,
        "repeats": args.repeats,
        "external_log_label_prefix": args.external_log_label_prefix,
        "limits": {
            "cooldown_target_c": args.cooldown_target_c,
            "min_voltage_v": args.min_voltage_v,
            "max_temperature_c": args.max_temperature_c,
            "max_position_error_deg": args.max_position_error_deg,
            "max_position_error_duration_s": args.max_position_error_duration_s,
            "max_repeatability_std_deg": args.max_repeatability_std_deg,
            "commission_max_torque_nm": COMMISSION_MAX_TORQUE_NM,
            "e3_max_static_torque_nm": E3_MAX_STATIC_TORQUE_NM,
            "e3_max_predicted_torque_nm": E3_MAX_PREDICTED_TORQUE_NM,
        },
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_stage_result(stage: BamStage, directory: Path) -> dict[str, Any]:
    if stage.kind == "commission":
        path = directory / "commission_summary.json"
        if not path.is_file():
            raise BamSuiteError(f"missing commission summary: {path}")
        summary = json.loads(path.read_text())
        if summary.get("status") != "stable":
            raise BamSuiteError(f"commission stage is not stable: {path}")
        return {
            "summary_path": str(path),
            "status": summary.get("status"),
            "aggregate": summary.get("aggregate"),
        }
    path = directory / "03_E3_inertial_bandwidth.json"
    if not path.is_file():
        raise BamSuiteError(f"missing E3 capture metadata: {path}")
    capture = json.loads(path.read_text())
    if capture.get("outcome") != "completed":
        raise BamSuiteError(f"E3 capture did not complete: {path}")
    return {
        "metadata_path": str(path),
        "outcome": capture.get("outcome"),
        "trace_summary": capture.get("trace_summary"),
        "health_summary": capture.get("health_summary"),
        "timing_summary": capture.get("timing_summary"),
        "predicted_envelope": capture.get("predicted_envelope"),
    }


def _suite_directory(args: argparse.Namespace) -> Path:
    if args.suite_dir is not None:
        return args.suite_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        args.output_dir.expanduser().resolve()
        / f"bam_low_load_{args.servo_label}_{timestamp}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    stages = selected_stages(args)
    preflight_root = Path("/tmp/wr2_bam_suite_preflight")
    print("WR2 HTD-45H bounded BAM suite", flush=True)
    print(f"mode={'HARDWARE' if args.execute else 'PREFLIGHT'}", flush=True)
    print("goals:", flush=True)
    for goal in BAM_DATA_GOALS:
        print(f"  - {goal}", flush=True)
    for index, stage in enumerate(stages, start=1):
        print(
            f"\nPREFLIGHT [{index}/{len(stages)}] {stage.name}: "
            f"{stage.description}",
            flush=True,
        )
        returncode = _run(stage_command(args, stage, preflight_root, execute=False))
        if returncode:
            return returncode
    if not args.execute:
        print("\nBAM suite preflight passed; no serial port was opened.", flush=True)
        return 0

    git_state = _git_state()
    if not git_state["worktree_clean"]:
        raise SystemExit("hardware capture requires a clean Git worktree")
    suite_dir = _suite_directory(args)
    manifest_path = suite_dir / "bam_suite_manifest.json"
    configuration = _configuration(args)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("configuration") != configuration:
            raise SystemExit("existing BAM suite uses a different configuration")
        if manifest.get("git", {}).get("revision") != git_state["revision"]:
            raise SystemExit("existing BAM suite uses a different Git revision")
    else:
        if suite_dir.exists():
            raise SystemExit(f"suite directory exists without a manifest: {suite_dir}")
        suite_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "git": git_state,
            "configuration": configuration,
            "data_goals": list(BAM_DATA_GOALS),
            "limitations": [
                "fixture torque is inferred from geometry, not measured shaft torque",
                "suite does not identify the full torque-speed or braking envelope",
                "external supply logger must be started and synchronized separately",
            ],
            "stages": [],
        }
        _write_manifest(manifest_path, manifest)

    completed = {
        record.get("name")
        for record in manifest.get("stages", [])
        if record.get("status") == "completed"
    }
    print(
        "\nHardware execution is limited to +10, -10, and E3. "
        "The external logger must already be recording.",
        flush=True,
    )
    for stage in stages:
        if stage.name in completed:
            print(f"SKIP completed stage {stage.name}", flush=True)
            continue
        directory = stage_directory(suite_dir, stage)
        if directory.exists():
            raise SystemExit(
                f"incomplete stage directory already exists: {directory}; "
                "inspect it before choosing a new suite directory"
            )
        print(f"\nRUN {stage.name}: {stage.description}", flush=True)
        returncode = _run(stage_command(args, stage, suite_dir, execute=True))
        record: dict[str, Any] = {
            **asdict(stage),
            "directory": str(directory),
            "returncode": returncode,
            "external_log_label": (
                f"{args.external_log_label_prefix}-{stage.name}"
            ),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if returncode:
            record["status"] = "failed"
            manifest["stages"].append(record)
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage.name
            _write_manifest(manifest_path, manifest)
            return returncode
        try:
            record["result"] = _load_stage_result(stage, directory)
        except BamSuiteError as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            manifest["stages"].append(record)
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage.name
            _write_manifest(manifest_path, manifest)
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        record["status"] = "completed"
        manifest["stages"].append(record)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)

    completed = {
        record.get("name")
        for record in manifest["stages"]
        if record.get("status") == "completed"
    }
    manifest["status"] = "completed" if completed == {s.name for s in STAGES} else "partial"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    print(f"\nBAM suite status={manifest['status']}: {suite_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
