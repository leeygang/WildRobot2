"""Run one versioned HTD-45H characterization plan."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Sequence, TextIO

from wr2.tools.servo_sysid.analysis.position import summarize_series
from wr2.tools.servo_sysid.core import DEFAULT_FIXTURE, file_sha256
from wr2.tools.servo_sysid.plan import (
    CampaignCondition,
    CampaignPlan,
    available_plans,
    load_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_PLAN = load_plan("legacy_deployment")
# Compatibility for the deployment analyzer and existing callers.
CONDITIONS = DEPLOYMENT_PLAN.conditions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", choices=available_plans(), default="legacy_deployment"
    )
    parser.add_argument("--servo-id", type=int, required=True)
    parser.add_argument("--board-port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fixture-mjcf", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--fixture-qpos-offset-deg", type=float, default=0.0)
    parser.add_argument("--fixture-label", default="wr2-htd45h-2650g")
    parser.add_argument("--servo-label")
    parser.add_argument("--external-log-label")
    parser.add_argument("--external-log-label-prefix")
    parser.add_argument("--measured-weight-kg", type=float)
    parser.add_argument("--measured-com-radius-m", type=float)
    parser.add_argument("--start-at")
    parser.add_argument("--stop-after")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-fixture-safe", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/servo_sysid"))
    parser.add_argument(
        "--campaign-dir",
        "--run-dir",
        dest="campaign_dir",
        type=Path,
        help="Stable directory used to resume a versioned plan.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Archive an incomplete --run-dir and start a fresh campaign there.",
    )
    parser.add_argument("--center-deg", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--prepare-speed-deg-s", type=float)
    parser.add_argument("--settle-s", type=float)
    parser.add_argument(
        "--cooldown-target-c",
        type=float,
        default=35.0,
        help="Starting-temperature ceiling in C; used directly for every condition (default: 35).",
    )
    parser.add_argument(
        "--min-voltage-v",
        type=float,
        default=9.6,
        help="qualified minimum that emits a yellow warning without stopping",
    )
    parser.add_argument(
        "--hard-min-voltage-v",
        type=float,
        default=5.0,
        help="absolute voltage floor that still aborts capture",
    )
    parser.add_argument("--max-temperature-c", type=float, default=80.0)
    parser.add_argument("--max-position-error-deg", type=float, default=5.0)
    parser.add_argument("--max-position-error-duration-s", type=float, default=0.15)
    parser.add_argument("--max-repeatability-std-deg", type=float, default=0.5)
    parser.add_argument("--max-static-torque-nm", type=float, default=3.2)
    parser.add_argument("--max-predicted-torque-nm", type=float, default=3.2)
    return parser


def _angle_label(value: float) -> str:
    sign = "plus" if value >= 0.0 else "minus"
    magnitude = f"{abs(value):g}".replace(".", "p")
    return f"{sign}{magnitude}"


def configured_plan(args: argparse.Namespace) -> CampaignPlan:
    plan = load_plan(args.plan)
    conditions = list(plan.conditions)
    if args.center_deg is not None:
        if plan.name != "bam_repeatability":
            raise SystemExit("--center-deg is only valid with --plan bam_repeatability")
        condition = conditions[0]
        conditions[0] = replace(
            condition,
            condition_id=f"repeatability_{_angle_label(args.center_deg)}",
            description=f"repeated low-load capture at {args.center_deg:+g} degrees",
            center_deg=args.center_deg,
        )
    if args.repeats is not None:
        if not 1 <= args.repeats <= 20:
            raise SystemExit("--repeats must be between 1 and 20")
        conditions = [
            replace(item, repeats=args.repeats)
            if item.kind == "repeatability"
            else item
            for item in conditions
        ]
    if args.prepare_speed_deg_s is not None:
        if not math.isfinite(args.prepare_speed_deg_s) or args.prepare_speed_deg_s <= 0:
            raise SystemExit("--prepare-speed-deg-s must be finite and positive")
        conditions = [
            replace(item, prepare_speed_deg_s=args.prepare_speed_deg_s)
            if item.kind == "repeatability"
            else item
            for item in conditions
        ]
    if args.settle_s is not None:
        if not math.isfinite(args.settle_s) or args.settle_s < 0:
            raise SystemExit("--settle-s must be finite and non-negative")
        conditions = [
            replace(item, settle_s=args.settle_s)
            if item.kind == "repeatability"
            else item
            for item in conditions
        ]
    return replace(plan, conditions=tuple(conditions))


def selected_conditions(
    args: argparse.Namespace, plan: CampaignPlan | None = None
) -> tuple[CampaignCondition, ...]:
    active = plan or load_plan(getattr(args, "plan", "legacy_deployment"))
    conditions = active.conditions
    identifiers = [item.condition_id for item in conditions]
    try:
        first = 0 if args.start_at is None else identifiers.index(args.start_at)
        last = (
            len(conditions) - 1
            if args.stop_after is None
            else identifiers.index(args.stop_after)
        )
    except ValueError as exc:
        raise SystemExit(
            f"condition is not in plan {active.name}; choose from {', '.join(identifiers)}"
        ) from exc
    if last < first:
        raise SystemExit("--stop-after precedes --start-at")
    return conditions[first : last + 1]


def _bounded(value: float, plan_limit: float | None) -> float:
    return value if plan_limit is None else min(value, plan_limit)


def _external_label(args: argparse.Namespace, condition_id: str) -> str | None:
    if args.external_log_label_prefix:
        return f"{args.external_log_label_prefix}-{condition_id}"
    return args.external_log_label


def capture_command(
    args: argparse.Namespace,
    condition: CampaignCondition,
    output: Path,
    *,
    execute: bool,
    condition_id: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "wr2.tools.servo_sysid.capture",
        "--condition-id",
        condition_id or condition.condition_id,
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
        "--move-time-ms",
        str(condition.move_time_ms),
        "--profile",
        condition.profile,
        "--sweep-rate-deg-s",
        str(condition.sweep_rate_deg_s),
        "--sweep-cycles",
        str(condition.sweep_cycles),
        "--cooldown-target-c",
        str(args.cooldown_target_c),
        "--min-voltage-v",
        str(args.min_voltage_v),
        "--hard-min-voltage-v",
        str(args.hard_min_voltage_v),
        "--max-temperature-c",
        str(_bounded(args.max_temperature_c, condition.max_temperature_c)),
        "--max-position-error-deg",
        str(args.max_position_error_deg),
        "--max-position-error-duration-s",
        str(args.max_position_error_duration_s),
        "--max-static-torque-nm",
        str(_bounded(args.max_static_torque_nm, condition.max_static_torque_nm)),
        "--max-predicted-torque-nm",
        str(_bounded(args.max_predicted_torque_nm, condition.max_predicted_torque_nm)),
        "--output",
        str(output),
    ]
    if condition.prepare_only or condition.kind == "repeatability":
        command.append("--prepare-only")
    if condition.constant_hold_s is not None:
        command.extend(("--constant-hold-s", str(condition.constant_hold_s)))
    if condition.required_static_torque_nm is not None:
        command.extend(
            ("--required-static-torque-nm", str(condition.required_static_torque_nm))
        )
    if args.servo_label:
        command.extend(("--servo-label", args.servo_label))
    external_label = _external_label(args, condition.condition_id)
    if external_label:
        command.extend(("--external-log-label", external_label))
    if args.measured_weight_kg is not None:
        command.extend(("--measured-weight-kg", str(args.measured_weight_kg)))
    if args.measured_com_radius_m is not None:
        command.extend(("--measured-com-radius-m", str(args.measured_com_radius_m)))
    if execute:
        command.extend(("--execute", "--confirm-fixture-safe"))
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


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def _campaign_lock(campaign_dir: Path) -> Iterator[TextIO]:
    """Reject overlapping hardware runners targeting the same campaign."""
    campaign_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = campaign_dir.parent / f".{campaign_dir.name}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip()
            detail = f"; owner={owner}" if owner else ""
            raise SystemExit(
                f"campaign is already running for {campaign_dir}{detail}"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _same_number(left: object, right: float) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _safety_limits(args: argparse.Namespace) -> dict[str, float]:
    return {
        "cooldown_target_c": args.cooldown_target_c,
        "min_voltage_v": args.min_voltage_v,
        "hard_min_voltage_v": args.hard_min_voltage_v,
        "max_temperature_c": args.max_temperature_c,
        "max_position_error_deg": args.max_position_error_deg,
        "max_position_error_duration_s": args.max_position_error_duration_s,
        "max_repeatability_std_deg": args.max_repeatability_std_deg,
        "max_static_torque_nm": args.max_static_torque_nm,
        "max_predicted_torque_nm": args.max_predicted_torque_nm,
    }


def _campaign_directory(args: argparse.Namespace, plan: CampaignPlan) -> Path:
    if args.campaign_dir is not None:
        return args.campaign_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.servo_label or f"servo-{args.servo_id}"
    return args.output_dir.expanduser().resolve() / f"{plan.name}_{label}_{timestamp}"


def _new_manifest(
    args: argparse.Namespace,
    plan: CampaignPlan,
    fixture_path: Path,
    git_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "plan": {
            "name": plan.name,
            "path": str(plan.path.resolve()),
            "sha256": file_sha256(plan.path),
            "description": plan.description,
            "setup": plan.setup,
            "required_instruments": list(plan.required_instruments),
            "data_goals": list(plan.data_goals),
            "limitations": list(plan.limitations),
        },
        "git": git_state,
        "servo_id": args.servo_id,
        "servo_label": args.servo_label,
        "fixture_mjcf": str(fixture_path),
        "fixture_sha256": file_sha256(fixture_path),
        "fixture_label": args.fixture_label,
        "fixture_direction": args.fixture_direction,
        "fixture_qpos_offset_deg": args.fixture_qpos_offset_deg,
        "measured_weight_kg": args.measured_weight_kg,
        "measured_com_radius_m": args.measured_com_radius_m,
        "external_log_label": args.external_log_label,
        "external_log_label_prefix": args.external_log_label_prefix,
        "safety_limits": _safety_limits(args),
        "conditions": [],
    }


def _validate_existing_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    plan: CampaignPlan,
    fixture_path: Path,
    git_state: dict[str, Any],
) -> None:
    if manifest.get("plan", {}).get("name") != plan.name:
        raise SystemExit("existing campaign directory uses a different plan")
    if manifest.get("plan", {}).get("sha256") != file_sha256(plan.path):
        raise SystemExit("existing campaign directory uses a different plan revision")
    if manifest.get("git", {}).get("revision") != git_state["revision"]:
        raise SystemExit("existing campaign directory uses a different Git revision")
    if int(manifest.get("servo_id", -1)) != args.servo_id:
        raise SystemExit("existing campaign directory uses a different servo ID")
    if manifest.get("servo_label") != args.servo_label:
        raise SystemExit("existing campaign directory uses a different servo label")
    if manifest.get("fixture_sha256") != file_sha256(fixture_path):
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
            raise SystemExit(f"existing campaign directory has a different {name}")
    if manifest.get("fixture_label") != args.fixture_label:
        raise SystemExit("existing campaign directory uses a different fixture label")
    if manifest.get("safety_limits") != _safety_limits(args):
        raise SystemExit("existing campaign directory uses different safety limits")


def _capture_record(
    output: Path, condition: CampaignCondition, returncode: int
) -> dict[str, Any]:
    metadata_path = output.with_suffix(".json")
    record: dict[str, Any] = {
        **asdict(condition),
        "returncode": returncode,
        "npz_path": str(output),
        "json_path": str(metadata_path),
    }
    if metadata_path.is_file():
        capture = json.loads(metadata_path.read_text())
        record.update(
            {
                "outcome": capture.get("outcome"),
                "error": capture.get("error"),
                "tested_static_torque_nm": capture.get("tested_static_torque_nm"),
                "predicted_envelope": capture.get("predicted_envelope"),
                "trace_summary": capture.get("trace_summary"),
                "hysteresis_summary": capture.get("hysteresis_summary"),
                "health_summary": capture.get("health_summary"),
                "timing_summary": capture.get("timing_summary"),
                "voltage_warnings": capture.get("voltage_warnings"),
            }
        )
    return record


def _run_capture_condition(
    args: argparse.Namespace,
    condition: CampaignCondition,
    output: Path,
) -> tuple[int, dict[str, Any]]:
    returncode = _run(capture_command(args, condition, output, execute=True))
    return returncode, _capture_record(output, condition, returncode)


def _run_repeatability_condition(
    args: argparse.Namespace,
    condition: CampaignCondition,
    directory: Path,
) -> tuple[int, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=False)
    captures: list[Path] = []
    records: list[dict[str, Any]] = []
    for repeat_index in range(1, condition.repeats + 1):
        repeat_id = f"{condition.condition_id}_repeat{repeat_index:02d}"
        output = directory / f"repeat_{repeat_index:02d}.npz"
        print(f"  [{repeat_index}/{condition.repeats}] {repeat_id}", flush=True)
        returncode = _run(
            capture_command(
                args,
                condition,
                output,
                execute=True,
                condition_id=repeat_id,
            )
        )
        record = _capture_record(output, condition, returncode)
        record["repeat_index"] = repeat_index
        record["capture_condition_id"] = repeat_id
        records.append(record)
        metadata_path = output.with_suffix(".json")
        if metadata_path.is_file():
            captures.append(metadata_path)
        if returncode:
            return returncode, {
                **asdict(condition),
                "outcome": "failed",
                "failed_repeat": repeat_index,
                "captures": records,
            }
    effective_temperature_limit = _bounded(
        args.max_temperature_c, condition.max_temperature_c
    )
    summary = summarize_series(
        captures,
        center_deg=condition.center_deg,
        expected_repeats=condition.repeats,
        max_repeatability_std_deg=args.max_repeatability_std_deg,
        max_position_error_deg=args.max_position_error_deg,
        min_voltage_v=args.min_voltage_v,
        max_temperature_c=effective_temperature_limit,
    )
    summary_path = directory / "repeatability_summary.json"
    voltage_failure = "voltage is below limit"
    blocking_failures = [
        failure for failure in summary["failures"] if failure != voltage_failure
    ]
    capture_voltage_warnings = any(
        int(record.get("voltage_warnings", {}).get("event_count", 0)) > 0
        for record in records
        if isinstance(record.get("voltage_warnings"), dict)
    )
    has_voltage_warning = (
        voltage_failure in summary["failures"] or capture_voltage_warnings
    )
    completed = not blocking_failures
    if not completed:
        summary["execution_status"] = "failed"
    elif has_voltage_warning:
        summary["execution_status"] = "completed_with_warnings"
    else:
        summary["execution_status"] = "completed"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    outcome = summary["execution_status"]
    return (0 if completed else 2), {
        **asdict(condition),
        "outcome": outcome,
        "summary_path": str(summary_path),
        "summary": summary,
        "captures": records,
    }


def _condition_output(
    campaign_dir: Path, plan: CampaignPlan, condition: CampaignCondition
) -> Path:
    index = plan.conditions.index(condition) + 1
    stem = campaign_dir / f"{index:02d}_{condition.condition_id}"
    return stem if condition.kind == "repeatability" else stem.with_suffix(".npz")


def _validate_args(args: argparse.Namespace, plan: CampaignPlan) -> None:
    if not 0 <= args.servo_id <= 253:
        raise SystemExit("--servo-id must be between 0 and 253")
    if args.restart:
        if args.campaign_dir is None:
            raise SystemExit("--restart requires --run-dir")
        if args.campaign_dir.expanduser().is_symlink():
            raise SystemExit("--restart refuses a symlink campaign directory")
        directory = args.campaign_dir.expanduser().resolve()
        if directory == REPO_ROOT or directory in REPO_ROOT.parents:
            raise SystemExit("--restart refuses the repository or its parent directories")
    unsupported = sorted(set(plan.required_instruments) - {"servo_bus"})
    if unsupported:
        raise SystemExit(
            f"plan {plan.name} requires unavailable instruments: {unsupported}"
        )
    if args.execute and not args.confirm_fixture_safe:
        raise SystemExit("--execute requires --confirm-fixture-safe")
    if args.execute and not args.run_all and args.stop_after is None:
        raise SystemExit(
            "hardware mode requires --stop-after for staged review, or explicit --run-all"
        )
    if args.execute and not args.servo_label:
        raise SystemExit("hardware mode requires --servo-label")
    if args.execute and (
        args.measured_weight_kg is None or args.measured_com_radius_m is None
    ):
        raise SystemExit(
            "hardware mode requires measured fixture weight and COM radius"
        )
    if args.external_log_label and args.external_log_label_prefix:
        raise SystemExit(
            "use either --external-log-label or --external-log-label-prefix, not both"
        )
    if args.hard_min_voltage_v > args.min_voltage_v:
        raise SystemExit(
            "--hard-min-voltage-v must not exceed the --min-voltage-v warning level"
        )


def _condition_completed(record: dict[str, Any]) -> bool:
    return record.get("outcome") in {"completed", "completed_with_warnings"}


def _condition_has_voltage_warning(record: dict[str, Any]) -> bool:
    if record.get("outcome") == "completed_with_warnings":
        return True
    direct = record.get("voltage_warnings")
    if isinstance(direct, dict) and int(direct.get("event_count", 0)) > 0:
        return True
    return any(
        isinstance(capture.get("voltage_warnings"), dict)
        and int(capture["voltage_warnings"].get("event_count", 0)) > 0
        for capture in record.get("captures", [])
    )


def _archive_campaign(
    campaign_dir: Path, args: argparse.Namespace, plan: CampaignPlan, fixture_path: Path
) -> dict[str, str]:
    """Preserve an incomplete campaign while holding its existing runner lock."""
    manifest_path = campaign_dir / "campaign_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SystemExit(f"cannot restart campaign without a manifest: {campaign_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("plan", {}).get("name") != plan.name:
        raise SystemExit("cannot restart campaign with a different plan")
    if (
        manifest.get("servo_id") != args.servo_id
        or manifest.get("servo_label") != args.servo_label
    ):
        raise SystemExit("cannot restart campaign with a different servo")
    if manifest.get("fixture_sha256") != file_sha256(fixture_path):
        raise SystemExit("cannot restart campaign with a different fixture model")
    if manifest.get("status") in {"completed", "completed_with_warnings"}:
        raise SystemExit("cannot restart a completed campaign; choose a new --run-dir")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = campaign_dir.with_name(f"{campaign_dir.name}.archived-{timestamp}")
    if archive.exists():
        raise SystemExit(f"refusing to overwrite campaign archive: {archive}")
    provenance = {
        "directory": str(archive),
        "manifest_sha256": file_sha256(manifest_path),
    }
    campaign_dir.rename(archive)
    print(f"ARCHIVED {campaign_dir} -> {archive}", flush=True)
    return provenance


def _run_hardware_campaign(
    args: argparse.Namespace,
    plan: CampaignPlan,
    selected: tuple[CampaignCondition, ...],
    git_state: dict[str, Any],
    campaign_dir: Path,
) -> int:
    manifest_path = campaign_dir / "campaign_manifest.json"
    fixture_path = args.fixture_mjcf.expanduser().resolve()
    restarted_from = None
    if args.restart and campaign_dir.exists():
        restarted_from = _archive_campaign(campaign_dir, args, plan, fixture_path)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        _validate_existing_manifest(manifest, args, plan, fixture_path, git_state)
    else:
        if campaign_dir.exists():
            raise SystemExit(
                f"campaign directory exists without a manifest: {campaign_dir}"
            )
        campaign_dir.mkdir(parents=True)
        manifest = _new_manifest(args, plan, fixture_path, git_state)
        if restarted_from is not None:
            manifest["restarted_from"] = restarted_from
        _write_manifest(manifest_path, manifest)

    completed = {
        item.get("condition_id")
        for item in manifest["conditions"]
        if _condition_completed(item)
    }
    for condition in selected:
        if condition.condition_id in completed:
            print(f"SKIP completed condition {condition.condition_id}", flush=True)
            continue
        output = _condition_output(campaign_dir, plan, condition)
        if output.exists() or (
            output.suffix == ".npz" and output.with_suffix(".json").exists()
        ):
            raise SystemExit(
                f"incomplete condition output already exists: {output}; "
                "use --restart to archive this campaign and start fresh, "
                "or choose a new campaign directory"
            )
        print(f"\nRUN {condition.condition_id}: {condition.description}", flush=True)
        if condition.kind == "repeatability":
            returncode, record = _run_repeatability_condition(args, condition, output)
        else:
            returncode, record = _run_capture_condition(args, condition, output)
        manifest["conditions"].append(record)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        if returncode:
            manifest["status"] = "failed"
            manifest["failed_condition"] = condition.condition_id
            _write_manifest(manifest_path, manifest)
            return returncode
        _write_manifest(manifest_path, manifest)

    completed = {
        item.get("condition_id")
        for item in manifest["conditions"]
        if _condition_completed(item)
    }
    if manifest.get("failed_condition") in completed:
        manifest.pop("failed_condition")
    has_voltage_warnings = any(
        _condition_has_voltage_warning(item) for item in manifest["conditions"]
    )
    if completed == {item.condition_id for item in plan.conditions}:
        manifest["status"] = (
            "completed_with_warnings" if has_voltage_warnings else "completed"
        )
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        manifest["status"] = (
            "partial_with_warnings" if has_voltage_warnings else "partial"
        )
    _write_manifest(manifest_path, manifest)
    print(f"Campaign status={manifest['status']}: {campaign_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = configured_plan(args)
    _validate_args(args, plan)
    selected = selected_conditions(args, plan)

    print(f"WR2 HTD-45H campaign plan={plan.name}", flush=True)
    print(f"mode={'HARDWARE' if args.execute else 'PREFLIGHT'}", flush=True)
    print(f"setup={plan.setup}", flush=True)
    print(f"cooldown_target_c={args.cooldown_target_c:.1f} (CLI)", flush=True)
    for index, condition in enumerate(selected, start=1):
        output = Path("/tmp") / f"wr2_preflight_{condition.condition_id}.npz"
        print(
            f"\nPREFLIGHT [{index}/{len(selected)}] {condition.condition_id}: "
            f"{condition.description}",
            flush=True,
        )
        returncode = _run(capture_command(args, condition, output, execute=False))
        if returncode:
            return returncode
    if not args.execute:
        print("\nCampaign preflight passed; no hardware was opened.")
        return 0

    git_state = _git_state()
    if not git_state["worktree_clean"]:
        raise SystemExit("hardware capture requires a clean Git worktree")
    campaign_dir = _campaign_directory(args, plan)
    with _campaign_lock(campaign_dir):
        return _run_hardware_campaign(args, plan, selected, git_state, campaign_dir)


if __name__ == "__main__":
    raise SystemExit(main())
