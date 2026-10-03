"""Analyze coarse loaded hysteresis from bounded bidirectional BAM sweeps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from wr2.tools.servo_sysid.core import radians_to_units


TELEMETRY_RESOLUTION_DEG = 0.24
_SWEEP_SEGMENT = re.compile(r"^sweep_(positive|negative)_cycle(\d+)$")


def _mean_by_command_unit(
    command_rad: np.ndarray, measured_position_rad: np.ndarray
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for command, position in zip(command_rad, measured_position_rad, strict=True):
        if not math.isfinite(float(command)) or not math.isfinite(float(position)):
            continue
        grouped.setdefault(radians_to_units(float(command)), []).append(
            math.degrees(float(position))
        )
    return {unit: float(np.mean(values)) for unit, values in grouped.items()}


def summarize_arrays(
    arrays: Mapping[str, np.ndarray], *, center_deg: float
) -> dict[str, Any]:
    """Measure output-position loop width at matched command units."""
    required = {"segment_name", "command_rad", "measured_position_rad"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"hysteresis capture is missing arrays: {missing}")
    segment = np.asarray(arrays["segment_name"]).astype(str)
    command = np.asarray(arrays["command_rad"], dtype=np.float64)
    measured = np.asarray(arrays["measured_position_rad"], dtype=np.float64)
    if not (segment.size == command.size == measured.size):
        raise ValueError("hysteresis capture arrays have different lengths")

    cycle_directions: dict[int, dict[str, np.ndarray]] = {}
    for name in np.unique(segment):
        match = _SWEEP_SEGMENT.fullmatch(str(name))
        if match is None:
            continue
        direction, cycle_text = match.groups()
        cycle_directions.setdefault(int(cycle_text), {})[direction] = segment == name

    center_unit = radians_to_units(math.radians(center_deg))
    failures: list[str] = []
    cycles: list[dict[str, Any]] = []
    all_absolute_widths: list[float] = []
    for cycle_index in sorted(cycle_directions):
        directions = cycle_directions[cycle_index]
        if set(directions) != {"positive", "negative"}:
            failures.append(f"cycle {cycle_index} does not contain both directions")
            continue
        positive_mask = directions["positive"]
        negative_mask = directions["negative"]
        positive = _mean_by_command_unit(command[positive_mask], measured[positive_mask])
        negative = _mean_by_command_unit(command[negative_mask], measured[negative_mask])
        common_units = sorted(set(positive) & set(negative))
        if len(common_units) < 3:
            failures.append(
                f"cycle {cycle_index} has fewer than three matched command units"
            )
            continue
        signed_width = np.asarray(
            [positive[unit] - negative[unit] for unit in common_units],
            dtype=np.float64,
        )
        absolute_width = np.abs(signed_width)
        all_absolute_widths.extend(absolute_width.tolist())
        matched_center_unit = min(common_units, key=lambda unit: abs(unit - center_unit))
        center_positive = positive[matched_center_unit]
        center_negative = negative[matched_center_unit]
        cycles.append(
            {
                "cycle_index": cycle_index,
                "matched_command_count": len(common_units),
                "command_unit_range": [common_units[0], common_units[-1]],
                "center_command_unit": matched_center_unit,
                "center_positive_position_deg": center_positive,
                "center_negative_position_deg": center_negative,
                "center_signed_output_loop_deg": center_positive - center_negative,
                "center_absolute_output_loop_deg": abs(
                    center_positive - center_negative
                ),
                "median_absolute_output_loop_deg": float(
                    np.median(absolute_width)
                ),
                "p95_absolute_output_loop_deg": float(
                    np.percentile(absolute_width, 95.0)
                ),
                "maximum_absolute_output_loop_deg": float(np.max(absolute_width)),
            }
        )

    center_widths = np.asarray(
        [item["center_absolute_output_loop_deg"] for item in cycles],
        dtype=np.float64,
    )
    all_widths = np.asarray(all_absolute_widths, dtype=np.float64)
    if not cycles:
        failures.append("no complete forward/reverse sweep cycles were found")
    return {
        "schema_version": 1,
        "status": "measured" if not failures else "incomplete",
        "failures": failures,
        "center_deg": center_deg,
        "telemetry_resolution_deg": TELEMETRY_RESOLUTION_DEG,
        "interpretation": (
            "Direction-dependent loaded output-position loop width. This is "
            "coarse hysteresis evidence, not isolated mechanical backlash; "
            "values at or below telemetry resolution are not resolvable."
        ),
        "aggregate": {
            "completed_cycles": len(cycles),
            "center_absolute_output_loop_mean_deg": (
                float(np.mean(center_widths)) if center_widths.size else None
            ),
            "center_absolute_output_loop_std_deg": (
                float(np.std(center_widths)) if center_widths.size else None
            ),
            "center_absolute_output_loop_range_deg": (
                float(np.ptp(center_widths)) if center_widths.size else None
            ),
            "all_commands_p95_absolute_output_loop_deg": (
                float(np.percentile(all_widths, 95.0)) if all_widths.size else None
            ),
            "all_commands_maximum_absolute_output_loop_deg": (
                float(np.max(all_widths)) if all_widths.size else None
            ),
        },
        "cycles": cycles,
    }


def summarize_capture(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    metadata_path = resolved.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"capture metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    with np.load(resolved, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    summary = summarize_arrays(arrays, center_deg=float(metadata["center_deg"]))
    return {
        "capture": str(path),
        "condition_id": metadata.get("condition_id"),
        "outcome": metadata.get("outcome"),
        "servo_label": metadata.get("servo_label"),
        "fixture_sha256": metadata.get("fixture_sha256"),
        "hysteresis": summary,
    }


def summarize_captures(paths: Sequence[Path]) -> dict[str, Any]:
    captures = [summarize_capture(path) for path in paths]
    failures = [
        f"{item['capture']}: capture outcome is {item['outcome']}"
        for item in captures
        if item["outcome"] != "completed"
    ]
    failures.extend(
        f"{item['capture']}: {failure}"
        for item in captures
        for failure in item["hysteresis"]["failures"]
    )
    center_means = np.asarray(
        [
            item["hysteresis"]["aggregate"][
                "center_absolute_output_loop_mean_deg"
            ]
            for item in captures
            if item["hysteresis"]["aggregate"][
                "center_absolute_output_loop_mean_deg"
            ]
            is not None
        ],
        dtype=np.float64,
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured" if not failures else "incomplete",
        "failures": failures,
        "capture_count": len(captures),
        "aggregate": {
            "capture_center_loop_mean_deg": (
                float(np.mean(center_means)) if center_means.size else None
            ),
            "capture_center_loop_std_deg": (
                float(np.std(center_means)) if center_means.size else None
            ),
            "capture_center_loop_range_deg": (
                float(np.ptp(center_means)) if center_means.size else None
            ),
        },
        "captures": captures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = summarize_captures(args.captures)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"status={report['status']}")
    print(f"Wrote {output}")
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
