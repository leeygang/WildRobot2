"""Analyze repeated zero and loaded-position captures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


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
        if float(samples[index]["elapsed_s"]) < float(samples[index - 1]["elapsed_s"]):
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
        temperature_rise_c=(temperature[-1] - temperature[0] if temperature else None),
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
        summarize_capture(path, index) for index, path in enumerate(captures, start=1)
    ]
    completed = [item for item in repeats if item.outcome == "completed"]
    failures: list[str] = []
    if expected_repeats is not None and len(repeats) != expected_repeats:
        failures.append(f"found {len(repeats)} captures; expected {expected_repeats}")
    if len(completed) != len(repeats):
        failures.append("one or more captures did not complete")

    def values(name: str) -> np.ndarray:
        return np.asarray([getattr(item, name) for item in completed], dtype=np.float64)

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
                float(np.max(final_temperature)) if final_temperature.size else None
            ),
        },
        "repeats": [asdict(item) for item in repeats],
    }
