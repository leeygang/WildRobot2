"""Versioned campaign-plan loading for HTD-45H characterization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PLAN_DIRECTORY = Path(__file__).with_name("plans")


@dataclass(frozen=True)
class CampaignCondition:
    condition_id: str
    description: str
    center_deg: float
    kind: str = "capture"
    repeats: int = 1
    amplitudes_deg: str = "2,5,8"
    chirp_end_hz: float = 2.0
    prepare_only: bool = False
    prepare_speed_deg_s: float = 20.0
    settle_s: float = 1.0
    write_deadband_units: int = 0
    move_time_ms: int = 20
    profile: str = "standard"
    sweep_rate_deg_s: float = 1.0
    sweep_cycles: int = 3
    constant_hold_s: float | None = None
    required_static_torque_nm: float | None = None
    max_static_torque_nm: float | None = None
    max_predicted_torque_nm: float | None = None
    max_temperature_c: float | None = None


@dataclass(frozen=True)
class CampaignPlan:
    schema_version: int
    name: str
    description: str
    setup: str
    required_instruments: tuple[str, ...]
    data_goals: tuple[str, ...]
    limitations: tuple[str, ...]
    conditions: tuple[CampaignCondition, ...]
    path: Path


_PLAN_KEYS = {
    "schema_version",
    "name",
    "description",
    "setup",
    "required_instruments",
    "data_goals",
    "limitations",
    "conditions",
}
_CONDITION_KEYS = set(CampaignCondition.__dataclass_fields__)


def available_plans() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in PLAN_DIRECTORY.glob("*.yaml")))


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _expect_keys(mapping: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {unknown}")


def load_plan(name: str) -> CampaignPlan:
    path = PLAN_DIRECTORY / f"{name}.yaml"
    if not path.is_file():
        choices = ", ".join(available_plans())
        raise ValueError(f"unknown servo campaign plan {name!r}; choose from {choices}")
    raw = _mapping(yaml.safe_load(path.read_text()), f"plan {name}")
    _expect_keys(raw, _PLAN_KEYS, f"plan {name}")
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError(f"plan {name} has an unsupported schema version")
    if raw.get("name") != name:
        raise ValueError(f"plan {name} must declare the same name")
    conditions_raw = raw.get("conditions")
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError(f"plan {name} must contain at least one condition")
    conditions: list[CampaignCondition] = []
    for index, item in enumerate(conditions_raw):
        condition = _mapping(item, f"plan {name} condition {index + 1}")
        _expect_keys(condition, _CONDITION_KEYS, f"plan {name} condition {index + 1}")
        parsed = CampaignCondition(**condition)
        if parsed.kind not in {"capture", "repeatability"}:
            raise ValueError(
                f"plan {name} condition {parsed.condition_id} has unknown kind "
                f"{parsed.kind!r}"
            )
        if parsed.profile not in {"standard", "hysteresis"}:
            raise ValueError(
                f"plan {name} condition {parsed.condition_id} has unknown profile "
                f"{parsed.profile!r}"
            )
        if parsed.repeats < 1:
            raise ValueError(
                f"plan {name} condition {parsed.condition_id} has no repetitions"
            )
        conditions.append(parsed)
    identifiers = [item.condition_id for item in conditions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"plan {name} contains duplicate condition IDs")
    return CampaignPlan(
        schema_version=1,
        name=name,
        description=str(raw["description"]),
        setup=str(raw["setup"]),
        required_instruments=tuple(str(item) for item in raw.get("required_instruments", [])),
        data_goals=tuple(str(item) for item in raw.get("data_goals", [])),
        limitations=tuple(str(item) for item in raw.get("limitations", [])),
        conditions=tuple(conditions),
        path=path,
    )
