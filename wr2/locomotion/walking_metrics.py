"""Walking quality and actuator-safety metrics shared by training agents."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


WALKING_GATE_NAMES = frozenset(
    {
        "walking_score",
        "episode_length",
        "fall_rate",
        "forward_velocity_ratio",
        "forward_velocity_error",
        "lateral_velocity",
        "yaw_rate_error",
        "contact_phase",
        "bilateral_foot_use",
        "double_support",
        "feet_phase",
        "torque_rms",
        "torque_peak",
        "torque_exposure",
        "action_saturation",
        "finite_state",
    }
)


@dataclass(frozen=True)
class WalkingGoal:
    """Acceptance thresholds for one deterministic walking evaluation."""

    min_walking_score: float
    min_episode_length: float
    max_fall_rate: float
    min_forward_velocity_ratio: float
    max_forward_velocity_error_m_s: float
    max_abs_lateral_velocity_m_s: float
    max_yaw_rate_error_rad_s: float
    min_contact_phase_match: float
    min_each_foot_swing_fraction: float
    max_double_support: float
    min_feet_phase_score: float
    max_actuator_torque_rms_nm: float
    max_mean_step_peak_torque_nm: float
    max_sustained_torque_exposure: float
    max_action_saturation_fraction: float
    max_nonfinite_state: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> "WalkingGoal":
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - value.keys())
        unknown = sorted(value.keys() - expected)
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")
        result = cls(**{name: float(value[name]) for name in expected})
        for name, number in asdict(result).items():
            if not math.isfinite(number):
                raise ValueError(f"{label}.{name} must be finite")
        if result.min_episode_length <= 0.0:
            raise ValueError(f"{label}.min_episode_length must be positive")
        probability_fields = (
            "max_fall_rate",
            "min_contact_phase_match",
            "min_each_foot_swing_fraction",
            "max_double_support",
            "max_action_saturation_fraction",
            "min_forward_velocity_ratio",
        )
        for name in probability_fields:
            if not 0.0 <= getattr(result, name) <= 1.0:
                raise ValueError(f"{label}.{name} must be in [0, 1]")
        for name in expected:
            if getattr(result, name) < 0.0:
                raise ValueError(f"{label}.{name} must be non-negative")
        return result


@dataclass(frozen=True)
class WalkingGoalResult:
    passed: bool
    score: float
    gates: dict[str, bool]
    values: dict[str, float]

    @property
    def passed_gate_count(self) -> int:
        return sum(self.gates.values())

    def passes(self, required_gates: Sequence[str]) -> bool:
        unknown = set(required_gates) - self.gates.keys()
        if unknown:
            raise ValueError(f"Unknown walking gates: {', '.join(sorted(unknown))}")
        return all(self.gates[name] for name in required_gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "gates": self.gates,
            "values": self.values,
        }


_METRIC_NAMES = {
    "episode_length": "eval/avg_episode_length",
    "fall_rate": "eval/episode_fall",
    "command_forward_m_s": "eval/episode_command_forward_m_s_per_step",
    "forward_velocity_m_s": "eval/episode_forward_velocity_m_s_per_step",
    "forward_velocity_error_m_s": ("eval/episode_forward_velocity_error_m_s_per_step"),
    "lateral_velocity_m_s": "eval/episode_lateral_velocity_m_s_per_step",
    "yaw_rate_error_rad_s": "eval/episode_yaw_rate_error_rad_s_per_step",
    "contact_phase_match": "eval/episode_contact_phase_match_per_step",
    "left_foot_contact_fraction": "eval/episode_left_foot_contact_per_step",
    "right_foot_contact_fraction": "eval/episode_right_foot_contact_per_step",
    "double_support": "eval/episode_double_support_per_step",
    "feet_phase_score": "eval/episode_feet_phase_tracking_per_step",
    "actuator_torque_rms_nm": ("eval/episode_actuator_torque_rms_nm_per_step"),
    "mean_step_peak_torque_nm": ("eval/episode_actuator_torque_peak_nm_per_step"),
    "sustained_torque_exposure": ("eval/episode_sustained_torque_exposure_per_step"),
    "action_saturation_fraction": ("eval/episode_action_saturation_fraction_per_step"),
    "nonfinite_state": "eval/episode_nonfinite_state",
}


def walking_score(
    *,
    episode_length: float,
    target_episode_length: int,
    velocity_error_m_s: float,
    velocity_sigma_m_s: float,
    contact_match: float,
    double_support: float,
) -> float:
    """Rank policies by survival, commanded motion, and alternating support."""
    survival = min(max(episode_length / target_episode_length, 0.0), 1.0)
    velocity_score = math.exp(-((velocity_error_m_s / velocity_sigma_m_s) ** 2))
    return (
        survival
        * velocity_score
        * (0.5 + 0.5 * contact_match)
        * (1.0 - 0.5 * double_support)
    )


def minimum_foot_swing_fraction(
    left_foot_contact_fraction: float,
    right_foot_contact_fraction: float,
) -> float:
    """Return the smaller commanded-walking swing fraction across both feet."""
    left_contact = min(max(left_foot_contact_fraction, 0.0), 1.0)
    right_contact = min(max(right_foot_contact_fraction, 0.0), 1.0)
    return min(1.0 - left_contact, 1.0 - right_contact)


def _scalar(metrics: Mapping[str, Any], name: str) -> float:
    key = _METRIC_NAMES[name]
    if key not in metrics:
        raise ValueError(f"Evaluation metrics are missing {key}")
    value = metrics[key]
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation metric {key} is not scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"Evaluation metric {key} is not finite")
    return result


def evaluate_transition_goal(
    metrics: Mapping[str, Any], goal: WalkingGoal
) -> WalkingGoalResult:
    """Gate recovery without letting standing samples dilute walking error.

    Do not apply the fixed-command walking score, speed ratio or foot-use gates
    to a mixed standing/walking episode. Reuse the stage's existing health,
    tracking and actuator thresholds; this introduces no new acceptance values.
    """
    names = (
        "episode_length",
        "fall_rate",
        "action_saturation_fraction",
        "nonfinite_state",
        "actuator_torque_rms_nm",
        "mean_step_peak_torque_nm",
        "sustained_torque_exposure",
    )
    values = {name: _scalar(metrics, name) for name in names}
    count = float(metrics["eval/episode_walking_sample_count"])
    error = float(metrics["eval/episode_walking_velocity_error_sum"])
    if not math.isfinite(count) or count <= 0 or not math.isfinite(error) or error < 0:
        raise ValueError(
            "Transition evaluation requires finite walking samples and error"
        )
    values["forward_velocity_error_m_s"] = error / count
    gates = {
        "episode_length": values["episode_length"] >= goal.min_episode_length,
        "fall_rate": values["fall_rate"] <= goal.max_fall_rate,
        "forward_velocity_error": values["forward_velocity_error_m_s"]
        <= goal.max_forward_velocity_error_m_s,
        "action_saturation": values["action_saturation_fraction"]
        <= goal.max_action_saturation_fraction,
        "finite_state": values["nonfinite_state"] <= goal.max_nonfinite_state,
        "torque_rms": values["actuator_torque_rms_nm"]
        <= goal.max_actuator_torque_rms_nm,
        "torque_peak": values["mean_step_peak_torque_nm"]
        <= goal.max_mean_step_peak_torque_nm,
        "torque_exposure": values["sustained_torque_exposure"]
        <= goal.max_sustained_torque_exposure,
    }
    return WalkingGoalResult(all(gates.values()), 0.0, gates, values)


def evaluate_walking_goal(
    metrics: Mapping[str, Any],
    goal: WalkingGoal,
    *,
    target_episode_length: int,
    velocity_sigma_m_s: float,
) -> WalkingGoalResult:
    """Evaluate locomotion and servo gates against one aggregate metric row."""
    values = {name: _scalar(metrics, name) for name in _METRIC_NAMES}
    if values["command_forward_m_s"] <= 0.0:
        raise ValueError("Walking-goal evaluation requires a positive forward command")
    values["forward_velocity_ratio"] = (
        values["forward_velocity_m_s"] / values["command_forward_m_s"]
    )
    values["minimum_foot_swing_fraction"] = minimum_foot_swing_fraction(
        values["left_foot_contact_fraction"],
        values["right_foot_contact_fraction"],
    )
    values["foot_contact_imbalance"] = abs(
        values["left_foot_contact_fraction"] - values["right_foot_contact_fraction"]
    )
    score = walking_score(
        episode_length=values["episode_length"],
        target_episode_length=target_episode_length,
        velocity_error_m_s=values["forward_velocity_error_m_s"],
        velocity_sigma_m_s=velocity_sigma_m_s,
        contact_match=values["contact_phase_match"],
        double_support=values["double_support"],
    )
    gates = {
        "walking_score": score >= goal.min_walking_score,
        "episode_length": values["episode_length"] >= goal.min_episode_length,
        "fall_rate": values["fall_rate"] <= goal.max_fall_rate,
        "forward_velocity_ratio": (
            values["forward_velocity_ratio"] + 1e-9 >= goal.min_forward_velocity_ratio
        ),
        "forward_velocity_error": (
            values["forward_velocity_error_m_s"] <= goal.max_forward_velocity_error_m_s
        ),
        "lateral_velocity": (
            abs(values["lateral_velocity_m_s"]) <= goal.max_abs_lateral_velocity_m_s
        ),
        "yaw_rate_error": (
            values["yaw_rate_error_rad_s"] <= goal.max_yaw_rate_error_rad_s
        ),
        "contact_phase": (
            values["contact_phase_match"] >= goal.min_contact_phase_match
        ),
        "bilateral_foot_use": (
            values["minimum_foot_swing_fraction"] >= goal.min_each_foot_swing_fraction
        ),
        "double_support": values["double_support"] <= goal.max_double_support,
        "feet_phase": values["feet_phase_score"] >= goal.min_feet_phase_score,
        "torque_rms": (
            values["actuator_torque_rms_nm"] <= goal.max_actuator_torque_rms_nm
        ),
        "torque_peak": (
            values["mean_step_peak_torque_nm"] <= goal.max_mean_step_peak_torque_nm
        ),
        "torque_exposure": (
            values["sustained_torque_exposure"] <= goal.max_sustained_torque_exposure
        ),
        "action_saturation": (
            values["action_saturation_fraction"] <= goal.max_action_saturation_fraction
        ),
        "finite_state": values["nonfinite_state"] <= goal.max_nonfinite_state,
    }
    return WalkingGoalResult(
        passed=all(gates.values()), score=score, gates=gates, values=values
    )


def walking_goal_to_dict(goal: WalkingGoal) -> dict[str, float]:
    return asdict(goal)
