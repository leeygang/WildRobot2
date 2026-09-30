"""Gate completed HTD-45H campaigns and emit a deployment specification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from wr2.tools.servo_sysid.campaign import CONDITIONS


REQUIRED_CONDITIONS = tuple(item.condition_id for item in CONDITIONS)


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_campaign(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    directory = path.expanduser().resolve()
    manifest_path = directory / "campaign_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    captures: dict[str, dict[str, Any]] = {}
    for condition_id in REQUIRED_CONDITIONS:
        matches = sorted(directory.glob(f"*_{condition_id}.json"))
        if len(matches) == 1:
            captures[condition_id] = json.loads(matches[0].read_text())
    return manifest, captures


def _preparation_health(capture: dict[str, Any]) -> tuple[list[float], list[float]]:
    voltage: list[float] = []
    temperature: list[float] = []
    for sample in capture.get("preparation", []):
        if sample.get("voltage_v") is not None:
            voltage.append(float(sample["voltage_v"]))
        if sample.get("temperature_c") is not None:
            temperature.append(float(sample["temperature_c"]))
    for sample in capture.get("cooldown", []):
        if sample.get("voltage_v") is not None:
            voltage.append(float(sample["voltage_v"]))
        if sample.get("temperature_c") is not None:
            temperature.append(float(sample["temperature_c"]))
    health = capture.get("health_summary", {})
    if health.get("min_voltage_v") is not None:
        voltage.append(float(health["min_voltage_v"]))
    if health.get("max_temperature_c") is not None:
        temperature.append(float(health["max_temperature_c"]))
    return voltage, temperature


def analyze_campaigns(
    paths: Sequence[Path],
    *,
    min_servos: int,
    min_voltage_v: float,
    max_temperature_c: float,
    max_tracking_p95_deg: float,
    max_thermal_slope_c_per_min: float,
    safety_factor: float,
    nominal_force_cap_nm: float,
    require_external_measurements: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    campaigns = []
    static_margins: list[float] = []
    dynamic_margins: list[float] = []
    hold_torques: list[float] = []
    voltages: list[float] = []
    temperatures: list[float] = []
    tracking_p95: list[float] = []
    speeds: list[float] = []
    thermal_slopes: list[float] = []
    peak_currents: list[float] = []
    rms_currents: list[float] = []
    servo_labels: set[str] = set()

    for path in paths:
        manifest, captures = _load_campaign(path)
        campaign_name = str(path.expanduser().resolve())
        missing = sorted(set(REQUIRED_CONDITIONS) - set(captures))
        if manifest.get("status") != "completed":
            failures.append(
                f"{campaign_name}: campaign status is {manifest.get('status')}"
            )
        if missing:
            failures.append(f"{campaign_name}: missing conditions {missing}")
        campaign_result: dict[str, Any] = {
            "path": campaign_name,
            "servo_id": manifest.get("servo_id"),
            "conditions": {},
        }
        external_path = path.expanduser().resolve() / "external_measurements.json"
        external = (
            json.loads(external_path.read_text()) if external_path.is_file() else None
        )
        if require_external_measurements and external is None:
            failures.append(f"{campaign_name}: missing external_measurements.json")
        elif external is not None:
            if not bool(external.get("clock_aligned")):
                failures.append(
                    f"{campaign_name}: external measurements are not clock-aligned"
                )
            for name, target in (
                ("peak_current_a", peak_currents),
                ("rms_current_a", rms_currents),
            ):
                value = _finite_number(external.get(name))
                if value is None or value <= 0.0:
                    failures.append(f"{campaign_name}: external {name} is unavailable")
                else:
                    target.append(value)
            external_voltage = _finite_number(external.get("minimum_voltage_v"))
            if external_voltage is None:
                failures.append(
                    f"{campaign_name}: external minimum voltage is invalid"
                )
            else:
                voltages.append(external_voltage)
                if external_voltage < min_voltage_v:
                    failures.append(f"{campaign_name}: external voltage below limit")
        campaign_result["external_measurements"] = external
        labels = {
            str(capture.get("servo_label"))
            for capture in captures.values()
            if capture.get("servo_label")
        }
        if len(labels) != 1:
            failures.append(
                f"{campaign_name}: captures must have one consistent servo label"
            )
        elif manifest.get("servo_label") not in labels:
            failures.append(
                f"{campaign_name}: capture label differs from campaign manifest"
            )
        servo_labels.update(labels)

        for condition_id, capture in captures.items():
            condition_failures: list[str] = []
            if capture.get("condition_id") != condition_id:
                condition_failures.append("condition ID mismatch")
            if capture.get("fixture_sha256") != manifest.get("fixture_sha256"):
                condition_failures.append("fixture hash mismatch")
            if capture.get("outcome") != "completed":
                condition_failures.append(f"outcome={capture.get('outcome')}")
            condition_voltage, condition_temperature = _preparation_health(capture)
            voltages.extend(condition_voltage)
            temperatures.extend(condition_temperature)
            if condition_voltage and min(condition_voltage) < min_voltage_v:
                condition_failures.append("voltage below limit")
            if condition_temperature and max(condition_temperature) >= max_temperature_c:
                condition_failures.append("temperature reaches or exceeds limit")
            trace = capture.get("trace_summary", {})
            if trace.get("tracking_abs_p95_deg") is not None:
                value = float(trace["tracking_abs_p95_deg"])
                tracking_p95.append(value)
                if value > max_tracking_p95_deg:
                    condition_failures.append("tracking p95 above limit")
            if trace.get("observed_peak_speed_rad_s") is not None:
                speeds.append(float(trace["observed_peak_speed_rad_s"]))
            if condition_failures:
                failures.append(
                    f"{campaign_name}/{condition_id}: " + ", ".join(condition_failures)
                )
            campaign_result["conditions"][condition_id] = {
                "passed": not condition_failures,
                "failures": condition_failures,
            }

        if all(name in captures for name in ("E1_static_plus72", "E2_static_minus72")):
            signed_static = [
                abs(float(captures[name]["center_static_torque_nm"]))
                for name in ("E1_static_plus72", "E2_static_minus72")
            ]
            static_margins.append(min(signed_static))
            if min(signed_static) < 3.0:
                failures.append(
                    f"{campaign_name}: signed static coverage is below 3 N m"
                )
        if all(name in captures for name in ("E4_loaded_plus60", "E5_loaded_minus60")):
            dynamic = [
                float(captures[name]["predicted_envelope"]["peak_total_torque_nm"])
                for name in ("E4_loaded_plus60", "E5_loaded_minus60")
            ]
            dynamic_margins.append(min(dynamic))
        if "E7_thermal_hold_plus60" in captures:
            hold = captures["E7_thermal_hold_plus60"]
            hold_torques.append(abs(float(hold["center_static_torque_nm"])))
            slope = hold.get("health_summary", {}).get(
                "temperature_slope_c_per_min_last_60s"
            )
            if slope is None or not math.isfinite(float(slope)):
                failures.append(f"{campaign_name}: thermal slope is unavailable")
            else:
                thermal_slopes.append(float(slope))
                if float(slope) > max_thermal_slope_c_per_min:
                    failures.append(
                        f"{campaign_name}: thermal slope {float(slope):.3f} C/min "
                        "does not demonstrate equilibrium"
                    )
        if all(
            name in captures
            for name in ("E4_loaded_plus60", "E6_deployment_deadband_plus60")
        ):
            baseline = (
                captures["E4_loaded_plus60"]
                .get("trace_summary", {})
                .get("tracking_rmse_deg")
            )
            deadband = (
                captures["E6_deployment_deadband_plus60"]
                .get("trace_summary", {})
                .get("tracking_rmse_deg")
            )
            if (
                baseline is not None
                and deadband is not None
                and float(deadband) > 1.5 * max(float(baseline), 1e-6)
            ):
                failures.append(
                    f"{campaign_name}: deployment deadband raises RMSE by >50%"
                )
        campaigns.append(campaign_result)

    if len(paths) < min_servos or len(servo_labels) < min_servos:
        failures.append(
            f"only {len(servo_labels)} distinct servo labels; require at least {min_servos}"
        )

    peak_tested = min(static_margins) if static_margins else None
    continuous_tested = min(hold_torques) if hold_torques and not failures else None
    conservative_peak = peak_tested / safety_factor if peak_tested else None
    conservative_continuous = (
        continuous_tested / safety_factor if continuous_tested else None
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification_status": "passed" if not failures else "incomplete_or_failed",
        "failures": failures,
        "campaigns": campaigns,
        "coverage": {
            "distinct_servos": len(servo_labels),
            "peak_signed_static_torque_nm": peak_tested,
            "minimum_loaded_dynamic_envelope_nm": (
                min(dynamic_margins) if dynamic_margins else None
            ),
            "thermal_hold_torque_nm": min(hold_torques) if hold_torques else None,
            "minimum_voltage_v": min(voltages) if voltages else None,
            "maximum_temperature_c": max(temperatures) if temperatures else None,
            "maximum_tracking_p95_deg": max(tracking_p95) if tracking_p95 else None,
            "maximum_observed_speed_rad_s": max(speeds) if speeds else None,
            "maximum_final_thermal_slope_c_per_min": (
                max(thermal_slopes) if thermal_slopes else None
            ),
            "maximum_external_peak_current_a": (
                max(peak_currents) if peak_currents else None
            ),
            "maximum_external_rms_current_a": (
                max(rms_currents) if rms_currents else None
            ),
        },
        "deployment_limits": {
            "safety_factor": safety_factor,
            "conservative_peak_torque_nm": conservative_peak,
            "conservative_continuous_torque_nm": conservative_continuous,
            "nominal_sim_force_cap_nm": nominal_force_cap_nm,
            "suggested_training_force_scale_upper": (
                min(1.0, conservative_peak / nominal_force_cap_nm)
                if conservative_peak is not None
                else None
            ),
            "suggested_training_force_scale_lower": (
                min(1.0, conservative_continuous / nominal_force_cap_nm)
                if conservative_continuous is not None
                else None
            ),
        },
        "limitations": [
            "Known fixture torque is inferred from mass properties unless a load-cell log is supplied.",
            "This report does not fit a voltage/temperature-dependent torque-speed curve.",
            "A ten-minute hold is not a continuous rating unless the final temperature slope passes.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/servo_sysid/deployment_spec.json")
    )
    parser.add_argument("--min-servos", type=int, default=3)
    parser.add_argument("--min-voltage-v", type=float, default=9.6)
    parser.add_argument("--max-temperature-c", type=float, default=55.0)
    parser.add_argument("--max-tracking-p95-deg", type=float, default=5.0)
    parser.add_argument("--max-thermal-slope-c-per-min", type=float, default=0.5)
    parser.add_argument("--safety-factor", type=float, default=1.2)
    parser.add_argument("--nominal-force-cap-nm", type=float, default=4.0)
    parser.add_argument(
        "--allow-missing-external-measurements",
        action="store_true",
        help="Allow fixture-only qualification without current-logger evidence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze_campaigns(
        args.campaign_dirs,
        min_servos=args.min_servos,
        min_voltage_v=args.min_voltage_v,
        max_temperature_c=args.max_temperature_c,
        max_tracking_p95_deg=args.max_tracking_p95_deg,
        max_thermal_slope_c_per_min=args.max_thermal_slope_c_per_min,
        safety_factor=args.safety_factor,
        nominal_force_cap_nm=args.nominal_force_cap_nm,
        require_external_measurements=not args.allow_missing_external_measurements,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"qualification_status={report['qualification_status']}")
    print(f"Wrote {output}")
    return 0 if report["qualification_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
