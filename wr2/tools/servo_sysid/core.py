"""Shared fixture, profile, and metric functions for HTD-45H SysID."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt


STANDARD_GRAVITY_M_S2 = 9.80665
SERVO_CENTER_UNIT = 500
SERVO_UNITS_PER_RAD = 1000.0 / math.radians(240.0)
SERVO_RANGE_RAD = (-math.radians(120.0), math.radians(120.0))
DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "htd45h_2650g.xml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def radians_to_units(value: float) -> int:
    return int(np.clip(round(SERVO_CENTER_UNIT + value * SERVO_UNITS_PER_RAD), 0, 1000))


def units_to_radians(value: int) -> float:
    return (int(value) - SERVO_CENTER_UNIT) / SERVO_UNITS_PER_RAD


@dataclass(frozen=True)
class ProfileSegment:
    name: str
    kind: str
    targets_rad: npt.NDArray[np.float64]


def _hold(
    name: str, target: float, duration_s: float, sample_hz: float
) -> ProfileSegment:
    count = max(1, round(duration_s * sample_hz))
    return ProfileSegment(name, "hold", np.full(count, target, dtype=np.float64))


def build_profile(
    *,
    center_deg: float,
    amplitudes_deg: Sequence[float],
    sample_hz: float = 50.0,
    settle_s: float = 1.0,
    step_hold_s: float = 1.0,
    chirp_duration_s: float = 10.0,
    chirp_start_hz: float = 0.1,
    chirp_end_hz: float = 2.0,
    chirp_decay_rate: float = 0.05,
    constant_hold_s: float | None = None,
) -> tuple[ProfileSegment, ...]:
    center = math.radians(center_deg)
    if constant_hold_s is not None:
        return (_hold("constant_hold", center, constant_hold_s, sample_hz),)
    amplitudes = sorted({abs(math.radians(value)) for value in amplitudes_deg})
    if not amplitudes or amplitudes[0] <= 0.0:
        raise ValueError("amplitudes must be positive")
    segments = [_hold("initial_center", center, settle_s, sample_hz)]
    for amplitude in amplitudes:
        label = f"{math.degrees(amplitude):g}deg"
        segments.extend(
            (
                _hold(
                    f"step_positive_{label}", center + amplitude, step_hold_s, sample_hz
                ),
                _hold(f"settle_positive_{label}", center, settle_s, sample_hz),
                _hold(
                    f"step_negative_{label}", center - amplitude, step_hold_s, sample_hz
                ),
                _hold(f"settle_negative_{label}", center, settle_s, sample_hz),
            )
        )
    count = max(2, round(chirp_duration_s * sample_hz))
    time_s = np.arange(count, dtype=np.float64) / sample_hz
    slope = (chirp_end_hz - chirp_start_hz) / chirp_duration_s
    phase = 2.0 * np.pi * (chirp_start_hz * time_s + 0.5 * slope * time_s**2)
    envelope = np.exp(-chirp_decay_rate * time_s)
    for amplitude in amplitudes:
        label = f"{math.degrees(amplitude):g}deg"
        targets = center + amplitude * envelope * np.sin(phase)
        segments.extend(
            (
                ProfileSegment(f"chirp_{label}", "chirp", targets),
                _hold(f"settle_chirp_{label}", center, settle_s, sample_hz),
            )
        )
    return tuple(segments)


@dataclass
class FixtureModel:
    path: Path
    mujoco: Any
    model: Any
    data: Any
    joint_id: int
    qpos_address: int
    dof_address: int
    direction: int
    qpos_offset_rad: float

    @classmethod
    def load(
        cls,
        path: Path = DEFAULT_FIXTURE,
        *,
        joint_name: str = "pitch",
        direction: int = 1,
        qpos_offset_rad: float = 0.0,
    ) -> "FixtureModel":
        import mujoco

        resolved = path.expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(resolved))
        if model.njnt != 1 or model.nq != 1 or model.nv != 1:
            raise ValueError("fixture must be fixed-base with exactly one DOF")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"fixture joint not found: {joint_name}")
        return cls(
            resolved,
            mujoco,
            model,
            mujoco.MjData(model),
            int(joint_id),
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
            int(direction),
            float(qpos_offset_rad),
        )

    @property
    def moving_mass_kg(self) -> float:
        body_id = int(self.model.jnt_bodyid[self.joint_id])
        return float(self.model.body_subtreemass[body_id])

    def _fixture_position(self, hardware_position: float) -> float:
        return self.qpos_offset_rad + self.direction * hardware_position

    def evaluate_static(
        self, positions_rad: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        positions = np.asarray(positions_rad, dtype=np.float64).reshape(-1)
        torque = np.empty_like(positions)
        inertia = np.empty_like(positions)
        full_mass = np.empty((self.model.nv, self.model.nv), dtype=np.float64)
        for index, position in enumerate(positions):
            self.mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.qpos_address] = self._fixture_position(float(position))
            self.mujoco.mj_forward(self.model, self.data)
            try:
                self.mujoco.mj_fullM(self.model, self.data, full_mass)
            except TypeError:
                self.mujoco.mj_fullM(self.model, full_mass, self.data.qM)
            torque[index] = self.direction * self.data.qfrc_bias[self.dof_address]
            inertia[index] = full_mass[self.dof_address, self.dof_address]
        return torque, inertia

    def dynamic_envelope(
        self, segments: Sequence[ProfileSegment], *, sample_hz: float
    ) -> dict[str, float]:
        """Conservative target-trajectory load estimate, excluding step jumps."""
        peak_static = 0.0
        peak_dynamic = 0.0
        peak_total = 0.0
        peak_speed = 0.0
        dt = 1.0 / sample_hz
        for segment in segments:
            positions = np.asarray(segment.targets_rad, dtype=np.float64)
            static, inertia = self.evaluate_static(positions)
            velocity = np.zeros_like(positions)
            acceleration = np.zeros_like(positions)
            if segment.kind == "chirp" and positions.size >= 3:
                velocity = np.gradient(positions, dt)
                acceleration = np.gradient(velocity, dt)
            dynamic = inertia * acceleration
            total = static + dynamic
            peak_static = max(peak_static, float(np.max(np.abs(static))))
            peak_dynamic = max(peak_dynamic, float(np.max(np.abs(dynamic))))
            peak_total = max(peak_total, float(np.max(np.abs(total))))
            peak_speed = max(peak_speed, float(np.max(np.abs(velocity))))
        return {
            "peak_static_torque_nm": peak_static,
            "peak_inertial_torque_nm": peak_dynamic,
            "peak_total_torque_nm": peak_total,
            "peak_speed_rad_s": peak_speed,
        }


def summarize_trace(
    command_rad: npt.ArrayLike,
    position_rad: npt.ArrayLike,
    timestamps_s: npt.ArrayLike,
) -> dict[str, float]:
    command = np.asarray(command_rad, dtype=np.float64)
    position = np.asarray(position_rad, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    error_deg = np.degrees(command - position)
    velocity = (
        np.gradient(position, timestamps)
        if position.size >= 3
        else np.zeros_like(position)
    )
    return {
        "tracking_rmse_deg": float(np.sqrt(np.mean(error_deg**2))),
        "tracking_abs_p95_deg": float(np.percentile(np.abs(error_deg), 95.0)),
        "tracking_abs_max_deg": float(np.max(np.abs(error_deg))),
        "observed_peak_speed_rad_s": float(np.max(np.abs(velocity))),
    }
