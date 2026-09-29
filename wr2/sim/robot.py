"""Load and validate the versioned WR2 robot description."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from wr2.sim.interface import PositionActionContract
from wr2.sensing.imu import normalize_quaternion_wxyz


DESCRIPTION_ROOT = Path(__file__).resolve().parents[1] / "descriptions"


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _read_order(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


@dataclass(frozen=True)
class MotorSpec:
    name: str
    joint: str
    model: str
    group: str
    home_position_rad: float
    lower_limit_rad: float
    upper_limit_rad: float


@dataclass(frozen=True)
class RobotDescription:
    name: str
    directory: Path
    actuator_names: tuple[str, ...]
    motors: tuple[MotorSpec, ...]
    simulation_timestep_s: float
    control_period_s: float
    imu_sensor_to_torso_quat_wxyz: tuple[float, float, float, float]
    action_home_position_rad_values: tuple[float, ...]
    action: PositionActionContract
    config: dict[str, Any]

    @property
    def actuator_count(self) -> int:
        return len(self.actuator_names)

    @property
    def observation_size(self) -> int:
        return sum(int(field["size"]) for field in self.config["observation"]["fields"])

    @property
    def home_position_rad(self) -> npt.NDArray[np.float32]:
        """Policy action origin from the configured reference keyframe."""
        return np.asarray(self.action_home_position_rad_values, dtype=np.float32)

    @property
    def neutral_position_rad(self) -> npt.NDArray[np.float32]:
        """Mechanical zero/home values declared for each motor."""
        return np.asarray(
            [motor.home_position_rad for motor in self.motors], dtype=np.float32
        )

    @property
    def lower_limit_rad(self) -> npt.NDArray[np.float32]:
        return np.asarray(
            [motor.lower_limit_rad for motor in self.motors], dtype=np.float32
        )

    @property
    def upper_limit_rad(self) -> npt.NDArray[np.float32]:
        return np.asarray(
            [motor.upper_limit_rad for motor in self.motors], dtype=np.float32
        )

    @classmethod
    def load(
        cls,
        name: str = "wr2",
        *,
        description_root: Path = DESCRIPTION_ROOT,
        include_local_calibration: bool = True,
    ) -> "RobotDescription":
        robot_dir = description_root / name
        config = copy.deepcopy(_load_yaml(description_root / "default.yml"))
        _deep_update(config, _load_yaml(robot_dir / "robot.yml"))
        local_calibration = robot_dir / "motors.yml"
        if include_local_calibration and local_calibration.exists():
            _deep_update(config, _load_yaml(local_calibration))

        order = _read_order(robot_dir / "actuator_order.txt")
        motor_config = config.get("motors")
        if not isinstance(motor_config, dict):
            raise ValueError("Configuration is missing the motors mapping")
        if tuple(motor_config) != order:
            raise ValueError(
                "robot.yml motor order must exactly match actuator_order.txt"
            )

        robot_config = config.get("robot", {})
        model_path = robot_dir / str(robot_config.get("mjcf", "wr2.xml"))
        root = ET.parse(model_path).getroot()
        servo_config = config["actuators"]["htd45hServo"]
        servo_default = root.find(".//default[@class='htd45hServo']")
        if servo_default is None:
            raise ValueError("Canonical MJCF is missing the htd45hServo default")
        servo_joint = servo_default.find("joint")
        servo_position = servo_default.find("position")
        if servo_joint is None or servo_position is None:
            raise ValueError("htd45hServo must define joint and position defaults")
        model_values = {
            "kp_sim": float(servo_position.get("kp", "nan")),
            "kv_sim": float(servo_position.get("kv", "nan")),
            "damping": float(servo_joint.get("damping", "nan")),
            "frictionloss": float(servo_joint.get("frictionloss", "nan")),
            "armature": float(servo_joint.get("armature", "nan")),
        }
        for parameter, model_value in model_values.items():
            if not np.isclose(model_value, float(servo_config[parameter])):
                raise ValueError(
                    f"Configured {parameter} does not match htd45hServo MJCF"
                )
        force_range = [
            float(value) for value in servo_position.get("forcerange", "").split()
        ]
        torque_limit = float(servo_config["torque_limit_nm"])
        if len(force_range) != 2 or not np.allclose(
            force_range, [-torque_limit, torque_limit]
        ):
            raise ValueError(
                "Configured torque_limit_nm does not match htd45hServo MJCF"
            )

        actuator_elements = list(root.findall("actuator/*"))
        model_order = tuple(element.get("name", "") for element in actuator_elements)
        if model_order != order:
            raise ValueError("MJCF actuator order does not match actuator_order.txt")

        joints = {
            joint.get("name", ""): joint for joint in root.findall("worldbody//joint")
        }
        motors: list[MotorSpec] = []
        for actuator_name, actuator_element in zip(order, actuator_elements):
            values = motor_config[actuator_name]
            joint_name = str(values["joint"])
            if actuator_element.get("joint") != joint_name:
                raise ValueError(
                    f"Actuator {actuator_name} targets {actuator_element.get('joint')}, "
                    f"but robot.yml declares {joint_name}"
                )
            joint = joints.get(joint_name)
            if joint is None or not joint.get("range"):
                raise ValueError(f"Missing range for joint {joint_name}")
            lower, upper = (float(value) for value in joint.get("range", "").split())
            home = float(values["home_pos_rad"])
            if not lower <= home <= upper:
                raise ValueError(
                    f"Home position for {joint_name} is outside its limits"
                )
            motors.append(
                MotorSpec(
                    name=actuator_name,
                    joint=joint_name,
                    model=str(values["model"]),
                    group=str(values["group"]),
                    home_position_rad=home,
                    lower_limit_rad=lower,
                    upper_limit_rad=upper,
                )
            )

        sim_dt = float(config["simulation"]["timestep_s"])
        control_dt = float(config["control"]["period_s"])
        decimation = int(config["control"]["decimation"])
        if not np.isclose(control_dt, sim_dt * decimation):
            raise ValueError("control period must equal timestep times decimation")

        action_config = config["action"]
        normalized_range = action_config["normalized_range"]
        if normalized_range != [-1.0, 1.0]:
            raise ValueError("WR2 v1 requires a normalized action range of [-1, 1]")
        reference_keyframe = str(action_config["reference_keyframe"])
        key = root.find(f"keyframe/key[@name='{reference_keyframe}']")
        if key is None or not key.get("ctrl"):
            raise ValueError(
                f"MJCF is missing action reference keyframe {reference_keyframe!r}"
            )
        action_home = tuple(float(value) for value in key.get("ctrl", "").split())
        if len(action_home) != len(order):
            raise ValueError(
                f"Keyframe {reference_keyframe!r} has {len(action_home)} controls; "
                f"expected {len(order)}"
            )

        imu_config = config["imu"]
        imu_quat = tuple(
            float(value) for value in imu_config["sensor_to_torso_quat_wxyz"]
        )
        if len(imu_quat) != 4:
            raise ValueError("IMU mounting quaternion must have four values")
        imu_site = root.find(f".//site[@name='{imu_config['site']}']")
        if imu_site is None or not imu_site.get("quat"):
            raise ValueError("Canonical MJCF is missing the configured IMU site")
        site_quat = normalize_quaternion_wxyz(
            [float(value) for value in imu_site.get("quat", "").split()]
        )
        config_quat = normalize_quaternion_wxyz(imu_quat)
        if not np.allclose(site_quat, config_quat, atol=1e-7):
            raise ValueError(
                "robot.yml IMU mounting rotation does not match the MJCF site"
            )

        return cls(
            name=name,
            directory=robot_dir,
            actuator_names=order,
            motors=tuple(motors),
            simulation_timestep_s=sim_dt,
            control_period_s=control_dt,
            imu_sensor_to_torso_quat_wxyz=imu_quat,
            action_home_position_rad_values=action_home,
            action=PositionActionContract(
                scale_rad=float(action_config["scale_rad"]),
                normalized_limit=1.0,
                joint_limit_margin_rad=float(action_config["joint_limit_margin_rad"]),
            ),
            config=config,
        )
