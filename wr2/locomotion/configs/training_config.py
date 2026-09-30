"""Typed loader for WR2's dedicated YAML training configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).with_name("ppo_walking.yaml")


@dataclass(frozen=True)
class RewardWeights:
    velocity_xy: float
    yaw_rate: float
    upright: float
    torso_height: float
    alive: float
    pose: float
    action_rate: float
    joint_velocity: float
    foot_slip: float
    mechanical_power: float
    actuator_torque_squared: float


@dataclass(frozen=True)
class ObservationNoise:
    joint_position_std_rad: float
    joint_velocity_std_rad_s: float
    gyro_std_rad_s: float
    projected_gravity_std: float


@dataclass(frozen=True)
class DynamicsRandomization:
    enabled: bool
    friction_scale: tuple[float, float]
    damping_scale: tuple[float, float]
    armature_scale: tuple[float, float]
    frictionloss_scale: tuple[float, float]
    kp_scale: tuple[float, float]
    kv_scale: tuple[float, float]
    torque_limit_scale: tuple[float, float]
    target_bias_rad: tuple[float, float]


@dataclass(frozen=True)
class WalkingEnvConfig:
    episode_length: int
    action_scale_rad: float
    active_groups: tuple[str, ...]
    action_delay_steps: int
    reset_joint_noise_rad: float
    reset_velocity_noise_rad_s: float
    command_forward_range_m_s: tuple[float, float]
    command_lateral_range_m_s: tuple[float, float]
    command_yaw_range_rad_s: tuple[float, float]
    zero_command_probability: float
    target_torso_height_m: float
    terminate_height_m: float
    terminate_projected_gravity_z: float
    velocity_tracking_sigma: float
    yaw_tracking_sigma: float
    height_tracking_sigma: float
    torque_exposure_time_constant_s: float
    observation_noise: ObservationNoise
    rewards: RewardWeights
    randomization: DynamicsRandomization


@dataclass(frozen=True)
class PPOConfig:
    num_timesteps: int
    num_envs: int
    num_evals: int
    num_eval_envs: int
    learning_rate: float
    entropy_cost: float
    discounting: float
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    normalize_observations: bool


@dataclass(frozen=True)
class OutputConfig:
    root: str
    run_prefix: str


@dataclass(frozen=True)
class TrainingConfig:
    version: str
    version_name: str
    seed: int
    environment: WalkingEnvConfig
    ppo: PPOConfig
    output: OutputConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _float_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value list")
    low = _float(value[0], f"{name}[0]")
    high = _float(value[1], f"{name}[1]")
    if high < low:
        raise ValueError(f"{name} must satisfy low <= high")
    return (low, high)


def _numeric_dataclass(cls, raw: dict[str, Any], section: str):
    expected = {field.name for field in fields(cls)}
    _expect_keys(raw, expected, section)
    return cls(**{name: _float(raw[name], f"{section}.{name}") for name in expected})


def load_training_config(
    config_path: str | Path = DEFAULT_TRAINING_CONFIG_PATH,
) -> TrainingConfig:
    """Load and validate one complete WR2 training YAML."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream), "training config")

    root_keys = {
        "version",
        "version_name",
        "seed",
        "environment",
        "reward_weights",
        "observation_noise",
        "domain_randomization",
        "ppo",
        "output",
    }
    _expect_keys(root, root_keys, "training config")

    rewards = _numeric_dataclass(
        RewardWeights,
        _mapping(root["reward_weights"], "reward_weights"),
        "reward_weights",
    )
    observation_noise = _numeric_dataclass(
        ObservationNoise,
        _mapping(root["observation_noise"], "observation_noise"),
        "observation_noise",
    )
    if any(value < 0.0 for value in asdict(observation_noise).values()):
        raise ValueError("observation_noise values must be non-negative")

    random_raw = _mapping(root["domain_randomization"], "domain_randomization")
    random_fields = {field.name for field in fields(DynamicsRandomization)}
    _expect_keys(random_raw, random_fields, "domain_randomization")
    randomization = DynamicsRandomization(
        enabled=_bool(random_raw["enabled"], "domain_randomization.enabled"),
        **{
            name: _float_range(random_raw[name], f"domain_randomization.{name}")
            for name in random_fields - {"enabled"}
        },
    )
    for name in random_fields - {"enabled", "target_bias_rad"}:
        if getattr(randomization, name)[0] <= 0.0:
            raise ValueError(f"domain_randomization.{name} must stay positive")

    env_raw = _mapping(root["environment"], "environment")
    nested_env_fields = {"observation_noise", "rewards", "randomization"}
    env_fields = {field.name for field in fields(WalkingEnvConfig)} - nested_env_fields
    _expect_keys(env_raw, env_fields, "environment")
    range_fields = {
        "command_forward_range_m_s",
        "command_lateral_range_m_s",
        "command_yaw_range_rad_s",
    }
    float_fields = (
        env_fields
        - {
            "episode_length",
            "active_groups",
            "action_delay_steps",
        }
        - range_fields
    )
    active_groups = env_raw["active_groups"]
    if not isinstance(active_groups, list) or not all(
        isinstance(value, str) and value for value in active_groups
    ):
        raise ValueError("environment.active_groups must be a list of names")
    environment = WalkingEnvConfig(
        episode_length=_positive_int(
            env_raw["episode_length"], "environment.episode_length"
        ),
        action_delay_steps=int(env_raw["action_delay_steps"]),
        active_groups=tuple(active_groups),
        **{name: _float(env_raw[name], f"environment.{name}") for name in float_fields},
        **{
            name: _float_range(env_raw[name], f"environment.{name}")
            for name in range_fields
        },
        observation_noise=observation_noise,
        rewards=rewards,
        randomization=randomization,
    )
    if environment.action_delay_steps not in (0, 1):
        raise ValueError("environment.action_delay_steps must be 0 or 1")
    if environment.action_scale_rad <= 0.0:
        raise ValueError("environment.action_scale_rad must be positive")
    if not 0.0 <= environment.zero_command_probability <= 1.0:
        raise ValueError("environment.zero_command_probability must be in [0, 1]")
    if environment.torque_exposure_time_constant_s <= 0.0:
        raise ValueError("environment.torque_exposure_time_constant_s must be positive")

    ppo_raw = _mapping(root["ppo"], "ppo")
    ppo_fields = {field.name for field in fields(PPOConfig)}
    _expect_keys(ppo_raw, ppo_fields, "ppo")
    ppo_integer_fields = {
        "num_timesteps",
        "num_envs",
        "num_evals",
        "num_eval_envs",
        "unroll_length",
        "batch_size",
        "num_minibatches",
        "num_updates_per_batch",
    }
    ppo = PPOConfig(
        **{
            name: _positive_int(ppo_raw[name], f"ppo.{name}")
            for name in ppo_integer_fields
        },
        **{
            name: _float(ppo_raw[name], f"ppo.{name}")
            for name in ppo_fields - ppo_integer_fields - {"normalize_observations"}
        },
        normalize_observations=_bool(
            ppo_raw["normalize_observations"], "ppo.normalize_observations"
        ),
    )
    if ppo.learning_rate <= 0.0:
        raise ValueError("ppo.learning_rate must be positive")
    if ppo.entropy_cost < 0.0:
        raise ValueError("ppo.entropy_cost must be non-negative")
    if not 0.0 < ppo.discounting <= 1.0:
        raise ValueError("ppo.discounting must be in (0, 1]")

    output_raw = _mapping(root["output"], "output")
    _expect_keys(output_raw, {"root", "run_prefix"}, "output")
    output = OutputConfig(
        root=str(output_raw["root"]), run_prefix=str(output_raw["run_prefix"])
    )
    if not output.root:
        raise ValueError("output.root must not be empty")
    if (
        not output.run_prefix
        or Path(output.run_prefix).name != output.run_prefix
        or output.run_prefix in {".", ".."}
    ):
        raise ValueError("output.run_prefix must be a single directory-name prefix")

    return TrainingConfig(
        version=str(root["version"]),
        version_name=str(root["version_name"]),
        seed=int(root["seed"]),
        environment=environment,
        ppo=ppo,
        output=output,
    )


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Return the effective typed config in its YAML section layout."""
    value = asdict(config)
    environment = value["environment"]
    rewards = environment.pop("rewards")
    observation_noise = environment.pop("observation_noise")
    randomization = environment.pop("randomization")
    return {
        "version": value["version"],
        "version_name": value["version_name"],
        "seed": value["seed"],
        "environment": environment,
        "reward_weights": rewards,
        "observation_noise": observation_noise,
        "domain_randomization": randomization,
        "ppo": value["ppo"],
        "output": value["output"],
    }
