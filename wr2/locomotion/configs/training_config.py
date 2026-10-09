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
    forward_progress: float
    yaw_rate: float
    angular_velocity_xy: float
    upright: float
    torso_height: float
    alive: float
    pose: float
    action_rate: float
    joint_velocity: float
    foot_slip: float
    mechanical_power: float
    actuator_torque_squared: float
    feet_phase: float
    contact_phase: float
    both_feet_contact: float
    close_feet: float
    feet_orientation: float


@dataclass(frozen=True)
class ObservationNoise:
    joint_position_uniform_rad: float
    joint_velocity_uniform_rad_s: float
    gyro_cutoff_hz: float
    gyro_colored_std_rad_s: float
    gyro_bias_walk_std_rad_s: float
    gyro_white_std_rad_s: float
    gyro_amplitude_min: float
    gyro_amplitude_max: float
    projected_gravity_cutoff_hz: float
    projected_gravity_colored_std_rad: float
    projected_gravity_bias_walk_std_rad: float
    projected_gravity_white_std_rad: float
    projected_gravity_amplitude_min: float
    projected_gravity_amplitude_max: float


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
    body_mass_scale: tuple[float, float]
    target_bias_rad: tuple[float, float]
    initial_torso_roll_rad: tuple[float, float]
    initial_torso_pitch_rad: tuple[float, float]
    backlash_rad: tuple[float, float]
    max_speed_scale: tuple[float, float]
    brake_torque_scale: tuple[float, float]
    passive_active_ratio_scale: tuple[float, float]


@dataclass(frozen=True)
class ZMPReferenceConfig:
    phase_samples: int
    command_samples: int
    com_height_m: float
    single_double_support_ratio: float
    ik_damping: float
    ik_max_iterations: int
    max_position_residual_m: float
    max_orientation_residual_rad: float


@dataclass(frozen=True)
class WalkingEnvConfig:
    episode_length: int
    solver_iterations: int
    heading_observation: bool
    active_groups: tuple[str, ...]
    pose_weights: tuple[float, ...]
    policy_action_scale_rad: float
    action_delay_steps: int
    privileged_linear_velocity_scale: float
    privileged_actuator_force_scale: float
    reset_joint_noise_rad: float
    reset_velocity_noise_rad_s: float
    command_forward_range_m_s: tuple[float, float]
    command_lateral_range_m_s: tuple[float, float]
    command_yaw_range_rad_s: tuple[float, float]
    zero_command_probability: float
    command_resample_steps: int
    command_active_threshold_m_s: float
    gait_cycle_s: float
    randomize_gait_phase_on_reset: bool
    swing_height_m: float
    feet_phase_tracking_sigma_m2: float
    min_feet_lateral_distance_m: float
    contact_force_threshold_n: float
    target_torso_height_m: float
    terminate_height_m: float
    terminate_max_height_m: float
    velocity_reward_sigma: float
    velocity_tracking_sigma: float
    yaw_tracking_sigma: float
    height_tracking_sigma: float
    torque_exposure_time_constant_s: float
    zmp_reference: ZMPReferenceConfig
    observation_noise: ObservationNoise
    rewards: RewardWeights
    randomization: DynamicsRandomization


@dataclass(frozen=True)
class PPOConfig:
    backend: str
    num_timesteps: int
    num_envs: int
    num_evals: int
    num_eval_envs: int
    evaluation_forward_command_m_s: float
    learning_rate: float
    entropy_cost: float
    discounting: float
    gae_lambda: float
    clipping_epsilon: float
    max_grad_norm: float
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    normalize_observations: bool
    normalize_advantage: bool
    value_loss_coef: float
    desired_kl: float
    learning_rate_schedule: str
    training_metrics_steps: int
    mirror_loss_coeff: float = 0.0


@dataclass(frozen=True)
class NetworkConfig:
    policy_hidden_layer_sizes: tuple[int, ...]
    value_hidden_layer_sizes: tuple[int, ...]
    activation: str
    distribution_type: str
    noise_std_type: str
    init_noise_std: float


@dataclass(frozen=True)
class CheckpointConfig:
    save_every_evaluation: bool
    keep_best: bool
    selection_metric: str


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
    network: NetworkConfig
    checkpoints: CheckpointConfig
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


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_float(item, f"{name}[{index}]") for index, item in enumerate(value))


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
        "network",
        "checkpoints",
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
    signed_or_zero_ranges = {
        "target_bias_rad",
        "initial_torso_roll_rad",
        "initial_torso_pitch_rad",
        "backlash_rad",
    }
    for name in random_fields - {"enabled"} - signed_or_zero_ranges:
        if getattr(randomization, name)[0] <= 0.0:
            raise ValueError(f"domain_randomization.{name} must stay positive")
    if randomization.backlash_rad[0] < 0.0:
        raise ValueError("domain_randomization.backlash_rad must be non-negative")

    env_raw = _mapping(root["environment"], "environment")
    # Historical snapshots retain their v3 observation/reward contract.
    env_raw = {"heading_observation": False, "solver_iterations": 1, **env_raw}
    if isinstance(env_raw["solver_iterations"], bool):
        raise ValueError("environment.solver_iterations must be an integer")
    nested_env_fields = {
        "observation_noise",
        "rewards",
        "randomization",
        "zmp_reference",
    }
    env_fields = {field.name for field in fields(WalkingEnvConfig)} - nested_env_fields
    _expect_keys(env_raw, env_fields | {"zmp_reference"}, "environment")
    range_fields = {
        "command_forward_range_m_s",
        "command_lateral_range_m_s",
        "command_yaw_range_rad_s",
    }
    integer_fields = {
        "episode_length", "solver_iterations", "action_delay_steps", "command_resample_steps"
    }
    boolean_fields = {"randomize_gait_phase_on_reset", "heading_observation"}
    vector_fields = {"pose_weights"}
    float_fields = (
        env_fields
        - integer_fields
        - boolean_fields
        - {"active_groups"}
        - range_fields
        - vector_fields
    )
    active_groups = env_raw["active_groups"]
    if not isinstance(active_groups, list) or not all(
        isinstance(value, str) and value for value in active_groups
    ):
        raise ValueError("environment.active_groups must be a list of names")
    zmp_raw = _mapping(env_raw["zmp_reference"], "environment.zmp_reference")
    zmp_fields = {field.name for field in fields(ZMPReferenceConfig)}
    _expect_keys(zmp_raw, zmp_fields, "environment.zmp_reference")
    zmp_integer_fields = {"phase_samples", "command_samples", "ik_max_iterations"}
    zmp_reference = ZMPReferenceConfig(
        **{
            name: _positive_int(zmp_raw[name], f"environment.zmp_reference.{name}")
            for name in zmp_integer_fields
        },
        **{
            name: _float(zmp_raw[name], f"environment.zmp_reference.{name}")
            for name in zmp_fields - zmp_integer_fields
        },
    )
    environment = WalkingEnvConfig(
        heading_observation=_bool(
            env_raw["heading_observation"], "environment.heading_observation"
        ),
        episode_length=_positive_int(
            env_raw["episode_length"], "environment.episode_length"
        ),
        solver_iterations=_positive_int(
            env_raw["solver_iterations"], "environment.solver_iterations"
        ),
        action_delay_steps=int(env_raw["action_delay_steps"]),
        command_resample_steps=_positive_int(
            env_raw["command_resample_steps"], "environment.command_resample_steps"
        ),
        randomize_gait_phase_on_reset=_bool(
            env_raw["randomize_gait_phase_on_reset"],
            "environment.randomize_gait_phase_on_reset",
        ),
        active_groups=tuple(active_groups),
        pose_weights=_float_tuple(env_raw["pose_weights"], "environment.pose_weights"),
        zmp_reference=zmp_reference,
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
    if environment.policy_action_scale_rad <= 0.0:
        raise ValueError("environment.policy_action_scale_rad must be positive")
    noise = environment.observation_noise
    for name in ("gyro_cutoff_hz", "projected_gravity_cutoff_hz"):
        if getattr(noise, name) <= 0.0:
            raise ValueError(f"observation_noise.{name} must be positive")
    for prefix in ("gyro", "projected_gravity"):
        minimum = getattr(noise, f"{prefix}_amplitude_min")
        maximum = getattr(noise, f"{prefix}_amplitude_max")
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(
                f"observation_noise.{prefix}_amplitude_min/max must be "
                "positive and ordered"
            )
    if environment.privileged_linear_velocity_scale <= 0.0:
        raise ValueError(
            "environment.privileged_linear_velocity_scale must be positive"
        )
    if environment.privileged_actuator_force_scale <= 0.0:
        raise ValueError("environment.privileged_actuator_force_scale must be positive")
    if any(weight < 0.0 for weight in environment.pose_weights):
        raise ValueError("environment.pose_weights must be non-negative")
    if not 0.0 <= environment.zero_command_probability <= 1.0:
        raise ValueError("environment.zero_command_probability must be in [0, 1]")
    if environment.torque_exposure_time_constant_s <= 0.0:
        raise ValueError("environment.torque_exposure_time_constant_s must be positive")
    if environment.command_active_threshold_m_s < 0.0:
        raise ValueError(
            "environment.command_active_threshold_m_s must be non-negative"
        )
    if environment.gait_cycle_s <= 0.0:
        raise ValueError("environment.gait_cycle_s must be positive")
    if environment.swing_height_m <= 0.0:
        raise ValueError("environment.swing_height_m must be positive")
    if environment.contact_force_threshold_n <= 0.0:
        raise ValueError("environment.contact_force_threshold_n must be positive")
    if environment.terminate_height_m >= environment.terminate_max_height_m:
        raise ValueError(
            "environment termination height range must have positive width"
        )
    if environment.velocity_reward_sigma <= 0.0:
        raise ValueError("environment.velocity_reward_sigma must be positive")
    if environment.velocity_tracking_sigma <= 0.0:
        raise ValueError("environment.velocity_tracking_sigma must be positive")
    if environment.feet_phase_tracking_sigma_m2 <= 0.0:
        raise ValueError("environment.feet_phase_tracking_sigma_m2 must be positive")
    if environment.min_feet_lateral_distance_m <= 0.0:
        raise ValueError("environment.min_feet_lateral_distance_m must be positive")
    if zmp_reference.phase_samples < 4 or zmp_reference.phase_samples % 2:
        raise ValueError(
            "environment.zmp_reference.phase_samples must be even and at least four"
        )
    if zmp_reference.com_height_m <= 0.0:
        raise ValueError("environment.zmp_reference.com_height_m must be positive")
    if zmp_reference.single_double_support_ratio <= 0.0:
        raise ValueError(
            "environment.zmp_reference.single_double_support_ratio must be positive"
        )
    if zmp_reference.ik_damping <= 0.0:
        raise ValueError("environment.zmp_reference.ik_damping must be positive")
    if zmp_reference.max_position_residual_m <= 0.0:
        raise ValueError(
            "environment.zmp_reference.max_position_residual_m must be positive"
        )
    if zmp_reference.max_orientation_residual_rad <= 0.0:
        raise ValueError(
            "environment.zmp_reference.max_orientation_residual_rad must be positive"
        )
    if environment.command_lateral_range_m_s != (0.0, 0.0):
        raise ValueError(
            "WR2's current ZMP lookup supports forward commands only; "
            "environment.command_lateral_range_m_s must be [0, 0]"
        )
    if environment.command_yaw_range_rad_s != (0.0, 0.0):
        raise ValueError(
            "WR2's current ZMP lookup supports forward commands only; "
            "environment.command_yaw_range_rad_s must be [0, 0]"
        )

    ppo_raw = _mapping(root["ppo"], "ppo")
    # Old checkpoints/config snapshots retain the mirror-disabled PPO path.
    ppo_raw = {"mirror_loss_coeff": 0.0, **ppo_raw}
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
        "training_metrics_steps",
    }
    ppo = PPOConfig(
        **{
            name: _positive_int(ppo_raw[name], f"ppo.{name}")
            for name in ppo_integer_fields
        },
        **{
            name: _float(ppo_raw[name], f"ppo.{name}")
            for name in ppo_fields
            - ppo_integer_fields
            - {
                "backend",
                "normalize_observations",
                "normalize_advantage",
                "learning_rate_schedule",
            }
        },
        backend=str(ppo_raw["backend"]),
        normalize_observations=_bool(
            ppo_raw["normalize_observations"], "ppo.normalize_observations"
        ),
        normalize_advantage=_bool(
            ppo_raw["normalize_advantage"], "ppo.normalize_advantage"
        ),
        learning_rate_schedule=str(ppo_raw["learning_rate_schedule"]),
    )
    if ppo.backend not in {"rsl_rl", "brax"}:
        raise ValueError("ppo.backend must be rsl_rl or brax")
    if ppo.mirror_loss_coeff < 0.0:
        raise ValueError("ppo.mirror_loss_coeff must be non-negative")
    if ppo.mirror_loss_coeff > 0.0 and ppo.backend != "rsl_rl":
        raise ValueError("ppo.mirror_loss_coeff requires the rsl_rl backend")
    if ppo.learning_rate <= 0.0:
        raise ValueError("ppo.learning_rate must be positive")
    if ppo.entropy_cost < 0.0:
        raise ValueError("ppo.entropy_cost must be non-negative")
    if not 0.0 < ppo.discounting <= 1.0:
        raise ValueError("ppo.discounting must be in (0, 1]")
    if not 0.0 < ppo.gae_lambda <= 1.0:
        raise ValueError("ppo.gae_lambda must be in (0, 1]")
    if ppo.clipping_epsilon <= 0.0:
        raise ValueError("ppo.clipping_epsilon must be positive")
    if ppo.max_grad_norm <= 0.0:
        raise ValueError("ppo.max_grad_norm must be positive")
    if ppo.value_loss_coef < 0.0:
        raise ValueError("ppo.value_loss_coef must be non-negative")
    if ppo.desired_kl <= 0.0:
        raise ValueError("ppo.desired_kl must be positive")
    if ppo.learning_rate_schedule not in {"adaptive", "fixed"}:
        raise ValueError("ppo.learning_rate_schedule must be adaptive or fixed")
    if ppo.evaluation_forward_command_m_s <= 0.0:
        raise ValueError("ppo.evaluation_forward_command_m_s must be positive")
    if not (
        environment.command_forward_range_m_s[0]
        <= ppo.evaluation_forward_command_m_s
        <= environment.command_forward_range_m_s[1]
    ):
        raise ValueError(
            "ppo.evaluation_forward_command_m_s must be inside the training range"
        )

    network_raw = _mapping(root["network"], "network")
    network_fields = {field.name for field in fields(NetworkConfig)}
    _expect_keys(network_raw, network_fields, "network")

    def hidden_sizes(name: str) -> tuple[int, ...]:
        values = network_raw[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"network.{name} must be a non-empty list")
        return tuple(
            _positive_int(value, f"network.{name}[{index}]")
            for index, value in enumerate(values)
        )

    network = NetworkConfig(
        policy_hidden_layer_sizes=hidden_sizes("policy_hidden_layer_sizes"),
        value_hidden_layer_sizes=hidden_sizes("value_hidden_layer_sizes"),
        activation=str(network_raw["activation"]),
        distribution_type=str(network_raw["distribution_type"]),
        noise_std_type=str(network_raw["noise_std_type"]),
        init_noise_std=_float(network_raw["init_noise_std"], "network.init_noise_std"),
    )
    if network.activation not in {"elu", "relu", "silu", "tanh"}:
        raise ValueError("network.activation must be one of: elu, relu, silu, tanh")
    if network.distribution_type not in {"normal", "tanh_normal"}:
        raise ValueError("network.distribution_type must be normal or tanh_normal")
    if network.noise_std_type not in {"scalar", "log"}:
        raise ValueError("network.noise_std_type must be scalar or log")
    if network.init_noise_std <= 0.0:
        raise ValueError("network.init_noise_std must be positive")

    checkpoints_raw = _mapping(root["checkpoints"], "checkpoints")
    checkpoint_fields = {field.name for field in fields(CheckpointConfig)}
    _expect_keys(checkpoints_raw, checkpoint_fields, "checkpoints")
    checkpoints = CheckpointConfig(
        save_every_evaluation=_bool(
            checkpoints_raw["save_every_evaluation"],
            "checkpoints.save_every_evaluation",
        ),
        keep_best=_bool(checkpoints_raw["keep_best"], "checkpoints.keep_best"),
        selection_metric=str(checkpoints_raw["selection_metric"]),
    )
    if checkpoints.selection_metric not in {"acquisition", "walking_score"}:
        raise ValueError(
            "checkpoints.selection_metric must be acquisition or walking_score"
        )

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
        network=network,
        checkpoints=checkpoints,
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
        "network": value["network"],
        "checkpoints": value["checkpoints"],
        "output": value["output"],
    }
