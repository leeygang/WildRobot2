"""Configuration for WR2's first standing and walking environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardWeights:
    velocity_xy: float = 1.5
    yaw_rate: float = 0.5
    upright: float = 0.5
    torso_height: float = 0.25
    alive: float = 0.2
    pose: float = -0.02
    action_rate: float = -0.01
    joint_velocity: float = -0.0005
    foot_slip: float = -0.05
    mechanical_power: float = -0.0002


@dataclass(frozen=True)
class ObservationNoise:
    """Provisional noise used for simulation development, not hardware sign-off."""

    joint_position_std_rad: float = 0.003
    joint_velocity_std_rad_s: float = 0.05
    gyro_std_rad_s: float = 0.01
    projected_gravity_std: float = 0.005


@dataclass(frozen=True)
class DynamicsRandomization:
    """Relative ranges around the nominal model."""

    friction_scale: tuple[float, float] = (0.7, 1.3)
    damping_scale: tuple[float, float] = (0.8, 1.2)
    armature_scale: tuple[float, float] = (0.8, 1.2)
    frictionloss_scale: tuple[float, float] = (0.7, 1.3)
    kp_scale: tuple[float, float] = (0.8, 1.2)
    kv_scale: tuple[float, float] = (0.8, 1.2)
    # One-sided because 4 N m is an unvalidated cap, not a measured continuous
    # rating. Fit this range from loaded WR2 tests before policy deployment.
    torque_limit_scale: tuple[float, float] = (0.6, 1.0)


@dataclass(frozen=True)
class WalkingEnvConfig:
    episode_length: int = 1000
    action_scale_rad: float = 0.25
    active_groups: tuple[str, ...] = ("leg",)
    action_delay_steps: int = 1
    reset_joint_noise_rad: float = 0.02
    reset_velocity_noise_rad_s: float = 0.05
    command_forward_range_m_s: tuple[float, float] = (0.0, 0.20)
    command_lateral_range_m_s: tuple[float, float] = (0.0, 0.0)
    command_yaw_range_rad_s: tuple[float, float] = (0.0, 0.0)
    zero_command_probability: float = 0.20
    target_torso_height_m: float = 0.301
    terminate_height_m: float = 0.20
    terminate_projected_gravity_z: float = -0.50
    velocity_tracking_sigma: float = 0.20
    yaw_tracking_sigma: float = 0.40
    height_tracking_sigma: float = 0.03
    observation_noise: ObservationNoise = field(default_factory=ObservationNoise)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    randomization: DynamicsRandomization = field(default_factory=DynamicsRandomization)
