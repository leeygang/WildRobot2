"""Versioned YAML configuration and typed loading for WR2 training."""

from wr2.locomotion.configs.training_config import (
    DEFAULT_TRAINING_CONFIG_PATH,
    DynamicsRandomization,
    ObservationNoise,
    OutputConfig,
    PPOConfig,
    RewardWeights,
    TrainingConfig,
    WalkingEnvConfig,
    load_training_config,
    training_config_to_dict,
)

__all__ = [
    "DEFAULT_TRAINING_CONFIG_PATH",
    "DynamicsRandomization",
    "ObservationNoise",
    "OutputConfig",
    "PPOConfig",
    "RewardWeights",
    "TrainingConfig",
    "WalkingEnvConfig",
    "load_training_config",
    "training_config_to_dict",
]
