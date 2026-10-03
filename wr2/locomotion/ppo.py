"""Shared construction of the WR2 Brax PPO networks."""

from __future__ import annotations

from wr2.locomotion.configs import NetworkConfig


def make_network_factory(config: NetworkConfig):
    """Build the ToddlerBot-aligned residual-action PPO networks."""
    import jax
    from brax.training.agents.ppo import networks as ppo_networks

    if config.distribution_type != "normal":
        raise ValueError("WR2 walking requires ToddlerBot's normal residual policy")
    activation = getattr(jax.nn, config.activation)

    def factory(
        observation_size,
        action_size: int,
        preprocess_observations_fn=lambda observation, _: observation,
    ):
        return ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=config.value_hidden_layer_sizes,
            activation=activation,
            policy_obs_key="state",
            value_obs_key="privileged_state",
            distribution_type=config.distribution_type,
            noise_std_type=config.noise_std_type,
            init_noise_std=config.init_noise_std,
        )

    return factory
