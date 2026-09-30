"""Shared construction of the WR2 Brax PPO networks."""

from __future__ import annotations

import functools

from wr2.locomotion.configs import NetworkConfig


def make_network_factory(config: NetworkConfig):
    """Build the exact network factory used by training and evaluation."""
    import jax
    from brax.training.agents.ppo import networks as ppo_networks

    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.value_hidden_layer_sizes,
        activation=getattr(jax.nn, config.activation),
        distribution_type=config.distribution_type,
        noise_std_type=config.noise_std_type,
        init_noise_std=config.init_noise_std,
    )
