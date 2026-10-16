"""Shared construction of the WR2 Brax PPO networks."""

from __future__ import annotations

from functools import partial

from wr2.locomotion.configs import NetworkConfig


def make_network_factory(config: NetworkConfig):
    """Build the ToddlerBot-aligned residual-action PPO networks."""
    import jax
    from brax.training.agents.ppo import networks as ppo_networks

    # Brax records partial keywords in checkpoint metadata. A closure hides
    # the architecture/distribution and produces an incomplete restore config.
    return partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.value_hidden_layer_sizes,
        activation=getattr(jax.nn, config.activation),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type=config.distribution_type,
        state_dependent_std=config.distribution_type == "tanh_normal",
        noise_std_type=config.noise_std_type,
        init_noise_std=config.init_noise_std,
        # Brax 0.14.2 cannot reload a null mean initializer from metadata.
        # This is its existing default for Normal, and ignored by NormalTanh.
        mean_kernel_init_fn=jax.nn.initializers.lecun_uniform,
    )
