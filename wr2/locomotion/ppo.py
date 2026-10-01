"""Shared construction of the WR2 Brax PPO networks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from wr2.locomotion.configs import NetworkConfig


def make_network_factory(
    config: NetworkConfig,
    *,
    home_action: Sequence[float],
):
    """Build training/evaluation networks with an exact walk-home initializer.

    Brax's generic tanh-normal policy ignores ``init_noise_std`` and initializes
    its mean around normalized action zero. For WR2, zero is the midpoint of
    each joint range rather than ``walk_home``. This policy head starts at the
    pre-tanh value of ``home_action`` and uses a learned, state-independent
    exploration standard deviation.
    """
    import jax
    import jax.numpy as jnp
    from flax import linen
    from brax.training import networks as brax_networks
    from brax.training.agents.ppo import networks as ppo_networks

    if config.distribution_type != "tanh_normal":
        raise ValueError("WR2's absolute action contract requires tanh_normal")
    minimum_std = 0.001
    if config.init_noise_std <= minimum_std:
        raise ValueError("network.init_noise_std must exceed 0.001")

    home = np.asarray(home_action, dtype=np.float32)
    if home.ndim != 1 or not np.all(np.isfinite(home)):
        raise ValueError("home_action must be a finite one-dimensional vector")
    if np.any(np.abs(home) >= 1.0):
        raise ValueError("home_action must lie strictly inside (-1, 1)")
    home_latent = np.arctanh(home).astype(np.float32)
    raw_std = float(np.log(np.expm1(config.init_noise_std - minimum_std)))
    activation = getattr(jax.nn, config.activation)

    class HomeCenteredTanhPolicy(linen.Module):
        hidden_layer_sizes: tuple[int, ...]

        @linen.compact
        def __call__(self, observation):
            hidden = brax_networks.MLP(
                layer_sizes=self.hidden_layer_sizes,
                activation=activation,
                activate_final=True,
            )(observation)

            def home_bias_initializer(_key, shape, dtype=jnp.float32):
                if tuple(shape) != tuple(home_latent.shape):
                    raise ValueError("Policy mean size does not match home_action")
                return jnp.asarray(home_latent, dtype=dtype)

            mean = linen.Dense(
                home.size,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=home_bias_initializer,
                name="mean",
            )(hidden)

            def std_initializer(_key, shape, dtype=jnp.float32):
                return jnp.full(shape, raw_std, dtype=dtype)

            std_parameter = self.param(
                "raw_std",
                std_initializer,
                (home.size,),
            )
            return jnp.concatenate(
                [mean, jnp.broadcast_to(std_parameter, mean.shape)], axis=-1
            )

    def factory(
        observation_size,
        action_size: int,
        preprocess_observations_fn=lambda observation, _: observation,
    ):
        if action_size != home.size:
            raise ValueError(
                f"Policy action size {action_size} does not match "
                f"home_action size {home.size}"
            )
        base = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=config.value_hidden_layer_sizes,
            activation=activation,
            policy_obs_key="state",
            value_obs_key="privileged_state",
            distribution_type="tanh_normal",
        )
        policy_module = HomeCenteredTanhPolicy(config.policy_hidden_layer_sizes)
        actor_observation_shape = (
            observation_size["state"]
            if isinstance(observation_size, Mapping)
            else observation_size
        )
        actor_observation_size = int(np.prod(actor_observation_shape))
        dummy_observation = jnp.zeros((1, actor_observation_size))

        def init(key):
            return policy_module.init(key, dummy_observation)

        def apply(processor_params, policy_params, observation):
            if isinstance(observation, Mapping):
                processor_state = (
                    None
                    if processor_params is None
                    else brax_networks.normalizer_select(processor_params, "state")
                )
                actor_observation = preprocess_observations_fn(
                    observation["state"],
                    processor_state,
                )
            else:
                actor_observation = preprocess_observations_fn(
                    observation, processor_params
                )
            return policy_module.apply(policy_params, actor_observation)

        policy_network = brax_networks.FeedForwardNetwork(init=init, apply=apply)
        return base.replace(policy_network=policy_network)

    return factory
