"""Attribute the installed Brax PPO actor gradient on recorded training batches.

GAE is linear in rewards and values. PPO clipping is not linear in advantages:
all components must share the branch selected by the *total* advantage.
References: https://arxiv.org/abs/1506.02438 and https://arxiv.org/abs/1707.06347.
"""

from functools import partial

import jax
import jax.numpy as jp
from brax.training.agents.ppo import losses


REWARD_GROUPS = (
    "xy_tracking",
    "yaw_tracking",
    "torso_stability",
    "leg_imitation",
    "air_time",
    "other_rewards",
)
COMPONENTS = REWARD_GROUPS + ("value_baseline", "entropy")


def group_rewards(rewards):
    """Group already weighted, dt-integrated rewards; retain every term."""
    members = (
        ("lin_vel_xy",),
        ("ang_vel_z",),
        ("torso_quat", "lin_vel_z", "ang_vel_xy", "torso_roll", "torso_pitch"),
        ("leg_motor_pos",),
        ("feet_air_time",),
    )
    used = {name for group in members for name in group}
    zero = jp.zeros_like(next(iter(rewards.values())))
    grouped = [
        sum((rewards.get(name, zero) for name in group), zero) for group in members
    ]
    grouped.append(
        sum((value for name, value in rewards.items() if name not in used), zero)
    )
    return jp.stack(grouped, axis=-1)


def split_advantages(
    rewards, values, bootstrap, truncation, termination, *, discount, lambda_, normalize
):
    """Return [components,T,B] using one shared total-advantage normalization.

    Reward groups use zero value estimates. The remaining term contains the native
    critic and bootstrap. These are an exact algebraic split of the sampled loss,
    not separately learned component critics or independent causal effects.
    """
    gae = partial(
        losses.compute_gae,
        truncation=truncation,
        termination=termination,
        lambda_=lambda_,
        discount=discount,
    )
    parts = jax.vmap(
        lambda reward: gae(
            rewards=reward,
            values=jp.zeros_like(values),
            bootstrap_value=jp.zeros_like(bootstrap),
        )[1]
    )(rewards)
    baseline = gae(
        rewards=jp.zeros_like(values), values=values, bootstrap_value=bootstrap
    )[1]
    parts = jp.concatenate([parts, baseline[None]], axis=0)
    total = gae(
        rewards=jp.sum(rewards, axis=0), values=values, bootstrap_value=bootstrap
    )[1]
    if normalize:
        denominator = total.std() + 1e-8
        parts = (parts - parts.mean(axis=(1, 2), keepdims=True)) / denominator
        total = (total - total.mean()) / denominator
    return parts, total


def clipped_component_losses(log_prob, old_log_prob, parts, total, epsilon):
    """Use the total surrogate's branch, including native tie derivatives."""
    ratio = jp.exp(log_prob - old_log_prob)
    clipped = jp.clip(ratio, 1 - epsilon, 1 + epsilon)
    first, second = ratio * total, clipped * total
    branch = jp.where(
        first < second, ratio, jp.where(first > second, clipped, (ratio + clipped) / 2)
    )
    return -jp.mean(parts * branch[None], axis=(1, 2))


def tree_dot(first, second):
    return sum(
        jp.vdot(a, b)
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    )


def actor_attribution(
    params, normalizer, data, rng, networks, settings, hip_indices, forward_sign
):
    """Verify actor backpropagation and project components onto forward targets.

    The projection is the infinitesimal change per unit plain SGD learning rate,
    on this fixed observation batch. Actual fresh-Adam changes are measured by the
    caller; Adam updates cannot be decomposed by separately optimizing each term.
    """
    time_data = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), data)
    privileged = (
        time_data.privileged_observation
        if hasattr(time_data, "privileged_observation")
        else time_data.observation
    )
    next_privileged = (
        time_data.next_privileged_observation
        if hasattr(time_data, "next_privileged_observation")
        else time_data.next_observation
    )
    values = networks.value_network.apply(normalizer, params.value, privileged)
    bootstrap = networks.value_network.apply(
        normalizer, params.value, jax.tree.map(lambda x: x[-1], next_privileged)
    )
    extras = time_data.extras["state_extras"]
    truncation = extras["truncation"]
    termination = (1 - time_data.discount) * (1 - truncation)
    rewards = jp.moveaxis(extras["reward_components"], -1, 0)
    parts, total = split_advantages(
        rewards,
        values,
        bootstrap,
        truncation,
        termination,
        discount=settings["discounting"],
        lambda_=settings["gae_lambda"],
        normalize=settings["normalize_advantage"],
    )
    distribution = networks.parametric_action_distribution
    policy = partial(networks.policy_network.apply, normalizer)
    logits, pullback = jax.vjp(
        lambda p: policy(p, time_data.observation), params.policy
    )

    def component_loss(output):
        log_prob = distribution.log_prob(
            output, time_data.extras["policy_extras"]["raw_action"]
        )
        components = clipped_component_losses(
            log_prob,
            time_data.extras["policy_extras"]["log_prob"],
            parts,
            total,
            settings["clipping_epsilon"],
        )
        entropy = -settings["entropy_cost"] * distribution.entropy(output, rng).mean()
        return jp.concatenate([components, entropy[None]])

    output_gradients = jax.jacrev(component_loss)(logits)
    reconstructed = pullback(output_gradients.sum(axis=0))[0]
    native_loss = partial(losses.compute_ppo_loss, ppo_network=networks, **settings)
    (_, metrics), native_gradient = jax.value_and_grad(native_loss, has_aux=True)(
        params, normalizer, data, rng
    )
    difference = jax.tree.map(lambda a, b: a - b, reconstructed, native_gradient.policy)
    relative_error = jp.sqrt(tree_dot(difference, difference)) / (
        jp.sqrt(tree_dot(native_gradient.policy, native_gradient.policy)) + 1e-8
    )
    command = extras["trace"]["command_tb"]
    forward = (
        (command[..., 0] > 0.05)
        & (jp.abs(command[..., 0]) > jp.abs(command[..., 1]))
        & (jp.abs(command[..., 2]) < 1e-6)
    )
    signs = jp.asarray(forward_sign)

    def mean_targets(p):
        targets = (
            distribution.mode(policy(p, time_data.observation))[..., hip_indices]
            * signs
            * 0.25
        )
        return jp.sum(targets * forward[..., None], axis=(0, 1)) / jp.maximum(
            forward.sum(), 1
        )

    target_gradients = jax.jacrev(mean_targets)(params.policy)
    projections = []
    for hip in range(2):
        direction = jax.tree.map(lambda x: -x[hip], target_gradients)
        _, output_direction = jax.jvp(
            lambda p: policy(p, time_data.observation), (params.policy,), (direction,)
        )
        projections.append(
            jp.sum(output_gradients * output_direction[None], axis=(1, 2, 3))
        )
    return {
        "gradient_relative_error": relative_error,
        "advantage_max_error": jp.max(jp.abs(parts.sum(axis=0) - total)),
        "reward_max_error": jp.max(jp.abs(rewards.sum(axis=0) - time_data.reward)),
        "projected_forward_target_sgd": jp.stack(projections, axis=-1),
        "local_forward_mu_pressure": -output_gradients[..., hip_indices] * signs,
        "native_metrics": metrics,
    }
