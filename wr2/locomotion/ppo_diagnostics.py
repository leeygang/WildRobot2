"""Attribute installed Brax PPO gradients without cancelling float32 components.

GAE/advantage normalization use the original scalar reward. Deterministic
component bookkeeping uses float64; native networks, entropy draws and PPO
updates retain their original dtype and matrix precision.
References: https://arxiv.org/abs/1506.02438, https://arxiv.org/abs/1707.06347.
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


def _native_advantages(
    rewards, values, bootstrap, truncation, termination, *, discount, lambda_, normalize
):
    total = losses.compute_gae(
        truncation,
        termination,
        rewards,
        values,
        bootstrap,
        lambda_=lambda_,
        discount=discount,
    )[1]
    denominator = total.std() + 1e-8 if normalize else jp.ones((), total.dtype)
    if normalize:
        total = (total - total.mean()) / denominator
    return total, denominator


def _component_advantages(
    rewards,
    total,
    denominator,
    truncation,
    termination,
    *,
    discount,
    lambda_,
    normalize,
):
    """Float64 reward-only GAE plus the native value/rounding residual.

    Run this deterministic function in a dedicated x64 scope/JIT. JAX 0.6 cannot
    reliably lower mixed x64 scopes inside a single native-dtype JIT.
    """
    zero = jp.zeros_like(total, dtype=jp.float64)
    parts = jax.vmap(
        lambda reward: losses.compute_gae(
            truncation.astype(jp.float64),
            termination.astype(jp.float64),
            reward,
            zero,
            zero[-1],
            lambda_=lambda_,
            discount=discount,
        )[1]
    )(rewards.astype(jp.float64))
    if normalize:
        parts = (parts - parts.mean(axis=(1, 2), keepdims=True)) / denominator.astype(
            jp.float64
        )
    # This residual includes the critic/bootstrap and native scalar-reward/GAE
    # rounding. It is not a separately learned component critic.
    baseline = total.astype(jp.float64) - parts.sum(axis=0)
    return jp.concatenate([parts, baseline[None]], axis=0)


def split_advantages(
    rewards,
    values,
    bootstrap,
    truncation,
    termination,
    *,
    discount,
    lambda_,
    normalize,
    native_rewards=None,
):
    """Eager helper; native and bookkeeping stages have separate dtype scopes."""
    with jax.experimental.disable_x64():
        scalar = rewards.sum(axis=0) if native_rewards is None else native_rewards
        total, denominator = _native_advantages(
            scalar,
            values,
            bootstrap,
            truncation,
            termination,
            discount=discount,
            lambda_=lambda_,
            normalize=normalize,
        )
    with jax.experimental.enable_x64():
        parts = _component_advantages(
            rewards,
            total,
            denominator,
            truncation,
            termination,
            discount=discount,
            lambda_=lambda_,
            normalize=normalize,
        )
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


def make_actor_attribution(networks, settings, hip_indices, forward_sign):
    """Cache native-dtype model JITs and a separate deterministic float64 JIT."""
    distribution = networks.parametric_action_distribution
    native_loss = partial(losses.compute_ppo_loss, ppo_network=networks, **settings)
    gae_settings = dict(
        discount=settings["discounting"],
        lambda_=settings["gae_lambda"],
        normalize=settings["normalize_advantage"],
    )

    @jax.jit
    def native_stage(params, normalizer, data, rng):
        data = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), data)
        privileged = (
            data.privileged_observation
            if hasattr(data, "privileged_observation")
            else data.observation
        )
        next_privileged = (
            data.next_privileged_observation
            if hasattr(data, "next_privileged_observation")
            else data.next_observation
        )
        values = networks.value_network.apply(normalizer, params.value, privileged)
        bootstrap = networks.value_network.apply(
            normalizer, params.value, jax.tree.map(lambda x: x[-1], next_privileged)
        )
        extras = data.extras["state_extras"]
        truncation = extras["truncation"]
        termination = (1 - data.discount) * (1 - truncation)
        total, denominator = _native_advantages(
            data.reward, values, bootstrap, truncation, termination, **gae_settings
        )
        policy = partial(networks.policy_network.apply, normalizer)
        logits = policy(params.policy, data.observation)

        def log_prob(output):
            return distribution.log_prob(
                output, data.extras["policy_extras"]["raw_action"]
            )

        old_probability = data.extras["policy_extras"]["log_prob"]
        epsilon = settings["clipping_epsilon"]
        score = jax.grad(lambda output: log_prob(output).sum())(logits)
        weights = jax.grad(
            lambda probability: clipped_component_losses(
                probability, old_probability, jp.ones_like(total)[None], total, epsilon
            ).sum()
        )(log_prob(logits))

        def entropy_loss(output):
            return -settings["entropy_cost"] * distribution.entropy(output, rng).mean()

        def native_head_loss(output):
            ratio = jp.exp(log_prob(output) - old_probability)
            return -jp.mean(
                jp.minimum(
                    ratio * total, jp.clip(ratio, 1 - epsilon, 1 + epsilon) * total
                )
            ) + entropy_loss(output)

        entropy_gradient = jax.grad(entropy_loss)(logits)
        head_gradient = jax.grad(native_head_loss)(logits)
        # Native function expects batch first; preserve its exact loss/RNG path.
        batch_data = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), data)
        (_, metrics), native_gradient = jax.value_and_grad(native_loss, has_aux=True)(
            params, normalizer, batch_data, rng
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
                distribution.mode(policy(p, data.observation))[..., hip_indices]
                * signs
                * 0.25
            )
            return jp.sum(targets * forward[..., None], axis=(0, 1)) / jp.maximum(
                forward.sum(), 1
            )

        target_gradients = jax.jacrev(mean_targets)(params.policy)
        directions = []
        for hip in range(2):
            direction = jax.tree.map(lambda x: -x[hip], target_gradients)
            _, output_direction = jax.jvp(
                lambda p: policy(p, data.observation), (params.policy,), (direction,)
            )
            directions.append(output_direction)
        return dict(
            total=total,
            denominator=denominator,
            truncation=truncation,
            termination=termination,
            score=score,
            weights=weights,
            entropy_gradient=entropy_gradient,
            head_gradient=head_gradient,
            gradient=native_gradient.policy,
            metrics=metrics,
            directions=jp.stack(directions),
            reward_error=jp.max(
                jp.abs(extras["reward_components"].sum(axis=-1) - data.reward)
            ),
        )

    @jax.jit
    def component_stage(rewards, native):
        parts = _component_advantages(
            rewards,
            native["total"],
            native["denominator"],
            native["truncation"],
            native["termination"],
            **gae_settings,
        )
        # Applying these advantages in jacrev would cast each large component
        # gradient to float32 before cancellation. Use the native score/clip
        # derivatives and retain float64 through the sum instead.
        gradients = (
            parts[..., None]
            * native["weights"].astype(jp.float64)[None, ..., None]
            * native["score"].astype(jp.float64)[None]
        )
        gradients = jp.concatenate(
            [gradients, native["entropy_gradient"].astype(jp.float64)[None]]
        )
        output = gradients.sum(axis=0)
        head = native["head_gradient"].astype(jp.float64)
        return dict(
            output=output.astype(native["score"].dtype),
            head_error=jp.linalg.norm(output - head) / (jp.linalg.norm(head) + 1e-8),
            advantage_error=jp.max(
                jp.abs(parts.sum(axis=0) - native["total"].astype(jp.float64))
            ),
            projections=jp.stack(
                [
                    jp.sum(
                        gradients * direction.astype(jp.float64)[None], axis=(1, 2, 3)
                    )
                    for direction in native["directions"]
                ],
                axis=-1,
            ),
            pressure=-gradients[..., hip_indices]
            * jp.asarray(forward_sign, dtype=jp.float64),
        )

    @jax.jit
    def backprop_stage(policy_params, normalizer, observation, output, reference):
        observation = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), observation)
        _, pullback = jax.vjp(
            lambda p: networks.policy_network.apply(normalizer, p, observation),
            policy_params,
        )
        reconstructed = pullback(output)[0]
        difference = jax.tree.map(lambda a, b: a - b, reconstructed, reference)
        return jp.sqrt(tree_dot(difference, difference)) / (
            jp.sqrt(tree_dot(reference, reference)) + 1e-8
        )

    def attribute(params, normalizer, data, rng):
        # Separate compilations keep JAX 0.6's lowering and default entropy
        # sample dtype correct. The x64 scope never includes native model/RNG.
        with jax.experimental.disable_x64():
            native = native_stage(params, normalizer, data, rng)
        with jax.experimental.enable_x64():
            rewards = jp.moveaxis(
                jp.swapaxes(data.extras["state_extras"]["reward_components"], 0, 1),
                -1,
                0,
            )
            components = component_stage(rewards, native)
        with jax.experimental.disable_x64():
            error = backprop_stage(
                params.policy,
                normalizer,
                data.observation,
                components["output"],
                native["gradient"],
            )
            native_head_error = backprop_stage(
                params.policy,
                normalizer,
                data.observation,
                native["head_gradient"],
                native["gradient"],
            )
        return {
            "gradient_relative_error": error,
            "native_head_backprop_relative_error": native_head_error,
            "output_gradient_relative_error": components["head_error"],
            "advantage_max_error": components["advantage_error"],
            "reward_max_error": native["reward_error"],
            "projected_forward_target_sgd": components["projections"],
            "local_forward_mu_pressure": components["pressure"],
            "native_metrics": native["metrics"],
        }

    return attribute
