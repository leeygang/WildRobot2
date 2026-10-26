"""Shared construction of the WR2 Brax PPO networks."""

from __future__ import annotations

from functools import partial
import copy

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


def expand_heading_params(params, observation_size, history_frames):
    """Append zero-weight sine/cosine inputs to every Paper history frame."""
    import jax.numpy as jp
    import numpy as np

    # Brax environments report flat vectors as either sizes or one-element shapes.
    observation_size = {key: int(np.prod(size)) for key, size in observation_size.items()}

    original_sizes = {
        key: params[index]["params"]["hidden_0"]["kernel"].shape[0]
        for key, index in (("state", 1), ("privileged_state", 2))
    }
    if original_sizes == observation_size:
        return params, None
    if any(observation_size[key] - size != 2 * history_frames
           for key, size in original_sizes.items()):
        raise ValueError("Only two appended heading features per history frame can be restored")

    def extend(value, key, fill=0):
        size = original_sizes[key]
        if value.shape[0] != size or size % history_frames:
            raise ValueError(f"Invalid saved {key} input shape: {value.shape}")
        blocks = jp.asarray(value).reshape((history_frames, size // history_frames, *value.shape[1:]))
        extra = jp.full((history_frames, 2, *value.shape[1:]), fill, dtype=value.dtype)
        return jp.concatenate([blocks, extra], axis=1).reshape((observation_size[key], *value.shape[1:]))

    normalizer = params[0].replace(
        mean={key: extend(value, key) for key, value in params[0].mean.items()},
        summed_variance={key: extend(value, key) for key, value in params[0].summed_variance.items()},
        std={key: extend(value, key, 1) for key, value in params[0].std.items()},
    )
    policy, value = copy.deepcopy(params[1:])
    for network, key in ((policy, "state"), (value, "privileged_state")):
        network["params"]["hidden_0"]["kernel"] = extend(
            network["params"]["hidden_0"]["kernel"], key,
        )
    report = {
        "kind": "zero_extended_heading_error_inputs",
        "history_frames": history_frames,
        "original_input_sizes": original_sizes,
        "restored_input_sizes": dict(observation_size),
        "new_input_weights": "zeros", "new_normalizer_std": 1,
    }
    return (normalizer, policy, value), report


def prepare_heading_restore(path, environment, network_config, normalize_observations):
    """Load/expand a Paper checkpoint and verify unchanged initial predictions."""
    import hashlib
    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint

    original = checkpoint.load(str(path.resolve()))
    if any(not np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(original)):
        raise ValueError("Restore checkpoint contains nonfinite parameters")
    restored, report = expand_heading_params(
        original, environment.observation_size, environment._observation_history_frames,
    )
    if report is None:
        return restored, None
    factory = make_network_factory(network_config)
    preprocess = running_statistics.normalize if normalize_observations else types.identity_observation_preprocessor
    old = factory(report["original_input_sizes"], environment.action_size,
                  preprocess_observations_fn=preprocess)
    new = factory(report["restored_input_sizes"], environment.action_size,
                  preprocess_observations_fn=preprocess)
    rng = np.random.default_rng(0)
    before, after = {}, {}
    frames = environment._observation_history_frames
    for key, size in report["original_input_sizes"].items():
        before[key] = jp.asarray(rng.normal(0, .1, (16, size)), dtype=jp.float32)
        angle = jp.asarray(rng.uniform(-np.pi, np.pi, (16, frames)), dtype=jp.float32)
        feedback = jp.stack([jp.sin(angle), jp.cos(angle)], axis=-1)
        after[key] = jp.concatenate([
            before[key].reshape(16, frames, size // frames), feedback,
        ], axis=2).reshape(16, report["restored_input_sizes"][key])
    def predictions(networks, params, observations):
        logits = networks.policy_network.apply(*params[:2], observations)
        return {
            "policy_logits": np.asarray(logits),
            "policy_actions": np.asarray(networks.parametric_action_distribution.mode(logits)),
            "critic_values": np.asarray(networks.value_network.apply(params[0], params[2], observations)),
        }

    # Record native arithmetic, but prove the mapping on CPU in float64.
    # Adding zero-weight columns changes GPU GEMM shapes/rounding, even at
    # "high" precision. This eager verification scope does not affect PPO.
    native_before = predictions(old, original, before)
    native_after = predictions(new, restored, after)
    report["native_backend"] = jax.default_backend()
    report["native_matmul_precision"] = jax.config.jax_default_matmul_precision
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu), jax.experimental.enable_x64(), jax.default_matmul_precision("highest"):
        cpu_original, cpu_restored, cpu_before, cpu_after = jax.device_put(
            (original, restored, before, after), cpu,
        )
        cpu_before = jax.tree.map(lambda value: jp.asarray(value, dtype=jp.float64), cpu_before)
        cpu_after = jax.tree.map(lambda value: jp.asarray(value, dtype=jp.float64), cpu_after)
        verified_before = predictions(old, cpu_original, cpu_before)
        verified_after = predictions(new, cpu_restored, cpu_after)
    report["verification_backend"] = "cpu"
    report["verification_dtype"] = "float64"
    report["verification_matmul_precision"] = "highest"
    for name, expected in verified_before.items():
        actual = verified_after[name]
        if expected.dtype != np.float64 or actual.dtype != np.float64:
            raise ValueError(f"Heading restore verification requires float64 {name}")
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                                   err_msg=f"Heading restore changed {name}")
        report[f"maximum_{name}_error"] = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        if not np.isfinite(native_before[name]).all() or not np.isfinite(native_after[name]).all():
            raise ValueError(f"Heading restore produced nonfinite native {name}")
        report[f"native_maximum_{name}_error"] = float(np.max(np.abs(native_after[name] - native_before[name])))
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(original):
        array = np.asarray(leaf)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    report["source_parameter_sha256"] = digest.hexdigest()
    report["source_checkpoint"] = str(path.resolve())
    return restored, report
