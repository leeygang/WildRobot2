"""Check attribution against native Brax, including clipping and termination."""

from functools import partial
from importlib import metadata
import json
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses, networks

from wr2.locomotion.ppo_diagnostics import (
    actor_attribution,
    clipped_component_losses,
    group_rewards,
    split_advantages,
)


def test_grouping_keeps_weighted_reward_units_and_unknown_terms():
    rewards = {
        "lin_vel_xy": jp.array([0.1, 0.2]),
        "feet_air_time": jp.array([1.0, 0.0]),
        "collision": jp.array([-0.01, -0.02]),
        "new_term": jp.array([0.03, 0.04]),
    }
    grouped = group_rewards(rewards)
    np.testing.assert_allclose(grouped.sum(axis=-1), sum(rewards.values()))
    np.testing.assert_allclose(grouped[:, 0], rewards["lin_vel_xy"])
    np.testing.assert_allclose(grouped[:, 4], rewards["feet_air_time"])


@pytest.mark.parametrize("normalize", [False, True])
def test_advantages_reconstruct_with_bootstrap_falls_and_timeouts(normalize):
    rewards = jax.random.normal(jax.random.PRNGKey(0), (6, 7, 4))
    values = jax.random.normal(jax.random.PRNGKey(1), (7, 4))
    bootstrap = jp.arange(4.0)
    truncation = jp.zeros((7, 4)).at[3, 0].set(1).at[-1, 1].set(1)
    termination = jp.zeros((7, 4)).at[4, 2].set(1)
    parts, total = split_advantages(
        rewards,
        values,
        bootstrap,
        truncation,
        termination,
        discount=0.97,
        lambda_=0.95,
        normalize=normalize,
    )
    np.testing.assert_allclose(parts.sum(axis=0), total, atol=2e-6)


def test_clipping_uses_total_advantage_for_opposing_components():
    # Include exact clip boundaries and zero-advantage ties.
    ratios = jp.array([[0.6, 0.8, 1.0, 1.2, 1.4, 1.4]])
    parts = jp.stack([jp.full_like(ratios, 2), jp.array([[-1.0, -3, -1, -3, -1, -2]])])
    total = parts.sum(axis=0)
    old = jp.zeros_like(ratios)

    def native(log_ratio):
        ratio = jp.exp(log_ratio)
        return -jp.mean(jp.minimum(ratio * total, jp.clip(ratio, 0.8, 1.2) * total))

    def attributed(log_ratio):
        return clipped_component_losses(log_ratio, old, parts, total, 0.2).sum()

    np.testing.assert_allclose(
        attributed(jp.log(ratios)), native(jp.log(ratios)), atol=1e-6
    )
    np.testing.assert_allclose(
        jax.grad(attributed)(jp.log(ratios)),
        jax.grad(native)(jp.log(ratios)),
        atol=1e-7,
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_full_actor_gradient_matches_native_ppo(normalize):
    # The same test can run under the TB fork's separate privileged-observation API.
    tb = "privileged_observation" in types.Transition._fields
    net_kwargs = dict(policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,))
    if tb:
        nets = networks.make_ppo_networks(4, 6, 2, **net_kwargs)
    else:
        nets = networks.make_ppo_networks(
            {"state": 4, "privileged_state": 6},
            2,
            policy_obs_key="state",
            value_obs_key="privileged_state",
            distribution_type="tanh_normal",
            **net_kwargs,
        )
    params = losses.PPONetworkParams(
        policy=nets.policy_network.init(jax.random.PRNGKey(0)),
        value=nets.value_network.init(jax.random.PRNGKey(1)),
    )
    obs = jax.random.normal(jax.random.PRNGKey(2), (4, 7, 4))
    privileged = jax.random.normal(jax.random.PRNGKey(3), (4, 7, 6))
    observation = obs if tb else {"state": obs, "privileged_state": privileged}
    normalizer = running_statistics.init_state(
        jp.zeros(4) if tb else {"state": jp.zeros(4), "privileged_state": jp.zeros(6)}
    )
    logits = nets.policy_network.apply(normalizer, params.policy, observation)
    dist = nets.parametric_action_distribution
    raw = dist.sample_no_postprocessing(logits, jax.random.PRNGKey(4))
    # Deliberately include clipped and unclipped samples.
    old_prob = dist.log_prob(logits, raw) - jp.log(
        jp.linspace(0.6, 1.4, 28).reshape(4, 7)
    )
    components = jax.random.normal(jax.random.PRNGKey(5), (4, 7, 6)) * 0.1
    extras = {
        "state_extras": {
            "truncation": jp.zeros((4, 7)).at[0, -1].set(1),
            "reward_components": components,
            "trace": {
                "command_tb": jp.broadcast_to(jp.array([0.1, 0.0, 0.0]), (4, 7, 3))
            },
        },
        "policy_extras": {
            "raw_action": raw,
            "log_prob": old_prob,
            "distribution_params": logits,
        },
    }
    data_kwargs = dict(
        observation=observation,
        next_observation=observation,
        action=dist.postprocess(raw),
        reward=components.sum(axis=-1),
        discount=jp.ones((4, 7)).at[1, 3].set(0),
        extras=extras,
    )
    if tb:
        data_kwargs.update(
            privileged_observation=privileged, next_privileged_observation=privileged
        )
    data = types.Transition(**data_kwargs)
    settings = dict(
        entropy_cost=0.0005,
        discounting=0.97,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        normalize_advantage=normalize,
    )
    result = jax.jit(
        partial(
            actor_attribution,
            networks=nets,
            settings=settings,
            hip_indices=(0, 1),
            forward_sign=(-1, 1),
        )
    )(params, normalizer, data, jax.random.PRNGKey(6))
    assert float(result["gradient_relative_error"]) < 1e-5
    assert float(result["advantage_max_error"]) < 1e-5
    assert float(result["reward_max_error"]) < 1e-6
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result))


@pytest.mark.skipif(
    "privileged_observation" in types.Transition._fields,
    reason="WR2 driver uses modern Brax",
)
@pytest.mark.parametrize("randomized", [False, True])
def test_driver_checks_the_same_minibatch_used_by_native_adam(
    tmp_path, monkeypatch, randomized
):
    from brax.envs.base import Env, State
    from flax import struct
    from wr2.tools import diagnose_ppo as driver

    @struct.dataclass
    class Coefficients:
        scale: object

    class EnvWithRewards(Env):
        config = SimpleNamespace(episode_length=20)
        dt = 0.02
        action_size = 2
        backend = "generalized"
        observation_size = {"state": 4, "privileged_state": 6}
        sys = Coefficients(jp.array(1.0))

        def reset(self, key):
            return State(
                None,
                {"state": jp.zeros(4), "privileged_state": jp.zeros(6)},
                jp.zeros(()),
                jp.zeros(()),
                {"leg_motor_pos": jp.zeros(())},
                {"tick": jp.zeros((), dtype=jp.int32)},
            )

        def step(self, state, action):
            reward = -jp.mean((action - 0.2) ** 2) * self.sys.scale
            return state.replace(
                obs={
                    name: value.at[0].set(action.mean())
                    for name, value in state.obs.items()
                },
                reward=reward,
                metrics={"leg_motor_pos": reward},
                info={**state.info, "tick": state.info["tick"] + 1},
            )

    env = EnvWithRewards()
    factory = partial(
        networks.make_ppo_networks,
        policy_hidden_layer_sizes=(8,),
        value_hidden_layer_sizes=(8,),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="tanh_normal",
    )
    nets = factory({"state": 4, "privileged_state": 6}, action_size=2)
    saved = (
        running_statistics.init_state(env.reset(None).obs),
        nets.policy_network.init(jax.random.PRNGKey(0)),
        nets.value_network.init(jax.random.PRNGKey(1)),
    )

    def trace(before, after, command, contact, step):
        return {
            "command_tb": command,
            "phase": jp.mod(step / 30, 1),
            "velocity_tb": jp.zeros(2),
            "yaw_rate_tb": jp.zeros(()),
            "landed": jp.zeros(2, dtype=bool),
            "contact": contact,
            "landing_lead_tb_m": jp.zeros(2),
            "actual_hip_pitch_rad": jp.zeros(2),
            "reference_hip_pitch_rad": jp.zeros(2),
        }

    context = SimpleNamespace(
        env=env,
        hips=(0, 1),
        signs=(-1, 1),
        home=np.zeros(2),
        scale=1.0,
        factory=factory,
        trace=trace,
        randomizer=(lambda system, rng: (Coefficients(jp.ones(4)), Coefficients(0)))
        if randomized
        else None,
        reward=lambda state: state.metrics,
        command=lambda state: jp.array([0.1, 0.0, 0.0]),
        contact=lambda state: jp.ones(2, dtype=bool),
        step=lambda state: state.info["tick"],
        settings=dict(
            entropy_cost=0.0005,
            discounting=0.97,
            gae_lambda=0.95,
            clipping_epsilon=0.2,
            normalize_advantage=True,
            vf_coefficient=0.5,
        ),
        learning_rate=1e-4,
        epochs=4,
        minibatches=4,
        unroll=20,
        batch_size=1,
        loader=lambda step, networks: saved,
        checkpoint_steps=[80],
        source={},
    )
    monkeypatch.setattr(driver, "load_wr2", lambda run_dir, count: context)
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "xla_flags": "",
                    "jax_default_matmul_precision": None,
                    "packages": {
                        name: metadata.version(name)
                        for name in (
                            "brax",
                            "jax",
                            "jaxlib",
                            "mujoco",
                            "mujoco-mjx",
                            "flax",
                            "optax",
                        )
                    },
                }
            }
        )
    )
    output = tmp_path / "diagnostic"
    driver.run(
        SimpleNamespace(
            run_dir=tmp_path,
            output=output,
            tb_root=None,
            allow_cpu=True,
            seed=22103,
            num_envs=4,
            warmup_steps=8,
            batches=1,
            checkpoint_steps=None,
        )
    )
    result = json.loads((output / "summary.json").read_text())
    assert result["verified_batches"] == 1
    row = result["results"][0]
    assert row["gradient_relative_error"] < 5e-5
    before = row["samples"]["all"]["hip_forward_target_mean_deg"]
    after = row["requests_after_fresh_adam_on_same_observations"]["all"][
        "hip_forward_target_mean_deg"
    ]
    assert not np.array_equal(before, after)
    assert not any("params" in path.name for path in output.iterdir())
