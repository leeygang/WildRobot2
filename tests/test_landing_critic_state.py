"""Privileged reward memory, historical compatibility and native PPO restore."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
import pytest
import yaml

from wr2.locomotion.configs import load_training_config, training_config_to_dict
from wr2.locomotion.ppo import make_network_factory, prepare_heading_restore
from wr2.locomotion.walking_env import WR2WalkingEnv
from wr2.reference.paper_walk import PaperWalkReference


CONFIGS = Path(__file__).resolve().parents[1] / "wr2/locomotion/configs"


def test_critic_trial_only_changes_information_and_round_trips(tmp_path):
    control = load_training_config(CONFIGS / "ppo_walking_size_continue.yaml")
    trial = load_training_config(CONFIGS / "ppo_walking_landing_critic.yaml")
    assert not control.environment.critic_landing_state
    assert trial == replace(
        control, version=trial.version, version_name=trial.version_name, output=trial.output,
        environment=replace(control.environment, critic_landing_state=True),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(training_config_to_dict(trial)))
    assert load_training_config(path) == trial


@pytest.mark.parametrize("bad", [0, "true"])
def test_critic_flag_rejects_non_boolean(bad, tmp_path):
    data = training_config_to_dict(load_training_config(CONFIGS / "ppo_walking_landing_critic.yaml"))
    data["environment"]["critic_landing_state"] = bad
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="critic_landing_state"):
        load_training_config(path)


def test_critic_state_requires_qualified_landing(tmp_path):
    data = training_config_to_dict(load_training_config(CONFIGS / "ppo_walking_landing_critic.yaml"))
    data["environment"]["air_time_min_stance_s"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="requires qualified paper air time"):
        load_training_config(path)


@pytest.fixture(scope="module")
def paired_envs():
    cfg = load_training_config(CONFIGS / "ppo_walking_landing_critic.yaml")

    def reference(model, **kwargs):
        s = kwargs["normalization_scale"]
        return PaperWalkReference(model, **kwargs, command_vectors=[
            [0., 0., 0.], [.1 * s, 0., 0.],
        ])

    nominal = replace(
        cfg.environment, zero_command_probability=1., turn_command_probability=0.,
        randomization=replace(cfg.environment.randomization, enabled=False),
    )
    with patch("wr2.locomotion.walking_env.PaperWalkReference", reference):
        old = WR2WalkingEnv(replace(nominal, critic_landing_state=False),
                            add_observation_noise=False, reward_discounting=cfg.ppo.discounting)
        new = WR2WalkingEnv(nominal, add_observation_noise=False, reward_discounting=cfg.ppo.discounting)
    return old, new


def holding_action(env):
    return (env._home_ctrl[env._active_indices] - env._action_center[env._active_indices]) / .25


def assert_features(state, env):
    air = state.info["step_air_time"]
    observed = np.asarray(state.obs["privileged_state"])[..., -6:]
    np.testing.assert_allclose(observed[..., :2], np.asarray(air["supported_air_time"]) / env.config.normalization_scale)
    np.testing.assert_allclose(observed[..., 2:4], np.asarray(air["pending_air_time"]) / env.config.normalization_scale)
    np.testing.assert_allclose(observed[..., 4:], np.minimum(np.asarray(air["stance_steps"]), 10) / 10)


def test_actor_physics_rewards_and_old_critic_history_are_identical(paired_envs):
    old, new = paired_envs
    command = jp.asarray([.1 * new.config.normalization_scale, 0., 0.])
    states = [jax.jit(e.reset)(jax.random.PRNGKey(0), command) for e in (old, new)]
    assert new.observation_size == {"state": (1005,), "privileged_state": (1521,)}
    np.testing.assert_array_equal(states[0].obs["state"], states[1].obs["state"])
    np.testing.assert_array_equal(states[0].obs["privileged_state"], states[1].obs["privileged_state"][:-6])
    np.testing.assert_array_equal(states[1].obs["privileged_state"][-6:], 0)
    hold = holding_action(new)
    states = [s.replace(info={**s.info, "previous_action": hold}) for s in states]
    steps = [jax.jit(e.step) for e in (old, new)]
    for tick in range(24):
        action = hold + .01 * jp.sin(jp.arange(12) + tick * .1)
        states = [step(s, action) for step, s in zip(steps, states, strict=True)]
        a, b = states
        np.testing.assert_array_equal(a.pipeline_state.q, b.pipeline_state.q)
        np.testing.assert_array_equal(a.pipeline_state.qd, b.pipeline_state.qd)
        np.testing.assert_array_equal(a.reward, b.reward)
        np.testing.assert_array_equal(a.obs["state"], b.obs["state"])
        np.testing.assert_array_equal(a.obs["privileged_state"], b.obs["privileged_state"][:-6])
        for name in a.metrics:
            np.testing.assert_array_equal(a.metrics[name], b.metrics[name])
        assert_features(b, new)


def test_pending_credit_is_visible_and_next_state_marks_payment(paired_envs):
    _, env = paired_envs
    state = jax.jit(env.reset)(jax.random.PRNGKey(1), jp.asarray([.1 * env.config.normalization_scale, 0., 0.]))
    hold = holding_action(env)
    after = []
    before = []
    for credit in (.18, .24):
        air = dict(state.info["step_air_time"])
        air["pending_air_time"] = jp.asarray([credit, 0.])
        air["stance_steps"] = jp.asarray([9, 30])
        s = state.replace(
            info={**state.info, "step_air_time": air, "previous_action": hold},
            obs={**state.obs, "privileged_state": jp.concatenate([
                state.obs["privileged_state"][:-6], env._critic_landing_features(air),
            ])},
        )
        before.append(s)
        after.append(jax.jit(env.step)(s, hold))
    np.testing.assert_array_equal(before[0].obs["state"], before[1].obs["state"])
    np.testing.assert_array_equal(before[0].obs["privileged_state"][:-6], before[1].obs["privileged_state"][:-6])
    assert before[0].obs["privileged_state"][-4] != before[1].obs["privileged_state"][-4]
    np.testing.assert_allclose(before[0].obs["privileged_state"][-2:], [.9, 1.])
    np.testing.assert_array_equal(after[0].pipeline_state.q, after[1].pipeline_state.q)
    assert float(after[1].reward - after[0].reward) == pytest.approx(.06 * 1.3946777882864203 * 500 * .02, abs=1e-6)
    for s in after:
        assert_features(s, env)
        np.testing.assert_array_equal(s.obs["privileged_state"][-4:-2], 0)


def test_stopping_command_cancels_pending_credit_in_next_observation(paired_envs):
    from wr2.locomotion.paper_evaluation import set_command

    _, env = paired_envs
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    air = dict(s.info["step_air_time"])
    air["pending_air_time"] = jp.asarray([.18, .24])
    air["stance_steps"] = jp.asarray([5, 5])
    s = s.replace(info={**s.info, "step_air_time": air, "previous_action": holding_action(env)})
    s = set_command(s, jp.zeros(3), env.config.normalization_scale)
    after = jax.jit(env.step)(s, holding_action(env))
    assert_features(after, env)
    np.testing.assert_array_equal(after.obs["privileged_state"][-4:-2], 0)
    assert float(after.metrics["air_time_credit_s_per_step"]) == 0


def test_timeout_and_fall_clear_current_reward_memory(paired_envs):
    from brax.envs import training

    _, env = paired_envs
    wrapped = training.wrap(env, episode_length=824)
    initial = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 1))
    before = initial.replace(info={**initial.info, "steps": jp.asarray([823.]), "step_count": jp.asarray([823])})
    after = jax.jit(wrapped.step)(before, jp.zeros((1, 12)))
    assert bool(after.done[0]) and bool(after.info["truncation"][0])
    np.testing.assert_array_equal(after.obs["privileged_state"], initial.obs["privileged_state"])
    assert_features(after, env)
    # Force a physical terminal condition while retaining the credited-state API.
    import copy
    falling = copy.copy(env)
    falling.config = replace(env.config, terminate_height_m=.5, terminate_max_height_m=1.)
    s = jax.jit(falling.reset)(jax.random.PRNGKey(1))
    s = jax.jit(falling.step)(s, holding_action(falling))
    assert bool(s.done)
    np.testing.assert_array_equal(s.obs["privileged_state"][-6:], 0)
    assert_features(s, env)


@pytest.mark.parametrize("normalize", [False, True])
def test_critic_restore_preserves_actor_and_value_predictions(paired_envs, tmp_path, normalize):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint

    old, new = paired_envs
    cfg = load_training_config(CONFIGS / "ppo_walking_landing_critic.yaml")
    factory = make_network_factory(cfg.network)
    net = factory(old.observation_size, 12)
    obs = old.reset(jax.random.PRNGKey(0)).obs
    original = (running_statistics.init_state(obs), net.policy_network.init(jax.random.PRNGKey(1)), net.value_network.init(jax.random.PRNGKey(2)))
    checkpoint.save(tmp_path, 1, original, checkpoint.network_config(old.observation_size, 12, normalize, factory))
    restored, report = prepare_heading_restore(tmp_path / "000000000001", new, cfg.network, normalize)
    assert report["kind"] == "zero_extended_critic_landing_state"
    assert report["verification_dtype"] == "float64"
    assert report["maximum_policy_actions_error"] < 1e-10
    assert report["maximum_critic_values_error"] < 1e-10
    for a, b in zip(jax.tree.leaves(original[1]), jax.tree.leaves(restored[1]), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(restored[2]["params"]["hidden_0"]["kernel"][-6:], 0)
    np.testing.assert_array_equal(restored[0].std["privileged_state"][-6:], 1)
    checkpoint.save(tmp_path, 2, restored, checkpoint.network_config(new.observation_size, 12, normalize, factory))
    reloaded, conversion = prepare_heading_restore(tmp_path / "000000000002", new, cfg.network, normalize)
    assert conversion is None
    for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(reloaded), strict=True):
        np.testing.assert_array_equal(a, b)


def test_native_ppo_updates_new_critic_and_saves_metadata(paired_envs, tmp_path):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, train
    from brax.training.agents.ppo.optimizer import LRSchedule

    old, env = paired_envs
    cfg = load_training_config(CONFIGS / "ppo_walking_landing_critic.yaml")
    factory = make_network_factory(cfg.network)
    net = factory(old.observation_size, 12)
    original = (running_statistics.init_state(old.reset(jax.random.PRNGKey(0)).obs),
                net.policy_network.init(jax.random.PRNGKey(1)), net.value_network.init(jax.random.PRNGKey(2)))
    checkpoint.save(tmp_path / "parent", 1, original, checkpoint.network_config(old.observation_size, 12, False, factory))
    restored, _ = prepare_heading_restore(tmp_path / "parent/000000000001", env, cfg.network, False)
    p = cfg.ppo
    seen = []
    _, params, metrics = train.train(
        environment=env, num_timesteps=64, num_envs=4, episode_length=824, num_evals=1, run_evals=False,
        learning_rate=p.learning_rate, learning_rate_schedule=LRSchedule.NONE,
        entropy_cost=p.entropy_cost, discounting=p.discounting, gae_lambda=p.gae_lambda,
        clipping_epsilon=p.clipping_epsilon, max_grad_norm=p.max_grad_norm,
        vf_loss_coefficient=2 * p.value_loss_coef, unroll_length=16,
        batch_size=1, num_minibatches=4, num_updates_per_batch=4,
        normalize_observations=False, normalize_advantage=True, network_factory=factory,
        seed=cfg.seed, restore_params=restored, save_checkpoint_path=str(tmp_path / "trained"),
        policy_params_fn=lambda step, fn, pp: seen.append((int(step), pp)),
    )
    assert [step for step, _ in seen] == [0, 64]
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((params, metrics)))
    for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(seen[0][1]), strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.any(np.asarray(params[2]["params"]["hidden_0"]["kernel"][-6:]) != 0)
    path = tmp_path / "trained/000000000064"
    loaded = checkpoint.load(path)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(a, b)
    metadata = checkpoint.load_config(path)
    assert tuple(metadata.observation_size["privileged_state"]["shape"]) == (1521,)
    policy = checkpoint.load_policy(path)
    action, _ = policy(env.reset(jax.random.PRNGKey(0)).obs, jax.random.PRNGKey(1))
    assert action.shape == (12,) and np.isfinite(np.asarray(action)).all()


@pytest.mark.parametrize("corrupt", [False, True])
def test_restore_distinguishes_width_roundoff_and_wrong_new_weights(tmp_path, monkeypatch, corrupt):
    from types import SimpleNamespace
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint
    from wr2.locomotion import ppo

    cfg = load_training_config(CONFIGS / "ppo_walking_landing_critic.yaml")
    factory = make_network_factory(cfg.network)
    sizes = {"state": 1005, "privileged_state": 1515}
    net = factory(sizes, 12)
    original = (running_statistics.init_state({k: jp.zeros(v) for k, v in sizes.items()}),
                net.policy_network.init(jax.random.PRNGKey(1)), net.value_network.init(jax.random.PRNGKey(2)))
    checkpoint.save(tmp_path, 1, original, checkpoint.network_config(sizes, 12, False, factory))

    def rounded_factory(obs_sizes, *args, **kwargs):
        networks = factory(obs_sizes, *args, **kwargs)
        if obs_sizes["privileged_state"] == 1521:
            apply = networks.value_network.apply

            def rounded_apply(*inputs):
                value = apply(*inputs)
                return value + 1e-4 if value.dtype == jp.float32 else value

            networks = networks.replace(value_network=replace(networks.value_network, apply=rounded_apply))
        return networks

    monkeypatch.setattr(ppo, "make_network_factory", lambda config: rounded_factory)
    if corrupt:
        expand = ppo.expand_landing_state_params

        def bad_expansion(*args):
            params, report = expand(*args)
            kernel = params[2]["params"]["hidden_0"]["kernel"]
            params[2]["params"]["hidden_0"]["kernel"] = kernel.at[-1, 0].set(1e-5)
            return params, report

        monkeypatch.setattr(ppo, "expand_landing_state_params", bad_expansion)
    env = SimpleNamespace(observation_size={"state": (1005,), "privileged_state": (1521,)},
                          action_size=12, _observation_history_frames=15, _critic_landing_state_size=6)
    settings = (jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision, jax.config.jax_default_device)
    if corrupt:
        with pytest.raises(AssertionError, match="Landing-state restore changed critic_values"):
            prepare_heading_restore(tmp_path / "000000000001", env, cfg.network, False)
    else:
        _, report = prepare_heading_restore(tmp_path / "000000000001", env, cfg.network, False)
        assert report["maximum_critic_values_error"] < 1e-10
        assert 9e-5 < report["native_maximum_critic_values_error"] < 2e-4
    assert settings == (jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision, jax.config.jax_default_device)
