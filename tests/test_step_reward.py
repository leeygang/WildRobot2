"""Qualified landing credit, PPO timing, and real environment reset checks."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import copy

import jax
import jax.numpy as jp
import numpy as np
import pytest
import yaml

from wr2.locomotion.configs import load_training_config, training_config_to_dict
from wr2.locomotion.paper_rewards import qualified_air_time_step
from wr2.locomotion.walking_env import WR2WalkingEnv
from wr2.reference.paper_walk import PaperWalkReference

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "wr2/locomotion/configs/ppo_walking_paper.yaml"
EXPERIMENT = ROOT / "wr2/locomotion/configs/ppo_walking_step.yaml"


def empty_tracker():
    return {
        "supported_air_time": jp.zeros(2),
        "pending_air_time": jp.zeros(2),
        "stance_steps": jp.zeros(2, dtype=jp.int32),
    }


def scan_contacts(contacts, minimum=3, gamma=0.97, active=None):
    contacts = jp.asarray(contacts, dtype=bool)
    active = jp.ones(len(contacts), bool) if active is None else jp.asarray(active)
    compensation = gamma ** (-(minimum - 1))

    def tick(state, inputs):
        contact, enabled = inputs
        return qualified_air_time_step(
            state,
            contact,
            enabled,
            dt=0.02,
            minimum_stance_steps=minimum,
            discount_compensation=compensation,
        )

    return jax.jit(lambda: jax.lax.scan(tick, empty_tracker(), (contacts, active)))()


def test_short_contacts_cancel_and_do_not_accumulate_stance_across_gaps():
    # Right swings for .10 s, but two contacts each last only two samples.
    contact = (
        [[True, False]] * 5 + [[True, True]] * 2 + [[True, False]] + [[True, True]] * 2
    )
    state, metrics = scan_contacts(contact)
    assert float(metrics["air_time_credit_s_per_step"].sum()) == 0
    assert float(metrics["candidate_air_time_s_per_step"].sum()) == pytest.approx(0.12)
    assert float(metrics["rejected_air_time_s_per_step"].sum()) == pytest.approx(0.10)
    assert float(state["pending_air_time"][1]) == pytest.approx(0.02)
    assert int(state["stance_steps"][1]) == 2


@pytest.mark.parametrize("gamma", [0.97, 0.9, 1.0])
def test_qualifying_landing_pays_once_and_preserves_discounted_landing_value(gamma):
    contacts = [[True, False]] * 5 + [[True, True]] * 6
    state, metrics = scan_contacts(contacts, gamma=gamma)
    paid = np.asarray(metrics["air_time_credit_s_per_step"])
    np.testing.assert_array_equal(np.flatnonzero(paid), [7])
    assert float(metrics["right_foot_stance_duration_s_per_step"][5]) == 0.0
    assert float(metrics["right_foot_stance_duration_s_per_step"][7]) == pytest.approx(
        0.04
    )
    assert float(metrics["qualified_air_time_s_per_step"].sum()) == pytest.approx(0.10)
    # Landing is at index 5; qualification/payment is two ticks later.
    assert float(np.dot(paid, gamma ** np.arange(len(paid)))) == pytest.approx(
        0.10 * gamma**5
    )
    assert float(state["pending_air_time"].sum()) == 0


def test_flight_does_not_fund_either_foot():
    state, metrics = scan_contacts([[False, False]] * 8 + [[True, True]] * 4)
    assert float(metrics["qualified_air_time_s_per_step"].sum()) == 0
    assert float(state["supported_air_time"].sum()) == 0


def test_only_opposite_foot_supported_part_of_swing_is_credited():
    contacts = [[True, False]] * 4 + [[False, False]] * 3 + [[True, True]] * 3
    _, metrics = scan_contacts(contacts)
    assert float(metrics["qualified_air_time_s_per_step"].sum()) == pytest.approx(0.08)


def test_stop_cancels_pending_credit_and_restart_cannot_pay_it():
    contacts = [[True, False]] * 5 + [[True, True]] * 5
    active = [True] * 6 + [False] + [True] * 3
    state, metrics = scan_contacts(contacts, active=active)
    assert float(metrics["air_time_credit_s_per_step"].sum()) == 0
    assert float(metrics["rejected_air_time_s_per_step"].sum()) == pytest.approx(0.10)
    assert float(state["pending_air_time"].sum()) == 0


def test_inactive_landing_cannot_earn_credit_after_command_restart():
    contacts = [[True, False]] * 5 + [[True, True]] * 5
    _, metrics = scan_contacts(contacts, active=[False] * 6 + [True] * 4)
    assert float(metrics["air_time_credit_s_per_step"].sum()) == 0


def test_experiment_changes_only_reward_treatment_and_run_metadata():
    baseline = training_config_to_dict(load_training_config(BASELINE))
    experiment = training_config_to_dict(load_training_config(EXPERIMENT))
    expected = copy.deepcopy(baseline)
    expected["version"] = "0.28.8"
    expected["version_name"] = "WR2 ToddlerBot paper qualified step air reward"
    expected["environment"]["air_time_min_stance_s"] = (
        0.20 * expected["environment"]["normalization_scale"]
    )
    expected["output"]["run_prefix"] = "wr2_paper_step"
    assert experiment == expected
    assert experiment["reward_weights"]["feet_air_time"] == 500
    assert experiment["ppo"]["num_timesteps"] == 300_000_000
    assert experiment["checkpoints"]["save_every_evaluation"]


def test_invalid_minimum_stance_is_rejected(tmp_path):
    config = yaml.safe_load(EXPERIMENT.read_text())
    config["environment"]["air_time_min_stance_s"] = -0.1
    path = tmp_path / "negative.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="air_time_min_stance_s"):
        load_training_config(path)


@pytest.fixture(scope="module")
def step_env():
    config = load_training_config(EXPERIMENT)
    environment = replace(
        config.environment,
        randomization=replace(config.environment.randomization, enabled=False),
    )

    def reference(model, **kwargs):
        s = kwargs["normalization_scale"]
        return PaperWalkReference(
            model, **kwargs, command_vectors=[[0.0, 0.0, 0.0], [0.1 * s, 0.0, 0.0]]
        )

    with patch("wr2.locomotion.walking_env.PaperWalkReference", reference):
        return WR2WalkingEnv(
            environment,
            add_observation_noise=False,
            reward_discounting=config.ppo.discounting,
        )


def test_real_environment_timeout_clears_pending_and_supported_credit(step_env):
    env = step_env
    assert env._air_time_min_stance_steps == 10
    assert env._air_time_discount_compensation == pytest.approx(0.97**-9)
    key = jax.random.PRNGKey(7)
    command = jp.asarray([0.1 * env.config.normalization_scale, 0.0, 0.0])
    reset = jax.jit(env.reset)
    state = reset(key, command)
    info = dict(state.info)
    info["step_count"] = jp.asarray(env.config.episode_length - 1, dtype=jp.int32)
    info["step_air_time"] = {
        "supported_air_time": jp.asarray([0.12, 0.14]),
        "pending_air_time": jp.asarray([0.10, 0.20]),
        "stance_steps": jp.asarray([1, 1], dtype=jp.int32),
    }
    result = jax.jit(env.step)(state.replace(info=info), env.home_action)
    assert int(result.info["step_count"]) == 0
    for value in result.info["step_air_time"].values():
        np.testing.assert_array_equal(value, [0, 0])
    assert np.isfinite(np.asarray(result.reward))
    assert result.obs["state"].shape == (975,)
    assert result.obs["privileged_state"].shape == (1485,)
    weighted = sum(
        float(v) for k, v in result.metrics.items() if k.startswith("paper_reward/")
    )
    assert float(result.reward) == pytest.approx(weighted, abs=1e-5)


def test_reward_change_preserves_physics_observations_and_other_terms(step_env):
    qualified = step_env
    baseline = copy.copy(qualified)
    baseline.config = replace(qualified.config, air_time_min_stance_s=0.0)
    baseline._air_time_min_stance_steps = 0
    baseline._air_time_discount_compensation = 1.0
    key = jax.random.PRNGKey(17)
    cmd = jp.asarray([0.1 * qualified.config.normalization_scale, 0.0, 0.0])
    a = jax.jit(qualified.reset)(key, cmd)
    b = jax.jit(baseline.reset)(key, cmd)
    action = jp.linspace(-0.05, 0.05, qualified.action_size)
    a = jax.jit(qualified.step)(a, action)
    b = jax.jit(baseline.step)(b, action)
    for x, y in zip(
        jax.tree.leaves(a.pipeline_state),
        jax.tree.leaves(b.pipeline_state),
        strict=True,
    ):
        np.testing.assert_allclose(x, y, atol=1e-6)
    for name in a.obs:
        np.testing.assert_allclose(a.obs[name], b.obs[name], atol=1e-6)
    for name in b.metrics:
        if name.startswith("paper_reward/") and name != "paper_reward/feet_air_time":
            assert float(a.metrics[name]) == pytest.approx(
                float(b.metrics[name]), abs=1e-6
            )
    assert "step_air_time" not in b.info


def test_native_brax_update_with_qualified_reward_and_timeout_reset(step_env):
    from brax.training.agents.ppo import train
    from brax.training.agents.ppo.optimizer import LRSchedule
    from wr2.locomotion.ppo import make_network_factory

    config = load_training_config(EXPERIMENT)
    env = copy.copy(step_env)
    env.config = replace(env.config, episode_length=20)
    seen = []
    _, params, _ = train.train(
        environment=env,
        num_timesteps=80,
        num_envs=4,
        episode_length=20,
        num_evals=1,
        num_eval_envs=1,
        unroll_length=20,
        batch_size=4,
        num_minibatches=1,
        num_updates_per_batch=1,
        learning_rate=config.ppo.learning_rate,
        learning_rate_schedule=LRSchedule.NONE,
        max_grad_norm=None,
        entropy_cost=config.ppo.entropy_cost,
        discounting=config.ppo.discounting,
        normalize_observations=False,
        normalize_advantage=True,
        gae_lambda=config.ppo.gae_lambda,
        vf_loss_coefficient=2 * config.ppo.value_loss_coef,
        clipping_epsilon=config.ppo.clipping_epsilon,
        network_factory=make_network_factory(config.network),
        seed=config.seed,
        training_metrics_steps=80,
        progress_fn=lambda step, metrics: seen.append((step, metrics)),
    )
    assert seen
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))
    final = seen[-1][1]
    assert np.isfinite(final["eval/episode_reward"])
    for name in ["qualified_air_time_s_per_step", "rejected_air_time_s_per_step"]:
        assert np.isfinite(final["eval/episode_" + name])
