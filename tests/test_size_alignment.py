"""Size-aligned learning clocks, support classification and native PPO gates."""

import copy
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
import pytest
import yaml

from wr2.locomotion.configs import load_training_config, training_config_to_dict
from wr2.locomotion.paper_evaluation import PaperEvaluator
from wr2.locomotion.paper_rewards import qualified_air_time_step
from wr2.locomotion.ppo import make_network_factory
from wr2.locomotion.walking_env import WR2WalkingEnv
from wr2.reference.paper_walk import PaperWalkReference


CONFIGS = Path(__file__).resolve().parents[1] / "wr2/locomotion/configs"


def test_complete_configs_isolate_size_recipe_and_round_trip(tmp_path):
    parent = load_training_config(CONFIGS / "ppo_walking_leg_imitation15.yaml")
    control = load_training_config(CONFIGS / "ppo_walking_size_control.yaml")
    aligned = load_training_config(CONFIGS / "ppo_walking_size_aligned.yaml")
    assert control == replace(
        parent, version=control.version, version_name=control.version_name,
        output=control.output,
    )
    assert aligned == replace(
        control, version_name=aligned.version_name, output=aligned.output,
        environment=replace(
            control.environment, episode_length=824,
            contact_force_threshold_n=0.6129301508068508,
        ),
        ppo=replace(
            control.ppo, unroll_length=16, discounting=0.9637121761767793,
            gae_lambda=0.9396526367723493,
        ),
    )
    for config in (control, aligned):
        path = tmp_path / f"{config.output.run_prefix}.yaml"
        path.write_text(yaml.safe_dump(training_config_to_dict(config)))
        assert load_training_config(path) == config
        assert config.ppo.num_timesteps == 200_000_000
        assert config.checkpoints.save_every_evaluation
        assert config.ppo.num_evals == 16


def test_learning_horizons_preserve_tb_time_constants():
    cfg = load_training_config(CONFIGS / "ppo_walking_size_aligned.yaml")
    s, dt = cfg.environment.normalization_scale, 0.02
    # Frozen paper-walking-baseline: 20 s episode, .4 s rollout,
    # gamma .97; native Brax's GAE default is .95.
    assert abs(cfg.environment.episode_length * dt / s - 20) <= dt / (2 * s)
    assert abs(cfg.ppo.unroll_length * dt / s - 0.4) <= dt / (2 * s)
    for corrected, tb in ((cfg.ppo.discounting, .97), (cfg.ppo.gae_lambda, .95)):
        assert -dt / (s * math.log(corrected)) == pytest.approx(-dt / math.log(tb))
    assert -dt / (s * math.log(cfg.ppo.discounting * cfg.ppo.gae_lambda)) == (
        pytest.approx(-dt / math.log(.97 * .95))
    )


@pytest.fixture(scope="module")
def aligned_env():
    cfg = load_training_config(CONFIGS / "ppo_walking_size_aligned.yaml")

    def reference(model, **kwargs):
        s = kwargs["normalization_scale"]
        # Native reference/IK, reduced exact grid only for CPU integration gates.
        return PaperWalkReference(model, **kwargs, command_vectors=[
            [0., 0., 0.], [.1 * s, 0., 0.],
        ])

    nominal = replace(
        cfg.environment, zero_command_probability=1., turn_command_probability=0.,
        randomization=replace(cfg.environment.randomization, enabled=False),
    )
    with patch("wr2.locomotion.walking_env.PaperWalkReference", reference):
        return WR2WalkingEnv(
            nominal, add_observation_noise=False, reward_discounting=cfg.ppo.discounting,
        )


def test_nominal_body_weight_threshold_uses_summed_solver_force(aligned_env):
    env = aligned_env
    wr_mass = env.sys.mj_model.body_mass.sum()
    tb_mass = 3.358176486  # Frozen TB model, audited native mass sum.
    assert env.config.contact_force_threshold_n / wr_mass == pytest.approx(1 / tb_mass)
    pipeline = SimpleNamespace(contact=SimpleNamespace(
        geom1=jp.full(4, env._floor_geom_id),
        geom2=jp.asarray([env._foot_geom_ids[0]] * 2 + [env._foot_geom_ids[1]] * 2),
        dist=jp.full(4, 0.001),
    ))
    forces = jp.zeros((4, 6)).at[:, 2].set(jp.asarray([.4, .4, .3, .3]))
    np.testing.assert_array_equal(env._foot_contact(pipeline, forces), [True, False])
    np.testing.assert_allclose(env._foot_vertical_forces(pipeline, forces), [.8, .6])


def test_landing_payment_remains_once_and_discount_equivalent(aligned_env):
    env = aligned_env
    gamma = load_training_config(CONFIGS / "ppo_walking_size_aligned.yaml").ppo.discounting
    assert env._air_time_min_stance_steps == 10
    assert env._air_time_discount_compensation == pytest.approx(1.3946777882864203)
    state = {
        "supported_air_time": jp.asarray([.2, 0.]),
        "pending_air_time": jp.zeros(2), "stance_steps": jp.asarray([0, 20]),
    }
    credit = []
    for _ in range(12):
        state, metrics = qualified_air_time_step(
            state, jp.asarray([True, True]), True, dt=env.dt,
            minimum_stance_steps=env._air_time_min_stance_steps,
            discount_compensation=env._air_time_discount_compensation,
        )
        credit.append(float(metrics["air_time_credit_s_per_step"]))
    assert np.count_nonzero(credit) == 1
    assert sum(gamma ** i * value for i, value in enumerate(credit)) == pytest.approx(.2)


def test_824_tick_timeout_preserves_delivered_target_and_native_autoreset(aligned_env):
    from brax.envs import training
    from wr2.tools.diagnose_walking import target_trace

    env = aligned_env
    wrapped = training.wrap(env, episode_length=824)
    initial = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 1))
    before = initial.replace(info={
        **initial.info, "steps": jp.asarray([823.]), "step_count": jp.asarray([823]),
        "previous_action": jp.full((1, 12), .2),
    })
    action = jp.zeros((1, 12))
    _, target, _, _, _ = jax.vmap(lambda s, a: target_trace(env, s, a))(before, action)
    after = jax.jit(wrapped.step)(before, action)
    assert bool(after.done[0]) and bool(after.info["truncation"][0])
    assert not bool(after.metrics["fall"][0])
    assert int(after.info["step_count"][0]) == 0
    np.testing.assert_array_equal(after.pipeline_state.q, initial.pipeline_state.q)
    np.testing.assert_array_equal(after.obs["state"], initial.obs["state"])
    np.testing.assert_array_equal(after.info["previous_target"], initial.info["reset_target"])
    np.testing.assert_array_equal(after.info["last_applied_target"], target)
    # The old diagnostic's reset history would give a false mismatch here.
    assert np.max(abs(target - after.info["previous_target"])) > .01


def test_native_diagnostic_keeps_all_824_finite_timeout_samples(aligned_env):
    from wr2.tools.validate_hip_action import rollout

    env = copy.copy(aligned_env)
    env.action_center = env._action_center  # ProbeEnv's public action-center metadata.
    evaluator = PaperEvaluator(env, None, 1, pushes=False)
    keys = jax.random.split(jax.random.PRNGKey(1), 1)

    def zero_policy(obs, key):
        return jp.zeros((1, 12)), {}

    trace = jax.jit(lambda: rollout(
        env, evaluator.env, zero_policy, keys, jp.zeros(3), 2, 824,
    ))()
    assert trace["valid"].sum() == 824
    assert not trace["fall"].any() and not trace["nonfinite"].any()
    np.testing.assert_array_equal(trace["target_error_rad"], np.zeros((824, 1)))
    assert env.observation_size == {"state": (1005,), "privileged_state": (1515,)}
    assert env._zmp_reference._lookup_length == 1000


def test_native_16_tick_ppo_updates_and_saves_loadable_parameters(aligned_env, tmp_path):
    from brax.training.agents.ppo import checkpoint, train
    from brax.training.agents.ppo.optimizer import LRSchedule

    cfg = load_training_config(CONFIGS / "ppo_walking_size_aligned.yaml")
    ppo = cfg.ppo
    received = []
    _, params, metrics = train.train(
        environment=aligned_env, num_timesteps=64, num_envs=4,
        episode_length=cfg.environment.episode_length, num_evals=1, run_evals=False,
        learning_rate=ppo.learning_rate, learning_rate_schedule=LRSchedule.NONE,
        entropy_cost=ppo.entropy_cost, discounting=ppo.discounting, gae_lambda=ppo.gae_lambda,
        clipping_epsilon=ppo.clipping_epsilon, max_grad_norm=ppo.max_grad_norm,
        vf_loss_coefficient=2 * ppo.value_loss_coef, desired_kl=ppo.desired_kl,
        unroll_length=ppo.unroll_length, batch_size=1, num_minibatches=4,
        num_updates_per_batch=4, normalize_observations=ppo.normalize_observations,
        normalize_advantage=ppo.normalize_advantage, network_factory=make_network_factory(cfg.network),
        seed=cfg.seed, policy_params_fn=lambda step, policy, params: received.append((int(step), params)),
        save_checkpoint_path=str(tmp_path),
    )
    assert [step for step, _ in received] == [0, 64]
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((params, metrics)))
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(received[0][1][1]), jax.tree.leaves(params[1]), strict=True,
    ))
    restored = checkpoint.load(next(p for p in tmp_path.iterdir() if p.is_dir()))
    for original, loaded in zip(jax.tree.leaves(params), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(original, loaded)
