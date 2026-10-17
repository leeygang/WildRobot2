"""Native Brax update/restore and independent episode accounting."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
from brax import base
from brax.envs.base import Env, State

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.paper_evaluation import PaperEvaluator
from wr2.locomotion.ppo import make_network_factory

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "wr2/locomotion/configs/ppo_walking_paper.yaml"
)


class TinyTrainingEnv(Env):
    """Cheap transitions with the real actor/critic dimensions and action head."""

    @property
    def observation_size(self):
        return {"state": 975, "privileged_state": 1485}

    @property
    def action_size(self):
        return 12

    @property
    def backend(self):
        return "generalized"

    def reset(self, rng):
        return State(
            None,
            {name: jp.zeros(size) for name, size in self.observation_size.items()},
            jp.zeros(()),
            jp.zeros(()),
            {"task_error": jp.zeros(()), "reward": jp.zeros(())},
            {"rng": rng},
        )

    def step(self, state, action):
        error = jp.mean((action - 0.2) ** 2)
        return state.replace(
            obs={
                name: value.at[0].set(action.mean())
                for name, value in state.obs.items()
            },
            reward=-error,
            metrics={**state.metrics, "task_error": error, "reward": -error},
        )


def test_native_ppo_update_and_parameter_only_restore(tmp_path):
    from brax.training import checkpoint as training_checkpoint
    from brax.training.agents.ppo import checkpoint, networks as ppo_networks, train
    from brax.training.agents.ppo.optimizer import LRSchedule

    config = load_training_config(CONFIG)
    ppo = config.ppo
    factory = make_network_factory(config.network)
    received = []
    kwargs = dict(
        environment=TinyTrainingEnv(),
        num_timesteps=80,
        num_envs=4,
        episode_length=20,
        num_evals=2,
        num_eval_envs=1,
        learning_rate=ppo.learning_rate,
        learning_rate_schedule=LRSchedule.NONE,
        max_grad_norm=ppo.max_grad_norm,
        entropy_cost=ppo.entropy_cost,
        discounting=ppo.discounting,
        gae_lambda=ppo.gae_lambda,
        clipping_epsilon=ppo.clipping_epsilon,
        vf_loss_coefficient=2 * ppo.value_loss_coef,
        unroll_length=20,
        batch_size=1,
        num_minibatches=4,
        num_updates_per_batch=4,
        normalize_observations=False,
        normalize_advantage=True,
        network_factory=factory,
        seed=0,
        deterministic_eval=True,
        policy_params_fn=lambda step, make_policy, params: received.append(
            (int(step), params)
        ),
    )
    _, params, metrics = train.train(
        **kwargs, save_checkpoint_path=str(tmp_path / "checkpoints")
    )
    assert received[0][0] == 0
    assert received[-1][0] == 80
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))
    assert all(np.isfinite(np.asarray(x)).all() for x in metrics.values())
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(received[0][1][1]),
            jax.tree.leaves(params[1]),
            strict=True,
        )
    )
    checkpoint_path = next(
        path for path in (tmp_path / "checkpoints").iterdir() if path.is_dir()
    )
    restored = checkpoint.load(checkpoint_path)
    for original, loaded in zip(
        jax.tree.leaves(params), jax.tree.leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(original, loaded)
    metadata = checkpoint.load_config(checkpoint_path)
    nets = training_checkpoint.get_network(metadata, ppo_networks.make_ppo_networks)
    make_policy = ppo_networks.make_inference_fn(nets)
    obs = TinyTrainingEnv().reset(jax.random.PRNGKey(0)).obs
    np.testing.assert_array_equal(
        make_policy(params, deterministic=True)(obs, jax.random.PRNGKey(0))[0],
        make_policy(restored, deterministic=True)(obs, jax.random.PRNGKey(0))[0],
    )
    np.testing.assert_array_equal(
        checkpoint.load_policy(checkpoint_path)(obs, jax.random.PRNGKey(0))[0],
        make_policy(params, deterministic=True)(obs, jax.random.PRNGKey(0))[0],
    )
    # Native Brax restores parameters, not Adam state/counters. Its step-0
    # callback must expose exactly the saved actor, critic and normalizer.
    received.clear()
    train.train(**kwargs, restore_checkpoint_path=str(checkpoint_path))
    assert received[0][0] == 0
    for original, loaded in zip(
        jax.tree.leaves(restored), jax.tree.leaves(received[0][1]), strict=True
    ):
        np.testing.assert_array_equal(original, loaded)


def test_paper_evaluation_loads_relative_brax_checkpoint(tmp_path, monkeypatch):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint
    from wr2.locomotion.evaluate import evaluate_paper_checkpoint

    config = load_training_config(CONFIG)
    env = TinyTrainingEnv()
    factory = make_network_factory(config.network)
    nets = factory(env.observation_size, env.action_size)
    params = (
        running_statistics.init_state(env.reset(jax.random.PRNGKey(0)).obs),
        nets.policy_network.init(jax.random.PRNGKey(1)),
        nets.value_network.init(jax.random.PRNGKey(2)),
    )
    checkpoint.save(
        tmp_path,
        1,
        params,
        checkpoint.network_config(
            env.observation_size,
            env.action_size,
            False,
            factory,
        ),
    )

    def measure(loaded):
        for expected, actual in zip(
            jax.tree.leaves(params), jax.tree.leaves(loaded), strict=True
        ):
            np.testing.assert_array_equal(expected, actual)
        return {
            "normalization_scale": config.environment.normalization_scale,
            "froude_velocity_m_s": 1.4,
            "conditions": [
                {
                    "episodes": 1,
                    "fall_count": 0,
                    "nonfinite_count": 0,
                    "transitions": False,
                    "per_episode": {"walking": {"xy_error_m_s": [0.1]}},
                }
            ],
        }

    noise, varied = [], []

    def environment(*args, **kwargs):
        noise.append(kwargs["add_observation_noise"])
        return env

    def evaluator(env, policy, num_envs, randomize, **kwargs):
        varied.append(randomize is not None)
        return SimpleNamespace(evaluate=measure)

    monkeypatch.setattr("wr2.locomotion.walking_env.WR2WalkingEnv", environment)
    monkeypatch.setattr("wr2.locomotion.paper_evaluation.PaperEvaluator", evaluator)
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        allow_cpu=True,
        config=CONFIG,
        checkpoint=Path("000000000001"),
        solver_iterations=None,
        seeds=(22102,),
        num_envs=1,
        engineering_horizon=False,
        nominal=False,
    )
    for nominal in (False, True):
        args.nominal = nominal
        report = evaluate_paper_checkpoint(args)
        assert report["checkpoint"] == str(tmp_path / "000000000001")
        assert report["seed_results"][0]["summary"]["nonfinite_count"] == 0
        assert report["training_config_version"] == config.version
        assert report["evaluation_source"]["git_commit"]
    assert noise == [True, False]
    assert varied == [True, False]


class EpisodeAccountingEnv(Env):
    """Two episodes: one finite fall at tick 2, one invalid transition at tick 3."""

    def __init__(self):
        self.config = load_training_config(CONFIG).environment
        self.add_observation_noise = False
        self.sys = None
        self._root_link_index = 0
        self.zmp_reference_diagnostics = SimpleNamespace(
            com_height_m=0.2022331821545004
        )

    @property
    def observation_size(self):
        return {"state": 5, "privileged_state": 5}

    @property
    def action_size(self):
        return 1

    @property
    def backend(self):
        return "generalized"

    def reset(self, rng, command=None):
        # Reset key chooses the case; record it for an independent assertion.
        case = jp.mod(rng[0], 2)
        p = base.State(
            jp.zeros(7),
            jp.zeros(6),
            base.Transform.zero((1,)),
            base.Motion.zero((1,)),
            None,
        )
        p = p.replace(x=p.x.replace(rot=jp.asarray([[1.0, 0.0, 0.0, 0.0]])))
        return State(
            p,
            {"state": jp.zeros(5), "privileged_state": jp.zeros(5)},
            jp.zeros(()),
            jp.zeros(()),
            self.metrics(jp.zeros(())),
            {"command": command, "ticks": jp.zeros((), dtype=jp.int32), "case": case},
        )

    def metrics(self, fell):
        return dict(
            fall=fell,
            leg_motor_mse_rad2_per_step=jp.asarray(0.04),
            actuator_torque_rms_nm_per_step=jp.asarray(2.0),
            actuator_torque_peak_nm_per_step=jp.asarray(3.0),
            target_clipping_fraction_per_step=jp.asarray(0.1),
            target_slew_limiting_fraction_per_step=jp.asarray(0.2),
            bounded_action_saturation_fraction_per_step=jp.asarray(0.3),
            left_foot_contact_per_step=jp.asarray(1.0),
            right_foot_contact_per_step=jp.asarray(0.0),
            double_support_per_step=jp.asarray(0.0),
            left_foot_displacement_m_per_step=jp.asarray(0.01),
            right_foot_displacement_m_per_step=jp.asarray(0.02),
            single_support_per_step=jp.asarray(1.0),
            no_foot_support_per_step=jp.asarray(0.0),
            left_foot_vertical_force_n_per_step=jp.asarray(1.6),
            right_foot_vertical_force_n_per_step=jp.asarray(0.8),
        )

    def step(self, state, action):
        tick = state.info["ticks"] + 1
        fell = (state.info["case"] == 0) & (tick >= 2)
        invalid = (state.info["case"] == 1) & (tick >= 3)
        q = state.pipeline_state.q.at[0].set(jp.where(invalid, jp.nan, 0.0))
        return state.replace(
            pipeline_state=state.pipeline_state.replace(q=q),
            metrics=self.metrics(fell.astype(jp.float32)),
            done=fell.astype(jp.float32),
            info={**state.info, "ticks": tick},
        )


def test_independent_evaluator_stops_after_fall_or_nonfinite_and_keeps_terminal_sample():
    def make_policy(params, deterministic):
        return lambda obs, rng: (jp.zeros((obs["state"].shape[0], 1)), {})

    evaluator = PaperEvaluator(EpisodeAccountingEnv(), make_policy, 8)
    assert not evaluator.env.config.randomization.enabled
    lengths, falls, nonfinite, counts, sums = jax.device_get(
        evaluator._run((None, None), jp.asarray([0.08, 0.0]), jp.asarray(False))
    )
    cases = np.asarray(evaluator.reset_keys[:, 0] % 2)
    np.testing.assert_array_equal(lengths, np.where(cases == 0, 2, 3))
    np.testing.assert_array_equal(falls, cases == 0)
    np.testing.assert_array_equal(nonfinite, cases == 1)
    # Finite fall's second transition counts; invalid third transition does not.
    np.testing.assert_array_equal(counts[:, 0], 2)
    np.testing.assert_allclose(sums[:, 0, 2], 0.16)
    np.testing.assert_allclose(sums[:, 0, 4], 0.08)
    np.testing.assert_allclose(sums[:, 0, 5], 8.0)
    np.testing.assert_allclose(sums[:, 0, 13], 0.02)
    np.testing.assert_allclose(sums[:, 0, 15], 2.0)
    np.testing.assert_allclose(sums[:, 0, 16], 0.0)
    np.testing.assert_allclose(sums[:, 0, 17], 3.2)
    np.testing.assert_allclose(sums[:, 0, 18], 1.6)
