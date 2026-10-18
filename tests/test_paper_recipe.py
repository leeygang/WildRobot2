"""Regression checks against the completed TB paper control (1982ef3)."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from brax import base

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.paper_evaluation import (
    CONDITIONS,
    GROUPS,
    MEASUREMENTS,
    command_and_groups,
    selection_summary,
    summarize,
)
from wr2.locomotion.paper_rewards import paper_reward_terms, range_reward
from wr2.locomotion.ppo import make_network_factory
from wr2.locomotion.walking_env import WR2WalkingEnv, sample_tb_xy_command
from wr2.reference.paper_walk import PaperWalkReference, footstep_plan, preview_com


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "wr2/locomotion/configs/ppo_walking_paper.yaml"
)


def reward_inputs():
    home = jp.linspace(-0.2, 0.2, 21)
    return dict(
        motor_position=home + jp.linspace(-0.02, 0.04, 21),
        home_position=home,
        reference_position=home + 0.05 * jp.cos(jp.arange(21)),
        leg_indices=jp.arange(12),
        alignment_indices=jp.asarray([[0, 4, 3], [6, 10, 9]]),
        motor_velocity=jp.linspace(-0.5, 0.5, 21),
        torque=jp.linspace(-2, 2, 21),
        linear_velocity=jp.asarray([0.14, -0.03, 0.02]),
        angular_velocity=jp.asarray([0.08, -0.07, 0.21]),
        command=jp.asarray([0.1, -0.05, 0.2]),
        orientation_error=0.0,
        euler=jp.zeros(3),
        action=jp.linspace(-0.3, 0.4, 12),
        last_action=jp.linspace(-0.2, 0.1, 12),
        second_last_action=-jp.linspace(-0.2, 0.1, 12),
        contact=jp.asarray([True, False]),
        previous_contact=jp.asarray([False, True]),
        reference_contact=jp.asarray([True, True]),
        air_time=jp.asarray([0.08, 0.12]),
        air_distance=jp.asarray([0.07, 0.0]),
        foot_velocity_xy=jp.asarray([[0.2, 0.4], [0.8, 0.1]]),
        foot_distance=0.045,
        foot_distance_scale=1.0,
        minimum_foot_distance=0.05,
        maximum_foot_distance=0.13,
        collision_count=2.0,
        terminal=jp.asarray(True),
        scale=1.0,
        deadzone=0.05,
    )


def test_complete_rewards_match_frozen_tb_numeric_oracle():
    # TB formula oracle on these inputs with local leg action indices. This
    # excludes the frozen global-index bug in its leg smoothness rewards.
    # The native slip broadcast bug is also excluded by the per-foot value.
    expected = {
        "torso_pos": 0.0,
        "torso_quat": 1.0,
        "lin_vel_xy": 0.8187307715,
        "lin_vel_z": 0.9607894421,
        "ang_vel_xy": 0.7538966537,
        "ang_vel_z": 0.9975031018,
        "leg_motor_pos": -0.001578521449,
        "neck_motor_pos": 0.0,
        "arm_motor_pos": 0.0,
        "waist_motor_pos": 0.0,
        "feet_contact": 1.0,
        "feet_air_time": 0.1999999881,
        "feet_clearance": 0.0700000003,
        "feet_distance": 0.8032653332,
        "feet_slip": -0.2,
        "align_ground": -0.02252499759,
        "stand_still": 0.0,
        "torso_roll": 1.0,
        "torso_pitch": 1.0,
        "collision": -2.0,
        "leg_action_rate": -0.02575757727,
        "leg_action_acc": -0.04393939301,
        "motor_torque": -1.466666698,
        "energy": -0.2412666678,
        "survival": -1.0,
    }
    actual = paper_reward_terms(**reward_inputs())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert float(actual[name]) == pytest.approx(value, abs=2e-6), name
    # Filtered contact reads old accumulators, and terminal survival is -10*dt.
    assert float(actual["feet_air_time"]) == pytest.approx(0.2)
    assert float(actual["survival"] * 10 * 0.02) == pytest.approx(-0.2)
    assert float(range_reward(0.12, -0.1, 0.1)) == pytest.approx((1 + np.exp(-2)) / 2)


def test_reference_imitation_and_fixed_home_have_separate_roles():
    args = reward_inputs()
    args.update(
        motor_position=args["home_position"],
        reference_position=args["home_position"],
        command=jp.zeros(3),
    )
    still = paper_reward_terms(**args)
    assert float(still["leg_motor_pos"]) == 0
    assert float(still["stand_still"]) == 0
    args["reference_position"] = args["reference_position"].at[:12].add(0.1)
    moving_teacher = paper_reward_terms(**args)
    assert float(moving_teacher["leg_motor_pos"]) == pytest.approx(-0.01)
    assert float(moving_teacher["stand_still"]) == 0
    # Nonzero lateral reference classification is independent of the deadzone.
    assert bool(
        WR2WalkingEnv._is_walking(
            SimpleNamespace(config=load_training_config(CONFIG).environment),
            jp.asarray([0.0, 0.04, 0.0]),
        )
    )


def test_reward_normalization_preserves_tb_equivalent_motion():
    args = reward_inputs()
    expected = paper_reward_terms(**args)
    s = 0.824
    args.update(
        scale=s,
        linear_velocity=args["linear_velocity"] * s,
        angular_velocity=args["angular_velocity"] / s,
        motor_velocity=args["motor_velocity"] / s,
        command=args["command"] * jp.asarray([s, s, 1 / s]),
        air_time=args["air_time"] * s,
        air_distance=args["air_distance"] * s**3,
        foot_velocity_xy=args["foot_velocity_xy"] * s,
    )
    actual = paper_reward_terms(**args)
    for name in expected:
        # Landing impulses scale with cycle duration at a fixed control dt;
        # their rates retain the same balance with continuous reward terms.
        factor = s if name in ("feet_air_time", "feet_clearance") else 1.0
        assert float(actual[name]) == pytest.approx(
            float(expected[name]) * factor, abs=2e-6
        ), name


def test_landing_reward_rates_match_tb_at_fixed_control_timestep():
    config = load_training_config(CONFIG).environment

    def cycle_rates(scale, frames):
        dt = 0.02
        swing_frames = frames // 3
        half = swing_frames // 2
        heights = (
            0.05
            * scale**2
            * np.concatenate([np.arange(half), np.arange(half - 1, -1, -1)])
            / (half - 1)
        )
        args = reward_inputs()
        args.update(
            scale=scale,
            command=jp.asarray([0.1 * scale, 0.0, 0.0]),
            linear_velocity=jp.asarray([0.1 * scale, 0.0, 0.0]),
            contact=jp.ones(2, dtype=bool),
            air_time=jp.full(2, swing_frames * dt),
            air_distance=jp.full(2, heights.sum()),
        )
        terms = paper_reward_terms(**args)
        duration = frames * dt
        # Two landings per cycle; linearity lets us sum their accumulators.
        return np.asarray(
            [
                float(terms["feet_air_time"]) * 500 * dt / duration,
                float(terms["feet_clearance"]) * 0.05 * dt / duration,
                float(terms["lin_vel_xy"]) * 5,
            ]
        )

    # Native TB: 0.72 s / 36 ticks. WR2: rounded 0.60 s / 30 ticks.
    native = cycle_rates(1.0, 36)
    scaled = cycle_rates(config.normalization_scale, round(config.gait_cycle_s / 0.02))
    np.testing.assert_allclose(native, [20 / 3, 1 / 1200, 5], rtol=2e-6)
    np.testing.assert_allclose(scaled, native, rtol=2e-6)


@pytest.mark.parametrize("distance", [-0.001, 0.001])
def test_paper_contact_threshold_applies_to_summed_foot_force(distance):
    # Frozen TB accumulates every point on a floor/foot pair before applying
    # 1 N. Active solver forces can also occur within a positive contact margin.
    env = object.__new__(WR2WalkingEnv)
    env.config = SimpleNamespace(contact_force_threshold_n=1.0)
    env._paper_recipe = True
    env._floor_geom_id = 0
    env._foot_geom_ids = (1, 2)
    pipeline = SimpleNamespace(
        contact=SimpleNamespace(
            geom1=jp.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1]),
            geom2=jp.asarray([1, 1, 1, 1, 2, 2, 2, 2, 3]),
            dist=jp.full(9, distance),
        )
    )
    forces = jp.zeros((9, 6)).at[:, 2].set(jp.asarray([0.4] * 4 + [0.2] * 4 + [100.0]))
    contacts = jax.jit(lambda f: env._foot_contact(pipeline, f))(forces)
    np.testing.assert_array_equal(contacts, [True, False])
    np.testing.assert_allclose(env._foot_vertical_forces(pipeline, forces), [1.6, 0.8])
    np.testing.assert_array_equal(
        env._foot_contact(pipeline, jp.zeros_like(forces)), [False, False]
    )


def test_preview_matches_native_four_state_tb_planner():
    feet = np.asarray([[0.0027, 0.0364, 0.0], [0.0027, -0.0364, 0.0]])
    _, times, desired = footstep_plan(np.asarray([0.1, 0.0, 0.0]), 0.72, feet, 1.0)
    com, cop = preview_com(times, desired, np.zeros(2), 0.3267, 0.02)
    indices = [0, 5, 25, 100, 500]
    # Golden native 4-state LQR/preview forward-Euler result, allowing float32
    # rounding. WR2 uses the equivalent independent 2-state axis equations.
    np.testing.assert_allclose(
        com[indices],
        [
            [0.0, 0.0],
            [0.000831897818, -0.000023544369],
            [0.02139137956, -0.000930666407],
            [0.18250111725, -0.000262499677],
            [0.99471334063, -0.000234099796],
        ],
        atol=2e-7,
    )
    np.testing.assert_allclose(
        cop[indices],
        [
            [-0.007079931602, 0.000150603945],
            [-0.005493624051, 0.000317629496],
            [0.01809314173, -0.001610684235],
            [0.18282593775, -0.000823069157],
            [0.99459572543, -0.000786325944],
        ],
        atol=2e-7,
    )


def test_signed_turn_sampler_matches_frozen_tb_and_normalization():
    config = load_training_config(CONFIG).environment
    # Native TB outputs for these seeds, multiplied by [s, s, 1/s].
    golden = {
        0: [0.0726323500276, -0.0061458288692, 0.0],
        3: [0.0, 0.0, 0.1348421871662],
        7: [0.0, 0.0, -0.4362189769745],
        8: [0.0, 0.0, 0.0],
    }
    for seed, native in golden.items():
        result = sample_tb_xy_command(jax.random.PRNGKey(seed), config)
        np.testing.assert_allclose(
            result,
            np.asarray(native)
            * [
                config.normalization_scale,
                config.normalization_scale,
                1 / config.normalization_scale,
            ],
            atol=1e-7,
        )
    commands = np.asarray(
        jax.jit(jax.vmap(lambda k: sample_tb_xy_command(k, config)))(
            jax.random.split(jax.random.PRNGKey(222), 10000)
        )
    )
    stand = np.linalg.norm(commands, axis=1) == 0
    turn = commands[:, 2] != 0
    assert 0.18 < stand.mean() < 0.22
    assert 0.28 < turn.mean() < 0.32
    assert np.all(np.abs(commands[turn, 2]) >= 0.05 / config.normalization_scale - 1e-7)
    assert np.all(commands[turn, :2] == 0)


@pytest.fixture(scope="module")
def paper_env():
    config = load_training_config(CONFIG).environment
    # Provide the explicit command grid through the constructor so command
    # indices and table rows retain the same relationship as production.
    from unittest.mock import patch

    def reference(model, **kwargs):
        s = kwargs["normalization_scale"]
        return PaperWalkReference(
            model,
            **kwargs,
            command_vectors=[
                [0.0, 0.0, 0.0],
                [0.1 * s, 0.0, 0.0],
                [0.0, 0.05 * s, 0.0],
                [0.0, 0.0, 0.5 / s],
            ],
        )

    with patch("wr2.locomotion.walking_env.PaperWalkReference", reference):
        return WR2WalkingEnv(config, add_observation_noise=False)


def home_pipeline(env):
    data = mujoco.MjData(env.sys.mj_model)
    data.qpos[:] = np.asarray(env._default_qpos)
    mujoco.mj_forward(env.sys.mj_model, data)
    return SimpleNamespace(
        q=jp.asarray(data.qpos),
        qd=jp.asarray(data.qvel),
        x=base.Transform(pos=jp.asarray(data.xpos[1:]), rot=jp.asarray(data.xquat[1:])),
        xd=base.Motion(
            vel=jp.zeros((env.sys.nbody - 1, 3)), ang=jp.zeros((env.sys.nbody - 1, 3))
        ),
        actuator_force=jp.zeros(env.robot.actuator_count),
    )


def test_paper_home_observation_layout_and_actor_reference_independence(paper_env):
    env = paper_env
    np.testing.assert_allclose(
        env._policy_target(jp.zeros(12)), env._default_qpos[env._joint_qpos_indices]
    )
    pipeline = home_pipeline(env)
    imu = {
        "gyro_colored": jp.zeros(3),
        "gyro_bias": jp.zeros(3),
        "gravity_colored": jp.zeros(3),
        "gravity_bias": jp.zeros(3),
    }
    args = (
        pipeline,
        jp.zeros(12),
        jp.asarray([0.1 * env.config.normalization_scale, 0.0, 0.0]),
        jp.asarray(0.3),
        jax.random.PRNGKey(5),
        imu,
        jp.zeros(21),
        jp.ones(2, dtype=bool),
        jp.ones(2, dtype=bool),
    )
    static, _ = env._observation(*args, reference_time_s=0.0)
    moving, _ = env._observation(*args, reference_time_s=0.1)
    assert static["state"].shape == (65,)
    assert static["privileged_state"].shape == (99,)
    np.testing.assert_array_equal(static["state"], moving["state"])
    assert not np.allclose(
        static["privileged_state"][59:80], moving["privileged_state"][59:80]
    )
    np.testing.assert_allclose(static["state"][5:47], 0, atol=1e-7)
    np.testing.assert_array_equal(env._policy_target(jp.zeros(12)), env._home_ctrl)
    assert env._single_observation_size * env._observation_history_frames == 975
    assert (
        env._privileged_single_observation_size * env._observation_history_frames
        == 1485
    )


def test_delayed_actions_and_native_auto_reset_preserve_paper_timing(paper_env):
    from brax.envs import training

    env = copy.copy(paper_env)
    env.config = replace(
        env.config,
        episode_length=3,
        zero_command_probability=1.0,
        turn_command_probability=0.0,
        randomization=replace(env.config.randomization, enabled=False),
    )
    wrapped = training.wrap(env, episode_length=3)
    reset, step = jax.jit(wrapped.reset), jax.jit(wrapped.step)
    initial = reset(jax.random.split(jax.random.PRNGKey(0), 1))
    state = initial
    for index, (action, applied, previous, rate) in enumerate(
        (
            (0.2, 0.0, 0.0, -0.04),
            (-0.1, 0.2, 0.0, -0.01),
            (0.3, -0.1, 0.2, -0.01),
        )
    ):
        state = step(state, jp.full((1, 12), action))
        metrics = state.metrics
        double = metrics["double_support_per_step"]
        single = metrics["single_support_per_step"]
        unsupported = metrics["no_foot_support_per_step"]
        np.testing.assert_allclose(double + single + unsupported, 1.0)
        np.testing.assert_allclose(
            unsupported,
            1
            - metrics["left_foot_contact_per_step"]
            - metrics["right_foot_contact_per_step"]
            + double,
        )
        for side in ("left", "right"):
            np.testing.assert_array_equal(
                metrics[f"{side}_foot_contact_per_step"],
                metrics[f"{side}_foot_vertical_force_n_per_step"]
                > env.config.contact_force_threshold_n,
            )
        assert float(state.metrics["paper_reward/leg_action_rate"][0]) == pytest.approx(
            0.05 * 0.02 * rate, abs=1e-9
        )
        if index < 2:
            np.testing.assert_allclose(state.info["last_applied_action"], applied)
            np.testing.assert_allclose(state.obs["state"][:, 47:59], previous)
            assert not bool(state.done[0])
    assert bool(state.done[0])
    assert bool(state.info["truncation"][0])
    np.testing.assert_array_equal(state.pipeline_state.q, initial.pipeline_state.q)
    np.testing.assert_array_equal(state.obs["state"], initial.obs["state"])
    for name in (
        "step_count",
        "previous_action",
        "last_applied_action",
        "second_last_applied_action",
        "feet_air_time",
        "feet_air_distance",
    ):
        np.testing.assert_array_equal(state.info[name], np.zeros_like(state.info[name]))
    np.testing.assert_array_equal(state.info["last_euler"], initial.info["last_euler"])
    state = step(state, jp.full((1, 12), 0.2))
    assert not bool(state.done[0])
    assert int(state.info["step_count"][0]) == 1
    assert float(state.metrics["paper_reward/leg_action_rate"][0]) == pytest.approx(
        -0.00004, abs=1e-9
    )


def test_explicit_checkpoint_initializer_preserves_legacy_normal_initialization():
    from brax.training.agents.ppo import networks as ppo_networks

    config = load_training_config().network
    assert config.distribution_type == "normal"
    observation_size = {"state": 10, "privileged_state": 20}
    explicit = make_network_factory(config)(observation_size, 12)
    native = ppo_networks.make_ppo_networks(
        observation_size,
        12,
        policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.value_hidden_layer_sizes,
        activation=getattr(jax.nn, config.activation),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="normal",
        noise_std_type=config.noise_std_type,
        init_noise_std=config.init_noise_std,
    )
    for a, b in zip(
        jax.tree.leaves(explicit.policy_network.init(jax.random.PRNGKey(3))),
        jax.tree.leaves(native.policy_network.init(jax.random.PRNGKey(3))),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_native_normal_tanh_log_probability_and_checkpoint_metadata():
    from brax.training import checkpoint as training_checkpoint
    from brax.training.agents.ppo import checkpoint, networks as ppo_networks
    from brax.training.acme import running_statistics

    config = load_training_config(CONFIG)
    factory = make_network_factory(config.network)
    obs_size = {"state": 975, "privileged_state": 1485}
    nets = factory(obs_size, 12)
    params = nets.policy_network.init(jax.random.PRNGKey(0))
    obs = {name: jp.zeros((4, size)) for name, size in obs_size.items()}
    statistics = running_statistics.init_state(
        {name: jp.zeros(size) for name, size in obs_size.items()}
    )
    logits = nets.policy_network.apply(statistics, params, obs)
    assert logits.shape == (4, 24)
    np.testing.assert_allclose(
        jax.nn.softplus(logits[:, 12:]) + 0.001, np.log(2) + 0.001
    )
    distribution = nets.parametric_action_distribution
    raw = distribution.sample_no_postprocessing(logits, jax.random.PRNGKey(4))
    action = distribution.postprocess(raw)
    assert np.all(np.abs(action) < 1)
    sigma = np.log(2) + 0.001
    gaussian = (
        -0.5 * (np.asarray(raw) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    )
    jacobian = 2 * (np.log(2) - np.asarray(raw) - np.logaddexp(0, -2 * np.asarray(raw)))
    np.testing.assert_allclose(
        distribution.log_prob(logits, raw),
        np.sum(gaussian - jacobian, axis=-1),
        atol=3e-6,
    )
    meta = checkpoint.network_config(obs_size, 12, False, factory)
    assert meta.network_factory_kwargs.distribution_type == "tanh_normal"
    assert tuple(meta.network_factory_kwargs.policy_hidden_layer_sizes) == (
        512,
        256,
        128,
    )
    assert meta.network_factory_kwargs.policy_obs_key == "state"
    assert meta.network_factory_kwargs.value_obs_key == "privileged_state"
    assert meta.network_factory_kwargs.state_dependent_std
    reconstructed = training_checkpoint.get_network(
        meta, ppo_networks.make_ppo_networks
    )
    restored_policy = ppo_networks.make_inference_fn(reconstructed)(
        (statistics, params), deterministic=True
    )
    np.testing.assert_allclose(
        restored_policy(obs, jax.random.PRNGKey(0))[0], action * 0, atol=1e-7
    )


def test_normalized_schedule_and_per_episode_aggregates():
    s = load_training_config(CONFIG).environment.normalization_scale
    target = jp.asarray([0.1 * s, 0.0])
    for tick, moving in (
        (123, False),
        (124, True),
        (370, True),
        (371, False),
        (493, False),
        (494, True),
        (699, True),
        (700, False),
    ):
        command, groups = command_and_groups(tick, target, True, s)
        assert bool(groups[2]) == moving
        assert bool(jp.linalg.norm(command) > 0) == moving
    sums = np.zeros((2, len(GROUPS), len(MEASUREMENTS)))
    counts = np.zeros((2, len(GROUPS)), dtype=int)
    counts[:, 0] = [1, 3]
    sums[:, 0, 2] = [1.0, 9.0]  # episode errors 1 and 3 => mean 2, not pooled 2.5
    sums[:, 0, 4] = [1.0, 27.0]  # episode leg RMSE 1 and 3 => mean 2
    sums[:, 0, 5] = [4.0, 48.0]  # episode torque RMS 2 and 4 => mean 3
    sums[:, 0, 13] = [2.0, 6.0]  # path is a sum, not a time mean
    sums[:, 0, 15] = [1.0, 1.0]  # support fraction is averaged within each episode
    sums[:, 0, 16] = [0.0, 1.0]
    sums[:, 0, 17] = [1.6, 4.8]  # foot force is a mean in N, not an RMS
    groups, episodes = summarize(sums, counts)
    assert groups["all"]["xy_error_m_s"]["mean"] == 2
    assert groups["all"]["leg_motor_rmse_rad"]["mean"] == 2
    assert groups["all"]["torque_rms_nm"]["mean"] == 3
    assert groups["all"]["left_foot_xy_path_m"]["mean"] == 4
    assert groups["all"]["single_support_fraction"]["mean"] == pytest.approx(2 / 3)
    assert groups["all"]["no_foot_support_fraction"]["mean"] == pytest.approx(1 / 6)
    assert groups["all"]["left_foot_vertical_force_n"]["mean"] == pytest.approx(1.6)
    assert groups["walking"]["xy_error_m_s"]["mean"] is None
    assert episodes["walking"]["xy_error_m_s"] == [None, None]
    assert len(CONDITIONS) == 15


def test_selection_requires_complete_finite_surviving_suite():
    conditions = [
        {
            "episodes": 128,
            "fall_count": 0,
            "nonfinite_count": 0,
            "transitions": transition,
            "per_episode": {"walking": {"xy_error_m_s": [0.03] * 128}},
        }
        for _, _, transition in CONDITIONS
    ]
    report = {
        "conditions": conditions,
        "normalization_scale": 0.824,
        "froude_velocity_m_s": 1.4,
    }
    assert selection_summary(report)["eligible"]
    conditions[0]["nonfinite_count"] = 1
    assert not selection_summary(report)["eligible"]
    conditions[0]["nonfinite_count"] = 0
    conditions[0]["fall_count"] = 20
    assert not selection_summary(report)["eligible"]
    assert not selection_summary({**report, "conditions": conditions[:2]})["eligible"]
