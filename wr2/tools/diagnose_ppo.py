"""Inspect stochastic PPO learning at saved WR2 or native TB Paper checkpoints.

No target overrides. Each batch is sampled from the original checkpoint and
gets one normal four-epoch/four-minibatch PPO probe with fresh Adam state.
Probe weights are discarded. This does not resume historical optimizer state.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def load_wr2(run_dir, count):
    import jax.numpy as jp
    from brax.training.agents.ppo import checkpoint
    from wr2.locomotion.configs import load_training_config
    from wr2.locomotion.domain_randomization import make_domain_randomizer
    from wr2.locomotion.ppo import make_network_factory
    from wr2.locomotion.walking_env import WR2WalkingEnv, _inverse_rotate_vector

    cfg = load_training_config(run_dir / "training_config.yaml")
    if cfg.environment.recipe != "tb_paper" or cfg.ppo.normalize_observations:
        raise ValueError("The saved unnormalized TB Paper WR2 recipe is required")
    env = WR2WalkingEnv(
        cfg.environment,
        add_observation_noise=True,
        reward_discounting=cfg.ppo.discounting,
    )
    recorded = json.loads((run_dir / "run_config.json").read_text())
    if sha(env.robot.directory / "wr2_mjx.xml") != recorded["physics"]["model_sha256"]:
        raise ValueError("WR2 physics differs from the completed training run")
    if env.robot.config["actuators"]["htd45hServo"] != recorded["actuator_model"]:
        raise ValueError("WR2 actuator model differs from the completed training run")
    np.testing.assert_allclose(
        env._home_ctrl, recorded["policy_home_motor_position_rad"], atol=1e-6
    )
    if (
        cfg.environment.policy_action_scale_rad != 0.25
        or cfg.ppo.max_grad_norm is not None
    ):
        raise ValueError(
            "This audit requires the saved .25-rad, unclipped-gradient recipe"
        )
    names = [env.robot.actuator_names[int(i)] for i in np.asarray(env._active_indices)]
    hips = tuple(names.index(f"{side}_hip_pitch") for side in ("left", "right"))
    hip_joints = env._joint_qpos_indices[env._active_indices[jp.asarray(hips)]]

    def trace(before, after, command, contact, step):
        scale = cfg.environment.normalization_scale
        positions = after.pipeline_state.geom_xpos[env._foot_geom_ids]
        delta = _inverse_rotate_vector(
            after.pipeline_state.x.rot[env._root_link_index],
            positions[0] - positions[1],
        )
        current_contact = (
            jp.asarray(
                [after.metrics[f"{side}_foot_contact"] for side in ("left", "right")]
            )
            > 0
        )
        reference = env._zmp_reference.get_reference(step * env.dt, command)[0]
        return {
            "command_tb": command * jp.asarray([1 / scale, 1 / scale, scale]),
            "phase": jp.mod(step * env.dt / cfg.environment.gait_cycle_s, 1.0),
            "velocity_tb": jp.asarray(
                [
                    after.metrics["forward_velocity_m_s_per_step"],
                    after.metrics["lateral_velocity_m_s_per_step"],
                ]
            )
            / scale,
            "yaw_rate_tb": after.metrics["yaw_rate_rad_s_per_step"] * scale,
            "landed": current_contact & ~contact & ~after.done.astype(bool),
            "contact": current_contact,
            "landing_lead_tb_m": jp.asarray([delta[0], -delta[0]]) / scale**2,
            "foot_position_tb_m": positions / scale**2,
            "actual_hip_pitch_rad": after.pipeline_state.q[hip_joints],
            "reference_hip_pitch_rad": reference[env._active_indices[jp.asarray(hips)]],
        }

    return SimpleNamespace(
        env=env,
        hips=hips,
        signs=(-1, 1),
        scale=cfg.environment.normalization_scale,
        home=np.asarray(env._home_ctrl)[np.asarray(env._active_indices)][list(hips)],
        factory=make_network_factory(cfg.network),
        trace=trace,
        randomizer=make_domain_randomizer(cfg.environment.randomization)
        if cfg.environment.randomization.enabled
        else None,
        reward=lambda state: {
            name.removeprefix("paper_reward/"): value
            for name, value in state.metrics.items()
            if name.startswith("paper_reward/")
        },
        command=lambda state: state.info["command"],
        contact=lambda state: state.info["previous_contact"],
        step=lambda state: state.info["step_count"],
        settings=dict(
            entropy_cost=cfg.ppo.entropy_cost,
            discounting=cfg.ppo.discounting,
            gae_lambda=cfg.ppo.gae_lambda,
            clipping_epsilon=cfg.ppo.clipping_epsilon,
            normalize_advantage=cfg.ppo.normalize_advantage,
            vf_coefficient=2 * cfg.ppo.value_loss_coef,
        ),
        learning_rate=cfg.ppo.learning_rate,
        epochs=cfg.ppo.num_updates_per_batch,
        minibatches=cfg.ppo.num_minibatches,
        unroll=cfg.ppo.unroll_length,
        batch_size=count // cfg.ppo.num_minibatches,
        loader=lambda step, networks: tuple(
            checkpoint.load(str((run_dir / "checkpoints" / f"{step:012d}").resolve()))
        ),
        checkpoint_steps=[80035840, 120053760],
        source={
            "training_config_sha256": sha(run_dir / "training_config.yaml"),
            "training_provenance": recorded["provenance"],
            "verified_model_sha256": recorded["physics"]["model_sha256"],
        },
    )


def load_tb(run_dir, count):
    """Use the installed native Paper fork and saved configuration, read only."""
    import jax
    import jax.numpy as jp
    from brax.io import model
    from brax.training.agents.ppo import networks
    from brax.training.agents.ppo import losses
    from orbax import checkpoint as ocp
    from toddlerbot.locomotion.diagnose_stance import verify_source
    from toddlerbot.locomotion.mjx_config import MJXConfig
    from toddlerbot.locomotion.train_mjx import (
        domain_randomize,
        get_body_mass_attr_range,
    )
    from toddlerbot.locomotion.walk_env import WalkEnv
    from toddlerbot.sim.robot import Robot
    from toddlerbot.utils.math_utils import quat_inv, rotate_vec

    source = verify_source(run_dir)
    cfg = MJXConfig()
    for section, values in json.loads(
        (run_dir / "env_config.json").read_text()
    ).items():
        for name, value in values.items():
            setattr(getattr(cfg, section), name, value)
    train = json.loads((run_dir / "train_config.json").read_text())
    env = WalkEnv(
        "walk", Robot("toddlerbot"), cfg, add_noise=True, add_domain_rand=True
    )
    names = [env.robot.motor_ordering[int(i)] for i in np.asarray(env.action_mask)]
    hips = tuple(names.index(f"{side}_hip_pitch") for side in ("left", "right"))
    hip_joints = env.q_start_idx + env.motor_indices[env.action_mask[jp.asarray(hips)]]
    # Use the sole pair, not the ankle-link collider pair in the stance traces.
    geoms = jp.asarray(
        [
            env.sys.mj_model.geom(name).id
            for name in ("ank_roll_link_collision", "ank_roll_link_2_collision")
        ]
    )
    mass = get_body_mass_attr_range(
        env.robot,
        cfg.domain_rand.body_mass_range,
        cfg.domain_rand.ee_mass_range,
        cfg.domain_rand.other_mass_range,
        count,
        rng=np.random.default_rng(22103),
    )

    def load_checkpoint(step, networks):
        # The exported policy pickle omits the critic. The adjacent native
        # Orbax checkpoint retains (normalizer, PPONetworkParams(policy,value)).
        normalizer, policy = model.load_params(run_dir / str(step) / "policy")
        target = (
            normalizer,
            losses.PPONetworkParams(
                policy=policy,
                value=networks.value_network.init(jax.random.PRNGKey(0)),
            ),
        )
        target = jax.tree.map(jp.asarray, target)
        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
        restore_args = jax.tree.map(
            lambda value: ocp.ArrayRestoreArgs(
                sharding=sharding, global_shape=value.shape
            ),
            target,
        )
        normalizer, restored = ocp.PyTreeCheckpointer().restore(
            str(run_dir / str(step)), item=target, restore_args=restore_args
        )
        for expected, actual in zip(
            jax.tree.leaves(policy),
            jax.tree.leaves(restored.policy),
            strict=True,
        ):
            np.testing.assert_array_equal(expected, actual)
        return normalizer, restored.policy, restored.value

    def trace(before, after, command, contact, step):
        pipeline = after.pipeline_state
        positions = pipeline.geom_xpos[geoms]
        delta = rotate_vec(positions[0] - positions[1], quat_inv(pipeline.x.rot[0]))
        current_contact = after.info["stance_mask"] > 0
        velocity = rotate_vec(pipeline.xd.vel[0], quat_inv(pipeline.x.rot[0]))
        angular = rotate_vec(pipeline.xd.ang[0], quat_inv(pipeline.x.rot[0]))
        return {
            "command_tb": command[5:8],
            "phase": jp.mod(step * env.dt / cfg.action.cycle_time, 1.0),
            "velocity_tb": velocity[:2],
            "yaw_rate_tb": angular[2],
            "landed": current_contact & ~contact & ~after.done.astype(bool),
            "contact": current_contact,
            "landing_lead_tb_m": jp.asarray([delta[0], -delta[0]]),
            "foot_position_tb_m": positions,
            "actual_hip_pitch_rad": pipeline.q[hip_joints],
            "reference_hip_pitch_rad": after.info["state_ref"][
                env.ref_start_idx + env.action_mask[jp.asarray(hips)]
            ],
        }

    return SimpleNamespace(
        env=env,
        hips=hips,
        signs=(1, -1),
        scale=1.0,
        home=np.asarray(env.default_action)[list(hips)],
        trace=trace,
        factory=partial(
            networks.make_ppo_networks,
            privileged_observation_size=env.privileged_observation_size,
            policy_hidden_layer_sizes=tuple(train["policy_hidden_layer_sizes"]),
            value_hidden_layer_sizes=tuple(train["value_hidden_layer_sizes"]),
        ),
        randomizer=partial(
            domain_randomize,
            friction_range=cfg.domain_rand.friction_range,
            damping_range=cfg.domain_rand.damping_range,
            armature_range=cfg.domain_rand.armature_range,
            frictionloss_range=cfg.domain_rand.frictionloss_range,
            body_mass_attr_range=mass,
        ),
        reward=lambda state: {
            name: state.metrics[name] * env.dt for name in env.reward_names
        },
        command=lambda state: state.info["command"],
        contact=lambda state: state.info["stance_mask"] > 0,
        step=lambda state: state.info["step"],
        settings=dict(
            entropy_cost=train["entropy_cost"],
            discounting=train["discounting"],
            gae_lambda=0.95,
            clipping_epsilon=train["clipping_epsilon"],
            normalize_advantage=True,
        ),
        learning_rate=train["learning_rate"],
        epochs=train["num_updates_per_batch"],
        minibatches=train["num_minibatches"],
        unroll=train["unroll_length"],
        batch_size=count // train["num_minibatches"],
        loader=load_checkpoint,
        checkpoint_steps=[80035840, 200089600],
        source=source,
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_samples(samples, indices, signs, home):
    """Describe requests and physical motion per command and reference phase."""
    command = samples["command_tb"]
    moving = np.linalg.norm(command, axis=-1) > 0.05
    turning = moving & (np.abs(command[..., 2]) > 1e-6)
    lateral = moving & ~turning & (np.abs(command[..., 1]) >= np.abs(command[..., 0]))
    forward = moving & ~turning & ~lateral & (command[..., 0] > 0)
    masks = {
        "all": np.ones_like(moving),
        "standing": ~moving,
        "forward": forward,
        "backward": moving & ~turning & ~lateral & ~forward,
        "lateral": lateral,
        "turn": turning,
    }
    for phase in range(4):
        masks[f"forward_phase_{phase}"] = forward & (
            np.floor(samples["phase"] * 4).astype(int) == phase
        )
    output = {}
    for name, mask in masks.items():
        if not mask.any():
            output[name] = {"samples": 0}
            continue
        selected = {key: value[mask] for key, value in samples.items()}
        mu = selected["mu"][:, indices]
        action = selected["mode"][:, indices]
        row = {
            "samples": int(mask.sum()),
            "hip_mu_mean": mu.mean(axis=0).tolist(),
            "hip_forward_target_mean_deg": np.rad2deg((home + action * 0.25) * signs)
            .mean(axis=0)
            .tolist(),
            "hip_forward_target_std_deg": np.rad2deg(action * 0.25)
            .std(axis=0)
            .tolist(),
            "hip_raw_std_mean": selected["std"][:, indices].mean(axis=0).tolist(),
            "hip_tanh_derivative_mean": (1 - action**2).mean(axis=0).tolist(),
            "hip_reference_mse_rad2": (
                (selected["actual_hip_pitch_rad"] - selected["reference_hip_pitch_rad"])
                ** 2
            )
            .mean(axis=0)
            .tolist(),
            "mean_velocity_tb_m_s": selected["velocity_tb"].mean(axis=0).tolist(),
            "mean_xy_error_tb_m_s": float(
                np.linalg.norm(
                    selected["velocity_tb"] - selected["command_tb"][:, :2], axis=-1
                ).mean()
            ),
            "mean_abs_yaw_error_tb_rad_s": float(
                np.abs(selected["yaw_rate_tb"] - selected["command_tb"][:, 2]).mean()
            ),
            "contact_fraction": selected["contact"].mean(axis=0).tolist(),
            "terminal_transition_fraction": float(selected["terminal"].mean()),
            "fall_transition_fraction": float(selected["fall"].mean()),
            "landing_count": selected["landed"].sum(axis=0).tolist(),
            "landing_lead_tb_mm": [
                float(
                    selected["landing_lead_tb_m"][:, hip][
                        selected["landed"][:, hip]
                    ].mean()
                    * 1000
                )
                if selected["landed"][:, hip].any()
                else None
                for hip in range(2)
            ],
        }
        if "local_pressure" in selected:
            row["component_local_forward_mu_pressure"] = (
                selected["local_pressure"].mean(axis=0).tolist()
            )
        output[name] = row
    return output


def replay_minibatch(args):
    """Compare precisions on the exact failed WR2 batch, without simulation."""
    if args.tb_root:
        raise ValueError("This replay targets the WR2 leg-only Paper batch")
    path = args.replay_minibatch.resolve()
    provenance = json.loads((path.parent / "provenance.json").read_text())
    os.environ["XLA_FLAGS"] = provenance["xla_flags"]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    from brax.training.agents.ppo import losses
    from wr2.locomotion.configs import load_training_config
    from wr2.locomotion.ppo import make_network_factory
    from wr2.locomotion.ppo_diagnostics import make_actor_attribution

    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise RuntimeError(
            "Use the GPU for the precision comparison; --allow-cpu is for development"
        )
    packages = {name: metadata.version(name) for name in provenance["packages"]}
    if jax.default_backend() == "gpu" and packages != provenance["packages"]:
        raise ValueError("Replay must use the saved GPU batch's dependency versions")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    config = load_training_config(args.run_dir / "training_config.yaml")
    data = payload["data"]
    if config.environment.active_groups != ("leg",) or data.action.shape[-1] != 12:
        raise ValueError("The saved 12-action WR2 Paper leg recipe is required")
    shape = jax.tree.map(lambda value: value.shape[2:], data.observation)
    networks = make_network_factory(config.network)(shape, data.action.shape[-1])
    saved = payload["saved_params"]
    params = losses.PPONetworkParams(policy=saved[1], value=saved[2])
    attribute = make_actor_attribution(networks, payload["settings"], (0, 6), (-1, 1))
    report = {
        "input": str(path),
        "input_sha256": sha(path),
        "checkpoint_step": payload["checkpoint_step"],
        "batch": payload["batch"],
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "packages": packages,
        "python": sys.version,
        "saved_packages": provenance["packages"],
        "saved_precision": provenance["jax_default_matmul_precision"],
        "math_sha256": sha(
            Path(__file__).resolve().parents[1] / "locomotion/ppo_diagnostics.py"
        ),
        "tool_sha256": sha(Path(__file__)),
        "results": [],
    }
    for precision in (provenance["jax_default_matmul_precision"], "high", "highest"):
        with jax.default_matmul_precision(precision):
            result = jax.device_get(attribute(params, saved[0], data, payload["rng"]))
        row = {
            "precision": precision,
            **{
                name: float(result[name])
                for name in (
                    "gradient_relative_error",
                    "native_head_backprop_relative_error",
                    "output_gradient_relative_error",
                    "advantage_max_error",
                    "reward_max_error",
                )
            },
            "native_metrics": {
                name: float(value) for name, value in result["native_metrics"].items()
            },
        }
        report["results"].append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    output = (args.output or path.parent / "replay").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved precision replay: {output / 'report.json'}", flush=True)


def run(args):
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "ppo_learning_diagnostic").resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output folder: {output}")
    if args.tb_root:
        root = args.tb_root.resolve()
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / "toddlerbot/brax")]
        os.environ["USE_JAX"] = "true"
    saved_provenance = json.loads(
        (
            run_dir / ("provenance.json" if args.tb_root else "run_config.json")
        ).read_text()
    )
    if not args.tb_root:
        saved_provenance = saved_provenance["provenance"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["XLA_FLAGS"] = saved_provenance["xla_flags"]
    import jax
    import jax.numpy as jp
    import optax
    from brax.envs.base import Wrapper
    from brax.envs.wrappers import training
    from brax.training import acting
    from brax.training.agents.ppo import losses, networks as ppo_networks
    from wr2.locomotion.ppo_diagnostics import (
        COMPONENTS,
        REWARD_GROUPS,
        make_actor_attribution,
        group_rewards,
    )

    precision = (
        "high" if args.tb_root else saved_provenance["jax_default_matmul_precision"]
    )
    jax.config.update("jax_default_matmul_precision", precision)
    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise RuntimeError("Run on GPU; --allow-cpu is for development checks")
    packages = {
        name: metadata.version(name)
        for name in ("brax", "jax", "jaxlib", "mujoco", "mujoco-mjx", "flax", "optax")
    }
    runtime_difference = {
        name: {"saved": saved_provenance["packages"][name], "current": version}
        for name, version in packages.items()
        if saved_provenance["packages"][name] != version
    }
    if jax.default_backend() == "gpu" and runtime_difference:
        raise ValueError(
            f"Use the completed training run's dependency lock/runtime: {runtime_difference}"
        )
    print(
        "Loading native environment, full reference grid and saved checkpoints...",
        flush=True,
    )
    context = (
        load_tb(run_dir, args.num_envs)
        if args.tb_root
        else load_wr2(run_dir, args.num_envs)
    )
    if args.warmup_steps is None:
        args.warmup_steps = round(8.0 * context.scale / float(context.env.dt))
    if args.num_envs % context.minibatches or args.num_envs < context.minibatches:
        raise ValueError(
            "--num-envs must be a positive multiple of the native minibatch count"
        )

    class TraceEnv(Wrapper):
        # Modern Brax's randomizer steps env.unwrapped directly; retain this
        # instrumentation layer. The Paper fork already steps through wrappers.
        @property
        def unwrapped(self):
            return self.env.unwrapped if args.tb_root else self

        @property
        def sys(self):
            return self.env.sys

        @sys.setter
        def sys(self, value):
            self.env.sys = value

        @property
        def backend(self):
            return self.env.backend

        def reset(self, key):
            state = self.env.reset(key)
            command, contact, step = (
                context.command(state),
                context.contact(state),
                context.step(state),
            )
            trace = context.trace(state, state, command, contact, step)
            return state.replace(
                info={
                    **state.info,
                    "reward_components": jp.zeros(len(REWARD_GROUPS)),
                    "trace": trace,
                }
            )

        def step(self, state, action):
            command = context.command(state).copy()
            contact = context.contact(state).copy()
            step = context.step(state).copy()
            next_state = self.env.step(state, action)
            trace = context.trace(state, next_state, command, contact, step)
            return next_state.replace(
                info={
                    **next_state.info,
                    "reward_components": group_rewards(context.reward(next_state)),
                    "trace": trace,
                }
            )

    key = jax.random.PRNGKey(args.seed)
    key, key_env = jax.random.split(key)
    randomizer = (
        partial(context.randomizer, rng=jax.random.split(key_env, args.num_envs))
        if context.randomizer
        else None
    )
    env = training.wrap(
        TraceEnv(context.env),
        context.env.config.episode_length if not args.tb_root else 1000,
        1,
        randomizer,
    )
    reset_keys = jax.random.split(key_env, args.num_envs)
    reset = jax.jit(env.reset)
    state = reset(reset_keys)
    observation_size = (
        state.obs.shape[-1]
        if args.tb_root
        else jax.tree.map(lambda x: x.shape[1:], state.obs)
    )
    networks = context.factory(observation_size, action_size=env.action_size)
    make_policy = ppo_networks.make_inference_fn(networks)
    optimizer = optax.adam(context.learning_rate)
    native_loss = partial(
        losses.compute_ppo_loss, ppo_network=networks, **context.settings
    )

    @jax.jit
    def collect(state, params, key, length=context.unroll):
        return acting.generate_unroll(
            env,
            state,
            make_policy((params[0], params[1])),
            key,
            length,
            extra_fields=("truncation", "reward_components", "trace"),
        )

    @jax.jit
    def warmup(state, params, key):
        # Retain only the state, not hundreds of large observation frames.
        policy = make_policy((params[0], params[1]))

        def step(carry, _):
            state, key = carry
            key, sample_key = jax.random.split(key)
            next_state, _ = acting.actor_step(env, state, policy, sample_key)
            return (next_state, key), None

        (state, _), _ = jax.lax.scan(step, (state, key), None, length=args.warmup_steps)
        return state

    @jax.jit
    def probe_update(params, normalizer, data, key):
        initial = params
        optimizer_state = optimizer.init(params)
        first_metrics = None
        for epoch in range(context.epochs):
            key, permutation_key = jax.random.split(key)
            permutation = jax.random.permutation(permutation_key, args.num_envs)
            shuffled = jax.tree.map(
                lambda x: x[permutation].reshape(
                    (context.minibatches, -1) + x.shape[1:]
                ),
                data,
            )

            def minibatch_step(carry, batch):
                params, opt_state, key = carry
                key, gradient_key = jax.random.split(key)
                (_, metrics), gradients = jax.value_and_grad(native_loss, has_aux=True)(
                    params, normalizer, batch, gradient_key
                )
                updates, opt_state = optimizer.update(gradients, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state, key), metrics

            (params, optimizer_state, key), metrics = jax.lax.scan(
                minibatch_step, (params, optimizer_state, key), shuffled
            )
            if epoch == 0:
                first_metrics = jax.tree.map(lambda x: x[0], metrics)
        old = networks.policy_network.apply(
            normalizer, initial.policy, data.observation
        )
        new = networks.policy_network.apply(normalizer, params.policy, data.observation)
        return params, {
            "mode_delta": networks.parametric_action_distribution.mode(new)
            - networks.parametric_action_distribution.mode(old),
            "mu_delta": networks.parametric_action_distribution.create_dist(new).loc
            - networks.parametric_action_distribution.create_dist(old).loc,
            "new_std": networks.parametric_action_distribution.create_dist(new).scale,
            "native_first_minibatch_metrics": first_metrics,
            "native_last_epoch_metrics": jax.tree.map(jp.mean, metrics),
        }

    attribute = make_actor_attribution(
        networks, context.settings, context.hips, context.signs
    )
    output.mkdir(parents=True)
    provenance = {
        "robot": "toddlerbot" if args.tb_root else "wr2",
        "run_dir": str(run_dir),
        "git_commit": subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "tool_sha256": sha(Path(__file__)),
        "math_sha256": sha(
            Path(__file__).resolve().parents[1] / "locomotion/ppo_diagnostics.py"
        ),
        "source": context.source,
        "packages": packages,
        "runtime_difference": runtime_difference,
        "devices": [str(device) for device in jax.devices()],
        "backend": jax.default_backend(),
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "xla_flags": os.environ["XLA_FLAGS"],
        "git_dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[2]),
                    "status",
                    "--porcelain",
                ],
                text=True,
            ).strip()
        ),
        "settings": context.settings,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "batches": args.batches,
        "warmup_steps": args.warmup_steps,
        "unroll_length": context.unroll,
        "minibatch_size": context.batch_size,
        "epochs": context.epochs,
        "learning_rate": context.learning_rate,
        "components": COMPONENTS,
        "hips": ["left", "right"],
        "forward_joint_signs": context.signs,
        "protocol": "frozen_checkpoint_on_policy_batches; independent_fresh_Adam_PPO_probe_per_batch; probe_weights_discarded",
        "bookkeeping_dtype": "float64; native model/loss/entropy and probes retain float32",
        "gradient_interpretation": "native scalar-reward advantage and clip branch; float64 reward-only GAE plus native value/bootstrapping/rounding residual; projections are infinitesimal SGD, not Adam or causal reward ablations",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    results = []
    for step in args.checkpoint_steps or context.checkpoint_steps:
        saved = context.loader(step, networks)
        params = losses.PPONetworkParams(policy=saved[1], value=saved[2])
        state = reset(reset_keys)
        key, warmup_key = jax.random.split(key)
        state = warmup(state, saved, warmup_key)
        jax.block_until_ready(state.reward)
        print(f"Checkpoint {step}: stochastic warmup complete", flush=True)
        for batch_index in range(args.batches):
            key, sample_key, audit_key, update_key = jax.random.split(key, 4)
            state, time_data = collect(state, saved, sample_key)
            data = jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), time_data)
            # Exactly the first shuffled minibatch and entropy key of the probe.
            first_epoch_key, permutation_key = jax.random.split(update_key)
            _, audit_key = jax.random.split(first_epoch_key)
            indices = jax.random.permutation(permutation_key, args.num_envs)[
                : context.batch_size
            ]
            minibatch = jax.tree.map(lambda x: x[indices], data)
            audit = attribute(params, saved[0], minibatch, audit_key)
            _, update = probe_update(params, saved[0], data, update_key)
            audit, update = jax.device_get((audit, update))
            if not all(np.isfinite(x).all() for x in jax.tree.leaves((audit, update))):
                raise RuntimeError("Nonfinite PPO audit")
            if (
                audit["gradient_relative_error"] > 5e-5
                or audit["output_gradient_relative_error"] > 5e-5
                or audit["advantage_max_error"] > 5e-4
                or audit["reward_max_error"] > 1e-5
            ):
                # Preserve the exact failing batch for debugging without another
                # full reference build/physics warmup. This stays in results.
                with (output / "failed_minibatch.pkl").open("wb") as stream:
                    pickle.dump(
                        jax.device_get(
                            {
                                "checkpoint_step": step,
                                "batch": batch_index,
                                "saved_params": saved,
                                "data": minibatch,
                                "rng": audit_key,
                                "settings": context.settings,
                            }
                        ),
                        stream,
                        protocol=5,
                    )
                raise RuntimeError(
                    f"PPO reconstruction failed: parameter_gradient={audit['gradient_relative_error']}, output_gradient={audit['output_gradient_relative_error']}, advantage={audit['advantage_max_error']}, reward={audit['reward_max_error']}; saved failed_minibatch.pkl"
                )
            for name in audit["native_metrics"]:
                np.testing.assert_allclose(
                    audit["native_metrics"][name],
                    update["native_first_minibatch_metrics"][name],
                    rtol=1e-5,
                    atol=1e-6,
                )
            logits = networks.policy_network.apply(saved[0], saved[1], data.observation)
            distribution = networks.parametric_action_distribution
            samples = jax.device_get(
                {
                    **data.extras["state_extras"]["trace"],
                    "mu": distribution.create_dist(logits).loc,
                    "std": distribution.create_dist(logits).scale,
                    "mode": distribution.mode(logits),
                    "terminal": 1 - data.discount,
                    "fall": (1 - data.discount)
                    * (1 - data.extras["state_extras"]["truncation"]),
                    "truncation": data.extras["state_extras"]["truncation"],
                    "sampled_action": data.action,
                    "reward": data.reward,
                    "reward_components": data.extras["state_extras"][
                        "reward_components"
                    ],
                }
            )
            audit_samples = {
                name: value[np.asarray(indices)] for name, value in samples.items()
            }
            audit_samples["local_pressure"] = np.moveaxis(
                audit["local_forward_mu_pressure"], (0, 1, 2), (2, 1, 0)
            )
            after = {
                **samples,
                "mu": samples["mu"] + update["mu_delta"],
                "mode": samples["mode"] + update["mode_delta"],
                "std": update["new_std"],
            }
            row = {
                "checkpoint_step": step,
                "batch": batch_index,
                "gradient_relative_error": float(audit["gradient_relative_error"]),
                "output_gradient_relative_error": float(
                    audit["output_gradient_relative_error"]
                ),
                "advantage_max_error": float(audit["advantage_max_error"]),
                "reward_max_error": float(audit["reward_max_error"]),
                "projected_forward_target_sgd": audit[
                    "projected_forward_target_sgd"
                ].tolist(),
                "native_metrics": {
                    name: float(value)
                    for name, value in audit["native_metrics"].items()
                },
                "native_probe_metrics": {
                    name: float(value)
                    for name, value in update["native_last_epoch_metrics"].items()
                },
                "samples": summarize_samples(
                    samples, list(context.hips), np.asarray(context.signs), context.home
                ),
                "local_pressure": summarize_samples(
                    audit_samples,
                    list(context.hips),
                    np.asarray(context.signs),
                    context.home,
                ),
                "requests_after_fresh_adam_on_same_observations": {
                    group: {
                        name: value
                        for name, value in metrics.items()
                        if name == "samples" or name.startswith("hip_")
                    }
                    for group, metrics in summarize_samples(
                        after,
                        list(context.hips),
                        np.asarray(context.signs),
                        context.home,
                    ).items()
                },
            }
            results.append(row)
            with (output / "batches.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            np.savez_compressed(
                output / f"{step}_batch{batch_index}.npz",
                **samples,
                mode_delta=update["mode_delta"],
                mu_delta=update["mu_delta"],
                audit_env_indices=np.asarray(indices),
                audit_local_mu_pressure=audit_samples["local_pressure"],
            )
            print(
                f"  batch {batch_index + 1}/{args.batches}: gradient error {row['gradient_relative_error']:.2g}; native PPO probe complete",
                flush=True,
            )
    (output / "summary.json").write_text(
        json.dumps(
            {**provenance, "verified_batches": len(results), "results": results},
            indent=2,
            allow_nan=False,
        )
    )
    print(f"Saved verified PPO diagnostic: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--replay-minibatch",
        type=Path,
        help="Replay a saved failed WR2 batch across matrix precisions; skip simulation",
    )
    parser.add_argument(
        "--tb-root",
        type=Path,
        help="Existing frozen TB Paper checkout; uses its native fork/runtime",
    )
    parser.add_argument("--checkpoint-steps", type=int, nargs="+")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Physical ticks; default is 8 size-normalized TB seconds",
    )
    parser.add_argument("--seed", type=int, default=22103)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if min(args.batches, args.num_envs) < 1 or (
        args.warmup_steps is not None and args.warmup_steps < 1
    ):
        parser.error("Batch, warmup and environment counts must be positive")
    if args.replay_minibatch:
        replay_minibatch(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
