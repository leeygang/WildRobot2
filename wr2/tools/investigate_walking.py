"""Replay held-out walking episodes with pre-autoreset traces and paired ablations."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.evaluate import _parse_seeds
from wr2.locomotion.train import (
    _configure_backend_logging,
    _filter_optional_backend_import_messages,
)


def evaluation_keys(seed, num_envs):
    """Match a single-command evaluate.py run and Evaluator's first split.

    For a command inside a multi-command report, pass its keyed seed:
    original_seed + 100_000 * command_index, as evaluate.py does.
    """
    import jax

    wrap_key, eval_key = jax.random.split(jax.random.PRNGKey(seed))
    _, unroll_key = jax.random.split(eval_key)
    return jax.random.split(wrap_key, num_envs), unroll_key


def ablation_config(config, case):
    """Dynamics includes model, reset pose, servo response, bias and backlash."""
    if case not in ("full", "dynamics", "noise", "nominal"):
        raise ValueError(f"Unknown ablation: {case}")
    dynamics = case in ("full", "dynamics")
    randomization = replace(config.randomization, enabled=dynamics)
    if not dynamics:
        randomization = replace(
            randomization,
            friction_scale=(1.0, 1.0),
            damping_scale=(1.0, 1.0),
            armature_scale=(1.0, 1.0),
            frictionloss_scale=(1.0, 1.0),
            body_mass_scale=(1.0, 1.0),
        )
    return replace(config, randomization=randomization), case in ("full", "noise")


def recording_environment(config, *, noise, restart_step=600, transitions=True):
    """Record inside the base env: the DR wrapper bypasses ordinary wrappers.

    AutoReset replaces pipeline_state on termination but retains this info field.
    Changing restart time is a timing sensitivity probe, NOT a phase-only causal
    intervention (the preceding standing duration changes too).
    """
    import jax.numpy as jp
    from wr2.locomotion.evaluate import transition_environment
    from wr2.locomotion.walking_env import WR2WalkingEnv
    from wr2.tools.diagnose_walking import sample_step

    if not 450 < restart_step <= 700:
        raise ValueError("restart_step must be between 451 and 700")
    parent = (
        type(transition_environment(config, add_observation_noise=noise))
        if transitions
        else WR2WalkingEnv
    )
    reward_names = tuple(asdict(config.rewards))

    class RecordingEnvironment(parent):
        def _next_command(self, step_count, command, rng):
            if not transitions or restart_step == 600:
                return super()._next_command(step_count, command, rng)
            walking = ((step_count >= 150) & (step_count < 450)) | (
                (step_count >= restart_step) & (step_count < restart_step + 250)
            )
            return jp.where(
                walking,
                jp.asarray([config.command_forward_range_m_s[0], 0.0, 0.0]),
                jp.zeros(3),
            )

        def _trace(self, before, after, action):
            trace = sample_step(self, before, after, action)
            trace.update(
                action=action,
                qpos=after.pipeline_state.q,
                qvel=after.pipeline_state.qd,
                actuator_torque_nm=after.pipeline_state.actuator_force,
                foot_position_m=after.pipeline_state.site_xpos[self._foot_site_ids],
                reward=after.reward,
                reward_components=jp.stack(
                    [
                        abs(getattr(config.rewards, name))
                        * (jp.asarray(1.0) if name == "alive" else after.metrics[name])
                        * self.robot.control_period_s
                        for name in reward_names
                    ]
                ),
            )
            return trace

        def reset(self, rng):
            state = super().reset(rng)
            state.info["diagnostic"] = self._trace(
                state, state, jp.zeros(self.action_size)
            )
            return state

        def step(self, state, action):
            after = super().step(state, action)
            after.info["diagnostic"] = self._trace(state, after, action)
            return after

    env_config = replace(config, zero_command_probability=1.0 if transitions else 0.0)
    return RecordingEnvironment(env_config, add_observation_noise=noise)


def summarize_trace(trace, dt, active_indices):
    """Include the first terminal step; exclude every subsequent autoreset step."""
    done = np.asarray(trace["episode_done"], dtype=bool)
    valid = (np.cumsum(done, axis=0) - done) == 0
    walking = trace["command_m_s"] >= 0.05
    error = np.abs(trace["velocity_torso_m_s"][:, :, 0] - trace["command_m_s"])
    clip = (trace["clip_lower"] | trace["clip_upper"])[:, :, active_indices]

    def mean(values, mask):
        return float(np.mean(values[mask])) if np.any(mask) else None

    failures = []
    for episode in range(done.shape[1]):
        terminal = np.flatnonzero(
            valid[:, episode]
            & ((trace["fall"][:, episode] > 0) | (trace["nonfinite"][:, episode] > 0))
        )
        if not terminal.size:
            continue
        step = int(terminal[0])
        failures.append(
            {
                "episode_index": episode,
                "step": step + 1,
                "time_s": (step + 1) * dt,
                "command_m_s": float(trace["command_m_s"][step, episode]),
                "phase_deg": float(np.rad2deg(trace["phase_rad"][step, episode])),
                "height_m": float(trace["position_m"][step, episode, 2]),
                "fall": bool(trace["fall"][step, episode]),
                "nonfinite": bool(trace["nonfinite"][step, episode]),
            }
        )
    return {
        "episodes": done.shape[1],
        "avg_episode_length": float(np.mean(valid.sum(axis=0))),
        "fall_count": sum(failure["fall"] for failure in failures),
        "failures": failures,
        "walking_speed_mae_m_s": mean(error, valid & walking),
        "standing_speed_mae_m_s": mean(error, valid & ~walking),
        "walking_vx_m_s": mean(trace["velocity_torso_m_s"][:, :, 0], valid & walking),
        "heading_mae_deg": mean(np.abs(trace["heading_error_deg"]), valid),
        "active_target_clipping_fraction": mean(clip.mean(axis=-1), valid),
        "reward_component_returns": np.sum(
            np.where(valid[:, :, None], trace["reward_components"], 0.0), axis=0
        )
        .mean(axis=0)
        .tolist(),
    }


def projected_policy(policy, source, signs, active, *, heading):
    """Diagnostic actor reflection; never modify checkpoint or training policy."""
    import jax.numpy as jp
    from wr2.tools.verify_walking_symmetry import mirror_actor

    source, signs, active = jp.asarray(source), jp.asarray(signs), jp.asarray(active)
    lookup = jp.full(17, -1, dtype=jp.int32).at[active].set(jp.arange(len(active)))
    action_source = lookup[source[active]]

    def projected(observation, key):
        action, extras = policy(observation, key)
        reflected = mirror_actor(
            observation["state"], source, signs, active, heading=heading
        )
        other, _ = policy({"state": reflected}, key)
        return 0.5 * (action + other[..., action_source] * signs[active]), extras

    return projected


def run(args):
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        from brax.envs import training as env_training
        from brax.envs.wrappers.training import EvalWrapper
        from brax.training import acting
        from wr2.locomotion.domain_randomization import make_domain_randomizer
        from wr2.locomotion.rsl_rl_training import (
            _jax_policy_factory,
            load_rsl_actor_parameters,
        )

    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise RuntimeError("Use --allow-cpu for intentional Mac/development replay")
    if args.num_envs < 1 or args.solver_iterations < 1:
        raise ValueError("num_envs and solver_iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_training_config(args.config)
    base = replace(
        config.environment,
        solver_iterations=args.solver_iterations,
        command_forward_range_m_s=(args.command, args.command),
        zero_command_probability=0.0,
    )
    policy_factory = _jax_policy_factory(config.network.activation)
    report = {
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "solver_iterations": args.solver_iterations,
        "restart_step": args.restart_step,
        "mode": "fixed" if args.fixed else "transitions",
        "restart_probe_changes_standing_duration": args.restart_step != 600,
        "seeds": args.seeds,
        "num_envs": args.num_envs,
        "force_sampling": "last_physics_substep_not_control_period_impulse",
        "common_reset_joint_and_velocity_jitter": True,
        "actor_projection": args.project_actor,
        "reward_names": list(asdict(base.rewards)),
        "batches": [],
    }
    params = [load_rsl_actor_parameters(path) for path in args.checkpoints]
    for case in args.cases:
        env_config, noise = ablation_config(base, case)
        env = recording_environment(
            env_config,
            noise=noise,
            restart_step=args.restart_step,
            transitions=not args.fixed,
        )
        report["actuator_names"] = list(env.robot.actuator_names)
        report["active_actuator_indices"] = np.asarray(env._active_indices).tolist()
        report["model_sha256"] = hashlib.sha256(
            (env.robot.directory / "wr2_mjx.xml").read_bytes()
        ).hexdigest()
        report["scene_sha256"] = hashlib.sha256(
            (env.robot.directory / "scene_mjx.xml").read_bytes()
        ).hexdigest()
        if args.project_actor:
            import mujoco
            from wr2.tools.verify_walking_symmetry import joint_reflection

            model = mujoco.MjModel.from_xml_path(
                str(env.robot.directory / "scene_mjx.xml")
            )
            source, signs = joint_reflection(model, env.robot.actuator_names)
        randomizer = make_domain_randomizer(env_config.randomization)
        for seed in args.seeds:
            model_keys, unroll_key = evaluation_keys(seed, args.num_envs)
            wrapped = EvalWrapper(
                env_training.wrap(
                    env,
                    episode_length=base.episode_length,
                    action_repeat=1,
                    randomization_fn=functools.partial(randomizer, rng=model_keys),
                )
            )

            def rollout(parameters, key):
                initial = wrapped.reset(jax.random.split(key, args.num_envs))
                policy = policy_factory(parameters)
                if args.project_actor:
                    policy = projected_policy(
                        policy,
                        source,
                        signs,
                        env._active_indices,
                        heading=env_config.heading_observation,
                    )
                final, trajectory = acting.generate_unroll(
                    wrapped,
                    initial,
                    policy,
                    key,
                    unroll_length=base.episode_length,
                    extra_fields=("diagnostic", "episode_done"),
                )
                return final.info["eval_metrics"], trajectory.extras["state_extras"]

            compiled = jax.jit(rollout)
            for index, (path, parameters) in enumerate(
                zip(args.checkpoints, params, strict=True)
            ):
                label = f"policy{index}_{case}_seed{seed}"
                print(f"Replay {label}: {path} ({args.num_envs} episodes)", flush=True)
                evaluation, extras = jax.tree.map(
                    np.asarray, compiled(parameters, unroll_key)
                )
                trace = {**extras["diagnostic"], "episode_done": extras["episode_done"]}
                summary = summarize_trace(
                    trace, env.robot.control_period_s, np.asarray(env._active_indices)
                )
                np.testing.assert_allclose(
                    trace["reward_components"].sum(axis=-1),
                    trace["reward"],
                    rtol=1e-5,
                    atol=1e-6,
                )
                # Both computations must describe the same FIRST episode only.
                np.testing.assert_allclose(
                    summary["avg_episode_length"],
                    np.mean(evaluation.episode_steps),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    summary["fall_count"],
                    np.sum(evaluation.episode_metrics["fall"]),
                    atol=1e-6,
                )
                np.savez_compressed(
                    args.output / f"{label}.npz",
                    **trace,
                    model_keys=np.asarray(model_keys),
                    unroll_key=np.asarray(unroll_key),
                )
                batch = {
                    "label": label,
                    "checkpoint": str(path.resolve()),
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "case": case,
                    "seed": seed,
                    **summary,
                }
                report["batches"].append(batch)
                (args.output / "summary.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(json.dumps(batch), flush=True)
            jax.clear_caches()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(707, 808, 909))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--command", type=float, default=0.10)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("full", "dynamics", "noise", "nominal"),
        default=["full", "dynamics", "noise", "nominal"],
    )
    parser.add_argument("--restart-step", type=int, default=600)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--project-actor", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
