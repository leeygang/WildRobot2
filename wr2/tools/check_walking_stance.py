"""Cross old/new actors at walking starts without replacing simulator state."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.evaluate import _parse_seeds
from wr2.locomotion.train import (
    _configure_backend_logging,
    _filter_optional_backend_import_messages,
)
from wr2.tools.investigate_walking import (
    evaluation_keys,
    recording_environment,
    summarize_trace,
)


def switched_unroll(env, state, donor, receiver, key, switch_step, length):
    """Change only actor weights at a control boundary, keeping the full carry.

    Match Brax generate_unroll's key splitting. Retain physics warm-start,
    velocities, IMU/RNG state, all observation frames and delayed donor actions.
    The first receiver action therefore still incurs the configured delay.
    """
    import jax
    from brax.training import acting

    def step(carry, index):
        before, current_key = carry
        action_key, next_key = jax.random.split(current_key)

        def policy(observation, policy_key):
            return jax.lax.cond(
                index < switch_step,
                lambda _: donor(observation, policy_key),
                lambda _: receiver(observation, policy_key),
                operand=None,
            )

        after, transition = acting.actor_step(
            env,
            before,
            policy,
            action_key,
            extra_fields=("diagnostic", "episode_done"),
        )
        return (after, next_key), transition.extras["state_extras"]

    (final, final_key), extras = jax.lax.scan(step, (state, key), np.arange(length))
    return final, final_key, extras


def foot_geometry(trace):
    """Signed left-minus-right offsets in the robot yaw frame, in metres.

    +x means left ahead of right; +y means left lateral to right. Do not take
    absolute values before comparing starts: that hides which foot is ahead.
    Site positions are the recorded last-substep samples, as in the main replay.
    """
    w, x, y, z = np.moveaxis(np.asarray(trace["qpos"])[..., 3:7], -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    feet = np.asarray(trace["foot_position_m"])
    delta = feet[..., 0, :] - feet[..., 1, :]
    c, s = np.cos(yaw), np.sin(yaw)
    return c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]


def stance_summary(trace, start, dt):
    """Separate pre-start geometry from falls in the next five seconds."""
    dx, dy = foot_geometry(trace)
    forces = np.asarray(trace["foot_force_path_n"])[..., 2]
    total = forces.sum(axis=-1)
    supported = total > 1.0
    left_share = forces[..., 0] / np.maximum(total, 1.0)
    done = np.asarray(trace["episode_done"], dtype=bool)
    valid = (np.cumsum(done, axis=0) - done) == 0
    pre = slice(start - round(1 / dt), start)
    window = slice(start, min(start + round(5 / dt), len(done)))
    mask = valid[pre] & supported[pre]
    falls = (trace["fall"][window] > 0) & valid[window]
    nonfinite = (trace["nonfinite"][window] > 0) & valid[window]
    eligible = np.flatnonzero(valid[start])

    def pre_mean(values):
        return float(np.mean(values[pre][mask])) if np.any(mask) else None

    return {
        "start_step": start,
        "time_s": start * dt,
        "alive_at_start_count": int(eligible.size),
        "phase_before_start_deg": float(
            np.rad2deg(trace["phase_rad"][start - 1, eligible[0]])
        )
        if eligible.size
        else None,
        "pre_start_left_load_share": pre_mean(left_share),
        "pre_start_signed_left_minus_right_x_mm": pre_mean(1000 * dx),
        "pre_start_abs_forward_offset_mm": pre_mean(1000 * np.abs(dx)),
        "pre_start_lateral_width_mm": pre_mean(1000 * dy),
        "fall_count_next_5s": int(np.count_nonzero(falls.any(axis=0))),
        "fall_episode_indices_next_5s": np.flatnonzero(falls.any(axis=0)).tolist(),
        "nonfinite_count_next_5s": int(np.count_nonzero(nonfinite.any(axis=0))),
    }


def assert_matching_prefix(trace, reference, length):
    """Every recorded field must agree before the actor handover."""
    for name in trace:
        np.testing.assert_array_equal(
            trace[name][:length],
            reference[name][:length],
            err_msg=f"Unmatched donor prefix: {name}",
        )


def run(args):
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        from brax.envs import training as env_training
        from brax.envs.wrappers.training import EvalWrapper
        from wr2.locomotion.domain_randomization import make_domain_randomizer
        from wr2.locomotion.rsl_rl_training import (
            _jax_policy_factory,
            load_rsl_actor_parameters,
        )

    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise RuntimeError("Use --allow-cpu for intentional Mac/development replay")
    if args.num_envs < 1 or args.solver_iterations < 1:
        raise ValueError("num-envs and solver-iterations must be positive")
    config = load_training_config(args.config)
    base = replace(
        config.environment,
        solver_iterations=args.solver_iterations,
        command_forward_range_m_s=(args.command, args.command),
    )
    env = recording_environment(base, noise=True)
    policy_factory = _jax_policy_factory(config.network.activation)
    parameters = [load_rsl_actor_parameters(path) for path in args.checkpoints]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(
            (env.robot.directory / "wr2_mjx.xml").read_bytes()
        ).hexdigest(),
        "solver_iterations": args.solver_iterations,
        "num_envs": args.num_envs,
        "seeds": args.seeds,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
        "checkpoint_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in args.checkpoints
        ],
        "actuator_names": list(env.robot.actuator_names),
        "intervention": "actor_weights_only_full_simulator_and_input_history_retained",
        "limitations": "Donor states differ in pose AND history; this is not a geometry-only or phase-only test. Policy handover can itself introduce a transient.",
        "force_sampling": "last_physics_substep_not_control_period_impulse",
        "batches": [],
    }
    for seed in args.seeds:
        model_keys, key = evaluation_keys(seed, args.num_envs)
        randomizer = make_domain_randomizer(base.randomization)
        wrapped = EvalWrapper(
            env_training.wrap(
                env,
                episode_length=base.episode_length,
                action_repeat=1,
                randomization_fn=functools.partial(randomizer, rng=model_keys),
            )
        )

        def rollout(donor, receiver, switch_step, reset_key):
            initial = wrapped.reset(jax.random.split(reset_key, args.num_envs))
            final, _, extras = switched_unroll(
                wrapped,
                initial,
                policy_factory(donor),
                policy_factory(receiver),
                reset_key,
                switch_step,
                base.episode_length,
            )
            return final.info["eval_metrics"], extras

        compiled = jax.jit(rollout)
        # Diagonal controls first; crossing at 150/600 preserves each donor's
        # exact first/second pre-start state. No qpos-only reconstruction.
        jobs = [(0, 0, 150), (1, 1, 150)] + [
            (donor, 1 - donor, start) for donor in (0, 1) for start in (150, 600)
        ]
        controls = {}
        for donor, receiver, start in jobs:
            label = f"donor{donor}_actor{receiver}_start{start}_seed{seed}"
            print(f"Matched stance {label}: {args.num_envs} episodes", flush=True)
            evaluation, extras = jax.tree.map(
                np.asarray,
                compiled(parameters[donor], parameters[receiver], start, key),
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
            reference_verified = False
            if donor == receiver:
                controls[donor] = trace
                if args.reference_traces:
                    path = args.reference_traces / f"policy{donor}_full_seed{seed}.npz"
                    with np.load(path) as reference:
                        # Independent original generate_unroll replay, not a
                        # self-generated reference for the new switching code.
                        for name in trace:
                            np.testing.assert_allclose(
                                trace[name],
                                reference[name],
                                rtol=1e-5,
                                atol=1e-6,
                                err_msg=name,
                            )
                    reference_verified = True
            else:
                if np.any(controls[donor]["episode_done"][:start]):
                    raise RuntimeError("Donor terminated before the handover")
                assert_matching_prefix(trace, controls[donor], start)
            np.savez_compressed(
                args.output / f"{label}.npz",
                **trace,
                model_keys=np.asarray(model_keys),
                unroll_key=np.asarray(key),
            )
            batch = {
                "label": label,
                "donor": donor,
                "receiver": receiver,
                "switch_step": start,
                "seed": seed,
                "original_replay_verified": reference_verified,
                "donor_prefix_verified": donor != receiver,
                "starts": [
                    stance_summary(trace, step, env.robot.control_period_s)
                    for step in (150, 600)
                ],
                **summary,
            }
            report["batches"].append(batch)
            (args.output / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {k: v for k, v in batch.items() if k != "reward_component_returns"}
                ),
                flush=True,
            )
        jax.clear_caches()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", type=Path, nargs=2, required=True, metavar=("OLDER", "NEWER")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-traces", type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=(707,))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--command", type=float, default=0.10)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--allow-cpu", action="store_true")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
