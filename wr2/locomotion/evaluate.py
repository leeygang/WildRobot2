"""Run independent fixed, scripted, or random-command WR2 evaluations."""

from __future__ import annotations

import argparse
import functools
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from wr2.locomotion.configs import DEFAULT_TRAINING_CONFIG_PATH, load_training_config
from wr2.locomotion.train import (
    _configure_backend_logging,
    _filter_optional_backend_import_messages,
)
from wr2.locomotion.walking_metrics import walking_score


def transition_environment(config, *, add_observation_noise):
    """Repeat standing -> walking -> standing at deterministic command times."""
    import jax.numpy as jp
    from wr2.locomotion.walking_env import WR2WalkingEnv

    if config.episode_length != 1000:
        raise ValueError("The standing/walking schedule requires episode_length=1000")

    class TransitionEnvironment(WR2WalkingEnv):
        def _next_command(self, step_count, command, rng):
            # 3 s stand, 6 s walk, 3 s stand, 5 s walk, 3 s stand.
            walking = ((step_count >= 150) & (step_count < 450)) | (
                (step_count >= 600) & (step_count < 850)
            )
            return jp.where(
                walking,
                jp.asarray([config.command_forward_range_m_s[0], 0.0, 0.0]),
                jp.zeros(3),
            )

    return TransitionEnvironment(
        replace(config, zero_command_probability=1.0),
        add_observation_noise=add_observation_noise,
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return seeds


def _parse_commands(value: str) -> tuple[float, ...]:
    try:
        commands = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "commands must be comma-separated numbers"
        ) from exc
    if (
        not commands
        or any(command <= 0.0 for command in commands)
        or len(set(commands)) != len(commands)
    ):
        raise argparse.ArgumentTypeError("commands must be unique and positive")
    return commands


def _conditional_tracking(metrics: dict) -> dict:
    """Use active sample counts, not standing-diluted episode means."""
    tracking = {}
    for category in ("standing", "walking", "start", "stop"):
        count = float(metrics[f"eval/episode_{category}_sample_count"])
        error_sum = float(metrics[f"eval/episode_{category}_velocity_error_sum"])
        tracking[category] = {
            "samples_per_episode": count,
            "velocity_error_m_s": error_sum / count if count > 0 else None,
        }
    walking_count = tracking["walking"]["samples_per_episode"]
    velocity_sum = sum(
        float(metrics[f"eval/episode_phase{index}_velocity_sum"])
        for index in range(4)
    )
    tracking["walking"]["forward_velocity_m_s"] = (
        velocity_sum / walking_count if walking_count > 0 else None
    )
    return tracking


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    config_path: Path,
    seeds: Sequence[int],
    forward_commands_m_s: Sequence[float] | None,
    num_envs: int,
    allow_cpu: bool,
    transitions: bool = False,
    random_commands: bool = False,
    solver_iterations: int | None = None,
) -> dict:
    """Evaluate one checkpoint independently for every requested seed."""
    if random_commands and (transitions or forward_commands_m_s is not None):
        raise ValueError("random_commands cannot be combined with transitions or commands")
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        from brax.envs import training as env_training
        from brax.training import acting, types
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import checkpoint
        from brax.training.agents.ppo import networks as ppo_networks

        from wr2.locomotion.domain_randomization import make_domain_randomizer
        from wr2.locomotion.ppo import make_network_factory
        from wr2.locomotion.rsl_rl_training import (
            _jax_policy_factory,
            load_rsl_actor_parameters,
        )
        from wr2.locomotion.walking_env import WR2WalkingEnv

    if jax.default_backend() != "gpu" and not allow_cpu:
        raise RuntimeError(
            f"JAX backend is {jax.default_backend()!r}; pass --allow-cpu only "
            "for an intentional development evaluation"
        )
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    config = load_training_config(config_path)
    if solver_iterations is not None:
        if solver_iterations < 1:
            raise ValueError("solver_iterations must be positive")
        config = replace(
            config,
            environment=replace(config.environment, solver_iterations=solver_iterations),
        )
    rsl_checkpoint = checkpoint_path.is_file()
    if rsl_checkpoint:
        params = load_rsl_actor_parameters(checkpoint_path)
        eval_policy_factory = _jax_policy_factory(config.network.activation)
    else:
        eval_environment_config = replace(
            config.environment,
            command_forward_range_m_s=(
                config.ppo.evaluation_forward_command_m_s,
                config.ppo.evaluation_forward_command_m_s,
            ),
            zero_command_probability=0.0,
        )
        network_environment = WR2WalkingEnv(
            eval_environment_config, add_observation_noise=False
        )
        network_factory = make_network_factory(config.network)
        preprocess = (
            running_statistics.normalize
            if config.ppo.normalize_observations
            else types.identity_observation_preprocessor
        )
        networks = network_factory(
            network_environment.observation_size,
            network_environment.action_size,
            preprocess_observations_fn=preprocess,
        )
        make_policy = ppo_networks.make_inference_fn(networks)
        params = checkpoint.load(str(checkpoint_path))
        eval_policy_factory = functools.partial(make_policy, deterministic=True)
    commands = (None,) if random_commands else tuple(
        forward_commands_m_s or (config.ppo.evaluation_forward_command_m_s,)
    )
    randomizer = (
        make_domain_randomizer(config.environment.randomization)
        if config.environment.randomization.enabled
        else None
    )

    command_results = []
    for command_index, command_forward_m_s in enumerate(commands):
        environment = WR2WalkingEnv(
            config.environment if random_commands else replace(
                config.environment,
                command_forward_range_m_s=(
                    command_forward_m_s,
                    command_forward_m_s,
                ),
                zero_command_probability=0.0,
            ),
            # Independent confirmation includes configured observation noise,
            # model variation, and reset/local actuator variation. Training
            # progress disables the first two, not local actuator variation.
            add_observation_noise=config.environment.randomization.enabled,
        )
        if transitions:
            environment = transition_environment(
                environment.config,
                add_observation_noise=config.environment.randomization.enabled,
            )
        seed_results = []
        for seed in seeds:
            keyed_seed = int(seed) + 100_000 * command_index
            wrap_key, eval_key = jax.random.split(jax.random.PRNGKey(keyed_seed))
            randomization_fn = None
            if randomizer is not None:
                randomization_fn = functools.partial(
                    randomizer, rng=jax.random.split(wrap_key, num_envs)
                )
            wrapped = env_training.wrap(
                environment,
                episode_length=config.environment.episode_length,
                action_repeat=1,
                randomization_fn=randomization_fn,
            )
            evaluator = acting.Evaluator(
                wrapped,
                eval_policy_factory,
                num_eval_envs=num_envs,
                episode_length=config.environment.episode_length,
                action_repeat=1,
                key=eval_key,
            )
            raw_metrics = evaluator.run_evaluation(params, {})
            metrics = {
                name: np.asarray(value).tolist() for name, value in raw_metrics.items()
            }
            score = None if transitions or random_commands else walking_score(
                episode_length=float(metrics["eval/avg_episode_length"]),
                target_episode_length=config.environment.episode_length,
                velocity_error_m_s=float(
                    metrics["eval/episode_forward_velocity_error_m_s_per_step"]
                ),
                velocity_sigma_m_s=config.environment.velocity_tracking_sigma,
                contact_match=float(
                    metrics["eval/episode_contact_phase_match_per_step"]
                ),
                double_support=float(metrics["eval/episode_double_support_per_step"]),
            )
            result = {
                "seed": int(seed),
                "walking_score": score,
                "conditional_tracking": _conditional_tracking(metrics),
                "metrics": metrics,
            }
            seed_results.append(result)
            command_label = "random" if random_commands else f"{command_forward_m_s:.3f}"
            score_label = "n/a" if score is None else f"{score:.3f}"
            error = result["conditional_tracking"]["walking"]["velocity_error_m_s"]
            error_label = "n/a" if error is None else f"{error:.3f}"
            print(
                f"command={command_label} seed={seed} "
                f"walking_score={score_label} "
                f"episode_length={float(metrics['eval/avg_episode_length']):.1f} "
                "walking_velocity_error="
                f"{error_label} "
                f"fall={float(metrics['eval/episode_fall']):.1%}",
                flush=True,
            )
        command_results.append(
            {
                "command_forward_m_s": command_forward_m_s,
                "seed_results": seed_results,
            }
        )

    return {
        "schema_version": 2,
        "checkpoint": str(checkpoint_path),
        "backend": "rsl_rl" if rsl_checkpoint else "brax",
        "config": str(config_path.resolve()),
        "solver_iterations": config.environment.solver_iterations,
        "commands_forward_m_s": [] if random_commands else list(commands),
        "training_command_distribution": {
            "forward_range_m_s": list(config.environment.command_forward_range_m_s),
            "standing_probability": config.environment.zero_command_probability,
            "resample_steps": config.environment.command_resample_steps,
        },
        "num_envs_per_seed": num_envs,
        "evaluation_mode": (
            "randomized_held_out"
            if config.environment.randomization.enabled
            else "nominal"
        ),
        "variation": {
            "observation_noise": config.environment.randomization.enabled,
            "model_randomization": config.environment.randomization.enabled,
            "reset_and_local_actuator": config.environment.randomization.enabled,
        },
        "seeds": list(seeds),
        "command_schedule": "stand150_walk300_stand150_walk250_stand150"
        if transitions
        else ("random_training_distribution" if random_commands else "fixed"),
        "command_results": command_results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--seeds", type=_parse_seeds, default=(101, 202, 303))
    parser.add_argument("--commands", type=_parse_commands)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--solver-iterations", type=int)
    schedules = parser.add_mutually_exclusive_group()
    schedules.add_argument(
        "--transitions",
        action="store_true",
        help="Evaluate scripted standing/walking transitions over 1000 steps",
    )
    schedules.add_argument(
        "--random-commands",
        action="store_true",
        help="Use the config's training command distribution and resampling cadence",
    )
    args = parser.parse_args(argv)
    if args.random_commands and args.commands is not None:
        parser.error("--random-commands cannot be combined with --commands")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    report = evaluate_checkpoint(
        args.checkpoint,
        config_path=args.config,
        seeds=args.seeds,
        forward_commands_m_s=args.commands,
        num_envs=args.num_envs,
        allow_cpu=args.allow_cpu,
        transitions=args.transitions,
        random_commands=args.random_commands,
        solver_iterations=args.solver_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote evaluation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
