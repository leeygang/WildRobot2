"""Smoke-test or train WR2's initial Brax PPO walking environment."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from wr2.locomotion.configs import (
    DEFAULT_TRAINING_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
    training_config_to_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def smoke_test(steps: int, training_config: TrainingConfig | None = None) -> None:
    import jax
    import jax.numpy as jp

    from wr2.locomotion.walking_env import WR2WalkingEnv

    training_config = training_config or load_training_config()
    environment = WR2WalkingEnv(
        training_config.environment, add_observation_noise=False
    )
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    state = reset(jax.random.PRNGKey(0))
    jax.block_until_ready(state.obs)
    started = time.monotonic()
    for _ in range(steps):
        state = step(state, jp.zeros(environment.action_size))
    jax.block_until_ready(state.obs)
    elapsed = time.monotonic() - started
    print(
        f"MJX smoke passed: steps={steps}, obs={state.obs.shape}, "
        f"action={environment.action_size}, reward={float(state.reward):.6f}, "
        f"done={float(state.done):.0f}, root_z={float(state.pipeline_state.q[2]):.6f}, "
        f"elapsed={elapsed:.3f}s"
    )


def _create_run_directory(
    output_root: Path,
    requested_run_id: str | None,
    seed: int,
    run_prefix: str = "wr2_ppo",
) -> tuple[str, Path]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if requested_run_id is not None:
        if Path(requested_run_id).name != requested_run_id or requested_run_id in {
            ".",
            "..",
        }:
            raise ValueError("Run ID must be a single directory name")
        output = output_root / requested_run_id
        output.mkdir(exist_ok=False)
        return requested_run_id, output

    base_run_id = f"{run_prefix}_{datetime.now().astimezone():%Y%m%d_%H%M%S}_seed{seed}"
    for suffix in range(100):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
        output = output_root / run_id
        try:
            output.mkdir()
        except FileExistsError:
            continue
        return run_id, output
    raise RuntimeError(f"Could not allocate a run directory under {output_root}")


def train(args: argparse.Namespace, training_config: TrainingConfig) -> None:
    import jax
    import numpy as np
    from brax.io import model
    from brax.training.agents.ppo import train as ppo

    from wr2.locomotion.domain_randomization import make_domain_randomizer
    from wr2.locomotion.walking_env import WR2WalkingEnv

    environment = WR2WalkingEnv(
        training_config.environment,
        add_observation_noise=True,
    )
    output_root = Path(training_config.output.root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_id, output = _create_run_directory(
        output_root,
        args.run_id,
        training_config.seed,
        training_config.output.run_prefix,
    )
    print(f"Run ID: {run_id}")
    print(f"Output: {output}")
    effective_config = training_config_to_dict(training_config)
    (output / "training_config.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
    )
    run_config = {
        "run_id": run_id,
        "output": str(output),
        "source_config": str(args.config.resolve()),
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "training_config": effective_config,
        "actuator_model": environment.robot.config["actuators"]["htd45hServo"],
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = output / "training_metrics.jsonl"
    metrics_path.write_text("")

    def progress(step: int, metrics) -> None:
        serializable_metrics = {
            name: np.asarray(value).tolist() for name, value in metrics.items()
        }
        with metrics_path.open("a") as stream:
            stream.write(
                json.dumps({"step": step, "metrics": serializable_metrics}) + "\n"
            )

        console_metrics = {
            "eval_reward": metrics.get(
                "eval/episode_reward", metrics.get("eval/episode_reward_mean")
            ),
            "torque_rms_nm": metrics.get(
                "eval/episode_actuator_torque_rms_nm_per_step"
            ),
            "mean_step_peak_nm": metrics.get(
                "eval/episode_actuator_torque_peak_nm_per_step"
            ),
            "torque_exposure": metrics.get(
                "eval/episode_sustained_torque_exposure_per_step"
            ),
        }
        summary = " ".join(
            f"{name}={float(np.asarray(value)):.6g}"
            for name, value in console_metrics.items()
            if value is not None
        )
        print(f"step={step} {summary}")

    randomization_fn = (
        None
        if not training_config.environment.randomization.enabled
        else make_domain_randomizer(environment.config.randomization)
    )
    ppo_config = training_config.ppo
    make_inference_fn, params, _ = ppo.train(
        environment=environment,
        num_timesteps=ppo_config.num_timesteps,
        num_envs=ppo_config.num_envs,
        episode_length=environment.config.episode_length,
        num_evals=ppo_config.num_evals,
        num_eval_envs=ppo_config.num_eval_envs,
        learning_rate=ppo_config.learning_rate,
        entropy_cost=ppo_config.entropy_cost,
        discounting=ppo_config.discounting,
        unroll_length=ppo_config.unroll_length,
        batch_size=ppo_config.batch_size,
        num_minibatches=ppo_config.num_minibatches,
        num_updates_per_batch=ppo_config.num_updates_per_batch,
        normalize_observations=ppo_config.normalize_observations,
        randomization_fn=randomization_fn,
        seed=training_config.seed,
        progress_fn=progress,
    )
    del make_inference_fn
    jax.block_until_ready(params)
    model.save_params(output / "params", params)
    print(f"Saved parameters to {output / 'params'}")


def _apply_cli_overrides(
    training_config: TrainingConfig, args: argparse.Namespace
) -> TrainingConfig:
    ppo_fields = (
        "num_timesteps",
        "num_envs",
        "num_evals",
        "num_eval_envs",
        "unroll_length",
        "batch_size",
        "num_minibatches",
        "num_updates_per_batch",
        "learning_rate",
        "entropy_cost",
        "discounting",
    )
    ppo_overrides = {
        name: getattr(args, name)
        for name in ppo_fields
        if getattr(args, name) is not None
    }
    ppo_config = replace(training_config.ppo, **ppo_overrides)

    environment = training_config.environment
    if args.episode_length is not None:
        environment = replace(environment, episode_length=args.episode_length)
    if args.no_domain_randomization:
        environment = replace(
            environment,
            randomization=replace(environment.randomization, enabled=False),
        )
    output = training_config.output
    if args.output_root is not None:
        output = replace(output, root=str(args.output_root))
    return replace(
        training_config,
        seed=training_config.seed if args.seed is None else args.seed,
        environment=environment,
        ppo=ppo_config,
        output=output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_PATH,
        help="Training YAML; explicit CLI options override its values",
    )
    parser.add_argument("--smoke", action="store_true", help="Run JIT reset/step only")
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-evals", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--unroll-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--num-updates-per-batch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-cost", type=float, default=None)
    parser.add_argument("--discounting", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument(
        "--output-root",
        "--output",
        dest="output_root",
        type=Path,
        default=None,
        help="Override the config's parent directory for automatically named runs",
    )
    parser.add_argument(
        "--run-id",
        help="Optional explicit run ID; defaults to a timestamped ID",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Loading training config: {args.config}")
    training_config = _apply_cli_overrides(load_training_config(args.config), args)
    print(
        f"Training setup: version={training_config.version} "
        f"seed={training_config.seed} "
        f"timesteps={training_config.ppo.num_timesteps} "
        f"envs={training_config.ppo.num_envs}"
    )
    if args.smoke:
        smoke_test(args.smoke_steps, training_config)
    else:
        train(args, training_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
