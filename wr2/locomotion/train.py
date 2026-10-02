"""Smoke-test or train WR2's phase-guided Brax PPO walking environment."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import warnings
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
from wr2.locomotion.walking_metrics import walking_score as calculate_walking_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_KNOWN_OPTIONAL_IMPORT_MESSAGES = (
    "Failed to import warp:",
    "Failed to import mujoco_warp:",
)


def _configure_backend_logging() -> None:
    """Hide known backend noise while preserving errors and WR2 progress."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    warnings.filterwarnings(
        "ignore",
        message=r"Brax System, piplines and environments are not actively being.*",
        category=UserWarning,
        module=r"brax\.io\.mjcf",
    )


@contextlib.contextmanager
def _filter_optional_backend_import_messages():
    """Suppress optional Warp import notices without hiding other output."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield
    except BaseException:
        sys.stdout.write(stdout.getvalue())
        sys.stderr.write(stderr.getvalue())
        raise
    else:
        for stream, destination in ((stdout, sys.stdout), (stderr, sys.stderr)):
            for line in stream.getvalue().splitlines():
                if not line.startswith(_KNOWN_OPTIONAL_IMPORT_MESSAGES):
                    print(line, file=destination)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _acquisition_checkpoint_score(
    *,
    episode_length: float,
    target_episode_length: int,
    fall_rate: float,
    command_forward_m_s: float,
    forward_velocity_m_s: float,
    contact_match: float,
    double_support: float,
    action_saturation: float,
    nonfinite_state: float,
    mean_step_peak_torque_nm: float,
) -> float:
    """Rank safe gait-acquisition checkpoints without rewarding standing."""
    if (
        nonfinite_state > 0.0
        or action_saturation > 0.15
        or mean_step_peak_torque_nm > 4.0
    ):
        return -math.inf
    survival = min(max(episode_length / target_episode_length, 0.0), 1.0)
    survival *= 1.0 - min(max(fall_rate, 0.0), 1.0)
    velocity_ratio = min(
        max(forward_velocity_m_s, 0.0) / max(command_forward_m_s, 1e-6),
        1.0,
    )
    contact_progress = min(max((contact_match - 0.5) / 0.5, 0.0), 1.0)
    single_support_progress = 1.0 - min(max(double_support, 0.0), 1.0)
    return survival * (
        0.15
        + 0.55 * velocity_ratio
        + 0.20 * contact_progress
        + 0.10 * single_support_progress
    )


def smoke_test(steps: int, training_config: TrainingConfig | None = None) -> None:
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax

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
        state = step(state, environment.home_action)
    jax.block_until_ready(state.obs)
    elapsed = time.monotonic() - started
    print(
        "MJX smoke passed: "
        f"steps={steps}, actor_obs={state.obs['state'].shape}, "
        f"critic_obs={state.obs['privileged_state'].shape}, "
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
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax
        import numpy as np
        from brax.io import model
        from brax.training.agents.ppo import train as ppo

        from wr2.locomotion.domain_randomization import make_domain_randomizer
        from wr2.locomotion.ppo import make_network_factory
        from wr2.locomotion.walking_env import WR2WalkingEnv

    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu" and not args.allow_cpu:
        raise RuntimeError(
            f"JAX backend is {backend!r}, not 'gpu' (devices={devices}). "
            "Fix the CUDA runtime or pass --allow-cpu for an intentional CPU run."
        )
    if args.restore_checkpoint is not None and not args.restore_checkpoint.exists():
        raise FileNotFoundError(
            f"Restore checkpoint does not exist: {args.restore_checkpoint}"
        )

    environment = WR2WalkingEnv(
        training_config.environment,
        add_observation_noise=True,
    )
    evaluation_environment = WR2WalkingEnv(
        replace(
            training_config.environment,
            command_forward_range_m_s=(
                training_config.ppo.evaluation_forward_command_m_s,
                training_config.ppo.evaluation_forward_command_m_s,
            ),
            zero_command_probability=0.0,
        ),
        add_observation_noise=False,
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
    print("=" * 72)
    print("WR2 PPO training")
    print("=" * 72)
    print(f"  Run ID:       {run_id}")
    print(f"  Output:       {output}")
    print(f"  JAX backend:  {backend}")
    print(f"  Devices:      {devices}")
    print(
        "  Observations: "
        f"actor={environment.observation_size['state']} "
        f"critic={environment.observation_size['privileged_state']}"
    )
    print(f"  Actions:      {environment.action_size} (leg joints active)")
    print(
        "  Action map:   absolute joint range; "
        f"max |walk_home|={float(np.max(np.abs(environment.home_action))):.3f}; "
        f"initial latent std={training_config.network.init_noise_std:.3f}"
    )
    print(f"  Environments: {training_config.ppo.num_envs:,}")
    print(f"  Target steps: {training_config.ppo.num_timesteps:,}")
    print(f"  Episode:      {training_config.environment.episode_length} steps")
    print(
        "  Evaluation:   "
        f"{training_config.ppo.num_evals} evaluations x "
        f"{training_config.ppo.num_eval_envs} parallel episodes; "
        f"fixed {training_config.ppo.evaluation_forward_command_m_s:.2f}m/s forward"
    )
    print(
        "  Gait:         "
        f"{training_config.environment.gait_cycle_s:.2f}s cycle, "
        f"{training_config.environment.swing_height_m:.3f}m swing"
    )
    zmp = environment.zmp_reference_diagnostics
    print(
        "  ZMP critic:   "
        f"{len(zmp.command_samples_m_s)} command x "
        f"{training_config.environment.zmp_reference.phase_samples} phase lookup; "
        f"IK residual={zmp.max_ik_position_residual_m * 1000.0:.2f}mm"
    )
    print(
        "  Network:      "
        f"policy={list(training_config.network.policy_hidden_layer_sizes)} "
        f"value={list(training_config.network.value_hidden_layer_sizes)} "
        f"activation={training_config.network.activation} "
        f"distribution={training_config.network.distribution_type}"
    )
    print(f"  Randomized:   {training_config.environment.randomization.enabled}")
    if args.restore_checkpoint is not None:
        print(f"  Restore:      {args.restore_checkpoint.resolve()}")
    print("=" * 72)
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
        "active_home_action": np.asarray(environment.home_action).tolist(),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = output / "training_metrics.jsonl"
    metrics_path.write_text("")
    rollout_metrics_path = output / "rollout_metrics.jsonl"
    rollout_metrics_path.write_text("")

    training_started = time.monotonic()
    progress_index = 0
    latest_params: dict[str, object] = {}
    best_selection_score = -math.inf
    pending_checkpoints: dict[int, dict[str, object]] = {}

    def save_best_checkpoint(params, candidate: dict[str, object]) -> None:
        nonlocal best_selection_score
        selection_score = float(candidate["selection_score"])
        if selection_score <= best_selection_score:
            return
        best_selection_score = selection_score
        model.save_params(output / "best_params", params)
        (output / "best_checkpoint.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "  └─ saved : new best "
            f"{candidate['selection_metric']} checkpoint "
            f"at step {int(candidate['step']):,} (score={selection_score:.3f})",
            flush=True,
        )

    def receive_policy_params(step: int, _make_policy, params) -> None:
        step = int(step)
        latest_params["step"] = step
        latest_params["value"] = params
        candidate = pending_checkpoints.pop(step, None)
        if candidate is not None:
            save_best_checkpoint(params, candidate)

    def progress(step: int, metrics) -> None:
        nonlocal best_selection_score, progress_index
        elapsed_s = time.monotonic() - training_started
        step = int(step)
        target_steps = training_config.ppo.num_timesteps
        progress_fraction = min(max(step / target_steps, 0.0), 1.0)

        def metric(name: str) -> float | None:
            value = metrics.get(name)
            if value is None:
                return None
            array = np.asarray(value)
            return float(array) if array.size == 1 else None

        serializable_metrics = {
            name: np.asarray(value).tolist() for name, value in metrics.items()
        }
        if "eval/avg_episode_length" not in metrics:
            with rollout_metrics_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "elapsed_s": elapsed_s,
                            "metrics": serializable_metrics,
                        }
                    )
                    + "\n"
                )

            def rollout_show(name: str, format_spec: str = ".3f") -> str:
                value = metric(f"episode/{name}")
                return "n/a" if value is None else format(value, format_spec)

            print(
                f"Rollout [{_format_duration(elapsed_s)}] steps={step:,} | "
                f"ep_len={rollout_show('length', '.0f')} "
                f"vx={rollout_show('forward_velocity_m_s_per_step')} "
                f"contact={rollout_show('contact_phase_match_per_step', '.1%')} "
                f"sat={rollout_show('action_saturation_fraction_per_step', '.1%')} "
                f"near={rollout_show('action_near_boundary_fraction_per_step', '.1%')} "
                f"|a-home|={rollout_show('action_deviation_from_home_abs_mean_per_step')}",
                flush=True,
            )
            return

        steps_per_second = metric("training/sps")
        if steps_per_second is None and step > 0 and elapsed_s > 0.0:
            steps_per_second = step / elapsed_s
        eta_s = (
            max(target_steps - step, 0) / steps_per_second
            if steps_per_second is not None and steps_per_second > 0.0
            else None
        )
        episode_return = metric("eval/episode_reward")
        reward_per_step = metric("eval/episode_reward_per_step")
        episode_length = metric("eval/avg_episode_length")
        velocity_error = metric("eval/episode_forward_velocity_error_m_s_per_step")
        contact_match = metric("eval/episode_contact_phase_match_per_step")
        double_support = metric("eval/episode_double_support_per_step")
        walking_score = None
        if (
            episode_length is not None
            and velocity_error is not None
            and contact_match is not None
            and double_support is not None
        ):
            walking_score = calculate_walking_score(
                episode_length=episode_length,
                target_episode_length=training_config.environment.episode_length,
                velocity_error_m_s=velocity_error,
                velocity_sigma_m_s=training_config.environment.velocity_tracking_sigma,
                contact_match=contact_match,
                double_support=double_support,
            )
        with metrics_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "evaluation": progress_index,
                        "step": step,
                        "progress_fraction": progress_fraction,
                        "elapsed_s": elapsed_s,
                        "eta_s": eta_s,
                        "walking_score": walking_score,
                        "metrics": serializable_metrics,
                    }
                )
                + "\n"
            )
        throughput = (
            f"{steps_per_second:,.0f}" if steps_per_second is not None else "warming-up"
        )
        print(
            f"Eval {progress_index + 1:>2}/{training_config.ppo.num_evals} "
            f"[{_format_duration(elapsed_s)}] "
            f"Steps: {step:>10,}/{target_steps:,} ({100.0 * progress_fraction:5.1f}%) "
            f"ETA {_format_duration(eta_s)} | steps/s={throughput}",
            flush=True,
        )

        def show(value: float | None, format_spec: str = ".3f") -> str:
            return "n/a" if value is None else format(value, format_spec)

        print(
            "  └─ return: "
            f"episode={show(episode_return, '.2f')} "
            f"reward/step={show(reward_per_step)} "
            f"avg_ep_len={show(episode_length, '.0f')} "
            f"walking_score={show(walking_score)}",
            flush=True,
        )
        print(
            "  └─ track : "
            f"cmd_vx={show(metric('eval/episode_command_forward_m_s_per_step'))} "
            f"vx={show(metric('eval/episode_forward_velocity_m_s_per_step'))} "
            f"|vx-cmd|={show(metric('eval/episode_forward_velocity_error_m_s_per_step'))} "
            f"vy={show(metric('eval/episode_lateral_velocity_m_s_per_step'))} "
            f"yaw_err={show(metric('eval/episode_yaw_rate_error_rad_s_per_step'))}",
            flush=True,
        )
        tilt_rad = metric("eval/episode_torso_tilt_rad_per_step")
        print(
            "  └─ state : "
            f"height={show(metric('eval/episode_torso_height_m_per_step'))}m "
            f"tilt={show(None if tilt_rad is None else math.degrees(tilt_rad), '.1f')}deg "
            f"fall={show(metric('eval/episode_fall'), '.1%')} "
            f"feet=L{show(metric('eval/episode_left_foot_contact_per_step'), '.0%')}"
            f"/R{show(metric('eval/episode_right_foot_contact_per_step'), '.0%')}",
            flush=True,
        )
        print(
            "  └─ gait  : "
            f"phase_score={show(metric('eval/episode_feet_phase_tracking_per_step'))} "
            f"contact_match={show(metric('eval/episode_contact_phase_match_per_step'), '.1%')} "
            f"double_support={show(metric('eval/episode_double_support_per_step'), '.1%')} "
            f"foot_z=L{show(metric('eval/episode_left_foot_height_m_per_step'))}m"
            f"/R{show(metric('eval/episode_right_foot_height_m_per_step'))}m "
            f"width={show(metric('eval/episode_feet_lateral_distance_m_per_step'))}m",
            flush=True,
        )
        print(
            "  └─ servo : "
            f"rms={show(metric('eval/episode_actuator_torque_rms_nm_per_step'))}Nm "
            f"mean_peak={show(metric('eval/episode_actuator_torque_peak_nm_per_step'))}Nm "
            f"exposure={show(metric('eval/episode_sustained_torque_exposure_per_step'))} "
            f"|action|={show(metric('eval/episode_action_abs_mean_per_step'))} "
            f"|a-home|={show(metric('eval/episode_action_deviation_from_home_abs_mean_per_step'))} "
            f"max|a|={show(metric('eval/episode_action_max_abs_per_step'))} "
            f"sat={show(metric('eval/episode_action_saturation_fraction_per_step'), '.1%')} "
            f"near={show(metric('eval/episode_action_near_boundary_fraction_per_step'), '.1%')}",
            flush=True,
        )
        print(
            "  └─ target: "
            f"excursion={show(metric('eval/episode_target_excursion_from_home_rad_per_step'))}rad "
            f">0.25rad={show(metric('eval/episode_target_excursion_over_tb_range_fraction_per_step'), '.1%')} "
            f"clip={show(metric('eval/episode_target_clipping_fraction_per_step'), '.1%')} "
            f"slew={show(metric('eval/episode_target_slew_limiting_fraction_per_step'), '.1%')}",
            flush=True,
        )
        total_loss = metric("training/total_loss")
        if total_loss is not None:
            print(
                "  └─ ppo   : "
                f"loss={show(total_loss, '.4f')} "
                f"policy={show(metric('training/policy_loss'), '.4f')} "
                f"value={show(metric('training/v_loss'), '.4f')} "
                f"entropy={show(metric('training/entropy_loss'), '.4f')} "
                f"kl={show(metric('training/kl_mean'), '.5f')}",
                flush=True,
            )
        command_forward = metric("eval/episode_command_forward_m_s_per_step")
        forward_velocity = metric("eval/episode_forward_velocity_m_s_per_step")
        fall_rate = metric("eval/episode_fall")
        action_saturation = metric("eval/episode_action_saturation_fraction_per_step")
        nonfinite_state = metric("eval/episode_nonfinite_state")
        mean_step_peak_torque = metric("eval/episode_actuator_torque_peak_nm_per_step")
        acquisition_score = None
        if all(
            value is not None
            for value in (
                episode_length,
                fall_rate,
                command_forward,
                forward_velocity,
                contact_match,
                double_support,
                action_saturation,
                nonfinite_state,
                mean_step_peak_torque,
            )
        ):
            acquisition_score = _acquisition_checkpoint_score(
                episode_length=episode_length,
                target_episode_length=training_config.environment.episode_length,
                fall_rate=fall_rate,
                command_forward_m_s=command_forward,
                forward_velocity_m_s=forward_velocity,
                contact_match=contact_match,
                double_support=double_support,
                action_saturation=action_saturation,
                nonfinite_state=nonfinite_state,
                mean_step_peak_torque_nm=mean_step_peak_torque,
            )
        selection_score = (
            acquisition_score
            if training_config.checkpoints.selection_metric == "acquisition"
            else walking_score
        )
        if training_config.checkpoints.keep_best and selection_score is not None:
            candidate = {
                "step": step,
                "eval_episode_reward": episode_return,
                "walking_score": walking_score,
                "acquisition_score": acquisition_score,
                "selection_metric": training_config.checkpoints.selection_metric,
                "selection_score": selection_score,
                "evaluation": progress_index,
            }
            if latest_params.get("step") == step:
                save_best_checkpoint(latest_params["value"], candidate)
            elif selection_score > best_selection_score:
                # Brax reports the initial evaluation before invoking
                # policy_params_fn. Defer saving until its step-0 parameters
                # arrive instead of losing the stable initial policy.
                pending_checkpoints[step] = candidate
        progress_index += 1

    randomization_fn = (
        None
        if not training_config.environment.randomization.enabled
        else make_domain_randomizer(environment.config.randomization)
    )
    ppo_config = training_config.ppo
    network_factory = make_network_factory(
        training_config.network,
        home_action=environment.home_action,
    )
    checkpoint_path = (
        str(output / "checkpoints")
        if training_config.checkpoints.save_every_evaluation
        else None
    )
    print(
        "Starting PPO; the first JIT compile and evaluation may take several minutes."
    )
    make_inference_fn, params, _ = ppo.train(
        environment=environment,
        num_timesteps=ppo_config.num_timesteps,
        num_envs=ppo_config.num_envs,
        episode_length=environment.config.episode_length,
        num_evals=ppo_config.num_evals,
        num_eval_envs=ppo_config.num_eval_envs,
        eval_env=evaluation_environment,
        learning_rate=ppo_config.learning_rate,
        entropy_cost=ppo_config.entropy_cost,
        discounting=ppo_config.discounting,
        gae_lambda=ppo_config.gae_lambda,
        clipping_epsilon=ppo_config.clipping_epsilon,
        max_grad_norm=ppo_config.max_grad_norm,
        unroll_length=ppo_config.unroll_length,
        batch_size=ppo_config.batch_size,
        num_minibatches=ppo_config.num_minibatches,
        num_updates_per_batch=ppo_config.num_updates_per_batch,
        normalize_observations=ppo_config.normalize_observations,
        network_factory=network_factory,
        randomization_fn=randomization_fn,
        seed=training_config.seed,
        progress_fn=progress,
        policy_params_fn=receive_policy_params,
        save_checkpoint_path=checkpoint_path,
        restore_checkpoint_path=(
            str(args.restore_checkpoint.resolve())
            if args.restore_checkpoint is not None
            else None
        ),
        deterministic_eval=True,
        log_training_metrics=True,
        training_metrics_steps=ppo_config.training_metrics_steps,
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
        "--allow-cpu",
        action="store_true",
        help="Allow full PPO training to run without a JAX GPU backend",
    )
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
    parser.add_argument(
        "--restore-checkpoint",
        type=Path,
        help="Resume from a Brax checkpoint directory in a new output run",
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
