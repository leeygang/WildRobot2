"""Smoke-test or train WR2's phase-guided MJX PPO walking environment."""

from __future__ import annotations

import argparse
import contextlib
import io
import hashlib
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

# Match ToddlerBot: JAX and Torch share the GPU, so do not reserve JAX's
# default preallocated pool before Torch allocates its PPO rollout buffers.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import yaml

from wr2.locomotion.configs import (
    DEFAULT_TRAINING_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
    training_config_to_dict,
)
from wr2.locomotion.walking_metrics import (
    evaluate_xy_goal,
    minimum_foot_swing_fraction,
    walking_score as calculate_walking_score,
)

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


def _checkpoint_evaluation_commands(
    command_range_m_s: tuple[float, float], primary_command_m_s: float
) -> tuple[float, ...]:
    """Return sorted range endpoints plus the configured primary command."""
    values = (*command_range_m_s, primary_command_m_s)
    commands: list[float] = []
    for value in sorted(values):
        if not commands or not math.isclose(value, commands[-1], abs_tol=1e-9):
            commands.append(float(value))
    return tuple(commands)


def _worst_endpoint_score(scores: list[float | None]) -> float | None:
    """Return the conservative score only when every endpoint was evaluated."""
    if not scores or any(score is None or not math.isfinite(score) for score in scores):
        return None
    return min(float(score) for score in scores if score is not None)


def _checkpoint_evaluation_xy_commands(config: TrainingConfig) -> tuple[tuple[float, float], ...]:
    """Preserve forward endpoints; cover cardinal and diagonal XY commands."""
    env = config.environment
    if env.recipe == "tb_paper":
        from wr2.locomotion.paper_evaluation import CONDITIONS

        return tuple(dict.fromkeys(
            (x * env.normalization_scale, y * env.normalization_scale)
            for _, (x, y), transitions in CONDITIONS if not transitions
        ))
    if env.command_lateral_range_m_s == (0.0, 0.0):
        return tuple((x, 0.0) for x in _checkpoint_evaluation_commands(
            env.command_forward_range_m_s, config.ppo.evaluation_forward_command_m_s
        ))
    xs, ys = env.command_forward_range_m_s, env.command_lateral_range_m_s
    commands = [(config.ppo.evaluation_forward_command_m_s, 0.0)]
    commands.extend((x, 0.0) for x in xs)
    commands.extend((x, 0.0) for x in (
        -env.command_active_threshold_m_s, env.command_active_threshold_m_s
    ))
    commands.extend((0.0, y) for y in ys)
    # TB samples an ellipse, not its enclosing box. Keep ranked diagonals
    # within that training distribution rather than evaluating box corners.
    commands.extend((x / math.sqrt(2), y / math.sqrt(2)) for x in xs for y in ys)
    if config.checkpoints.qualification_goal is not None:
        commands.append((0.0, 0.0))
    return tuple(dict.fromkeys(commands))


def _checkpoint_transition_xy_commands(config: TrainingConfig) -> tuple[tuple[float, float], ...]:
    if config.environment.recipe == "tb_paper":
        x, y = config.ppo.evaluation_forward_command_m_s, config.ppo.evaluation_lateral_command_m_s
        return ((x, 0.0), (-x, 0.0), (0.0, -y), (0.0, y))
    commands = ((config.ppo.evaluation_forward_command_m_s, 0.0),)
    if config.environment.command_lateral_range_m_s != (0.0, 0.0):
        commands += (
            (config.environment.command_forward_range_m_s[0], 0.0),
            (0.0, config.environment.command_lateral_range_m_s[0]),
            (0.0, config.environment.command_lateral_range_m_s[1]),
        )
    return tuple(dict.fromkeys(commands))


def _worst_endpoint_selection_score(
    endpoint_evaluations: list[dict[str, float | None]],
    selection_metric: str,
) -> float | None:
    """Select the configured score from every fixed-command evaluation."""
    score_key = {
        "acquisition": "acquisition_score",
        "walking_score": "walking_score",
    }[selection_metric]
    return _worst_endpoint_score(
        [endpoint[score_key] for endpoint in endpoint_evaluations]
    )


def _validate_training_zmp_reference(
    training_config: TrainingConfig,
):
    """Run the reference-only geometry gate for every PPO lookup command."""
    from wr2.reference.zmp_validation import (
        create_reference_context,
        validate_zmp_reference,
    )
    from wr2.reference.walk_zmp import reference_command_grid

    environment = training_config.environment
    if environment.recipe == "tb_paper":
        from wr2.reference.zmp_validation import validate_paper_training_reference

        print("Building and checking the TB-style WR2 XY/turn reference table...", flush=True)
        results = validate_paper_training_reference(training_config)
        failures = [f"({r.command_forward_m_s:.3f},{r.command_lateral_m_s:.3f},{r.command_yaw_rad_s:.3f}): {f}"
                    for r in results for f in r.failures]
        if failures:
            raise RuntimeError("Paper reference preflight failed: " + "; ".join(failures[:5]))
        return results
    zmp = environment.zmp_reference
    low, high = environment.command_forward_range_m_s
    varying_range = not math.isclose(low, high) or (
        environment.command_lateral_range_m_s[0] != environment.command_lateral_range_m_s[1]
    )
    if varying_range and zmp.command_samples < 2:
        raise ValueError("ZMP command_samples must be at least two for a varying command range")
    commands = reference_command_grid(
        (low, high), environment.command_lateral_range_m_s, zmp.command_samples
    )
    for command in _checkpoint_evaluation_xy_commands(training_config):
        if not any(
            all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(command, item))
            for item in commands
        ):
            commands += (command,)
    context = create_reference_context(
        training_config=training_config, commands_xy_m_s=commands
    )
    results = [
        validate_zmp_reference(context, *command, samples=2 * zmp.phase_samples)
        for command in commands
    ]

    failures = [
        f"vxy=({result.command_forward_m_s:.3f},{result.command_lateral_m_s:.3f}): {failure}"
        for result in results
        for failure in result.failures
    ]
    if failures:
        preview = "; ".join(failures[:5])
        remainder = "" if len(failures) <= 5 else f"; and {len(failures) - 5} more"
        raise RuntimeError(f"ZMP PPO reference preflight failed: {preview}{remainder}")
    return tuple(results)


def _format_zmp_preflight(results) -> str:
    if hasattr(results[0], "command_yaw_rad_s"):
        commands = f"{len(results)} XY/turn lookup points"
        optimal_margin_mm = min(result.minimum_optimal_cop_support_margin_m for result in results) * 1000.0
        optimal_info = f" optimal_COP_margin>={optimal_margin_mm:.2f}mm (soft preview cost)"
    else:
        optimal_info = ""
        commands = f"{len(results)} XY lookup points" if any(
            item.command_lateral_m_s != 0.0 for item in results
        ) else ",".join(f"{item.command_forward_m_s:.3f}" for item in results)
    support_margin_mm = (
        min(result.minimum_zmp_support_margin_m for result in results) * 1000.0
    )
    stance_height_mm = (
        max(result.worst_stance_bottom_z_m for result in results) * 1000.0
    )
    return (
        f"commands=[{commands}] support_margin>={support_margin_mm:.2f}mm "
        f"stance_z<={stance_height_mm:.2f}mm{optimal_info}"
    )


def smoke_test(steps: int, training_config: TrainingConfig | None = None) -> None:
    _configure_backend_logging()
    with _filter_optional_backend_import_messages():
        import jax

        from wr2.locomotion.walking_env import WR2WalkingEnv

    training_config = training_config or load_training_config()
    zmp_preflight = _validate_training_zmp_reference(training_config)
    print(f"ZMP PPO preflight passed: {_format_zmp_preflight(zmp_preflight)}")
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


def _training_provenance() -> dict:
    from importlib import metadata
    import subprocess

    versions = {}
    for package in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "rsl-rl-lib", "torch", "numpy", "scipy", "flax", "optax", "orbax-checkpoint"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, capture_output=True, text=True,
    )
    return {
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "python": sys.version.split()[0],
        "packages": versions,
    }


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

    zmp_preflight = _validate_training_zmp_reference(training_config)
    environment = WR2WalkingEnv(
        training_config.environment,
        add_observation_noise=True,
    )
    evaluation_commands = _checkpoint_evaluation_xy_commands(training_config)
    paper_recipe = training_config.environment.recipe == "tb_paper"
    if paper_recipe:
        # Brax's native evaluation is progress telemetry; the independent
        # paper evaluator below measures all 15 conditions per checkpoint.
        evaluation_commands = ((training_config.ppo.evaluation_forward_command_m_s, 0.0),)
    evaluation_environments = tuple(
        (
            command,
            WR2WalkingEnv(
                replace(
                    training_config.environment,
                    command_forward_range_m_s=(command[0], command[0]),
                    command_lateral_range_m_s=(command[1], command[1]),
                    command_yaw_range_rad_s=(0.0, 0.0),
                    turn_command_probability=0.0,
                    zero_command_probability=0.0,
                ),
                add_observation_noise=(
                    training_config.ppo.evaluation_randomization
                    and training_config.environment.randomization.enabled
                ),
            ),
        )
        for command in evaluation_commands
    )
    evaluation_environment = next(
        environment
        for command, environment in evaluation_environments
        if math.isclose(
            command[0],
            training_config.ppo.evaluation_forward_command_m_s,
            abs_tol=1e-9,
        ) and command[1] == 0.0
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
    print(f"  PPO backend:  {training_config.ppo.backend}")
    print(f"  JAX backend:  {backend}")
    print(f"  JAX memory:   preallocate={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
    print(f"  Devices:      {devices}")
    print(
        "  Observations: "
        f"actor={environment.observation_size['state']} "
        f"critic={environment.observation_size['privileged_state']}"
    )
    print(f"  Actions:      {environment.action_size} (leg joints active)")
    print(
        "  Action map:   fixed q_home + "
        f"{training_config.environment.policy_action_scale_rad:.3f}rad * residual; "
        f"initial residual std={training_config.network.init_noise_std:.3f}"
    )
    print(f"  Environments: {training_config.ppo.num_envs:,}")
    print(f"  Target steps: {training_config.ppo.num_timesteps:,}")
    if training_config.ppo.backend == "rsl_rl":
        from wr2.locomotion.rsl_rl_training import learning_iterations

        print(
            "  Iterations:   "
            f"{learning_iterations(training_config.ppo.num_timesteps, training_config.ppo.num_envs, training_config.ppo.unroll_length):,} "
            f"x {training_config.ppo.num_envs * training_config.ppo.unroll_length:,} steps"
        )
    print(f"  Episode:      {training_config.environment.episode_length} steps")
    print(
        f"  Physics:      Newton iterations={int(environment.sys.opt.iterations)} "
        f"line_search={int(environment.sys.opt.ls_iterations)}; "
        f"{environment.robot.simulation_timestep_s * 1000:.0f}ms x "
        f"{environment._n_frames} substeps"
    )
    print(
        "  Evaluation:   "
        f"{training_config.ppo.num_evals} evaluations x "
        f"{training_config.ppo.num_eval_envs} parallel episodes; "
        "fixed endpoints "
        f"{evaluation_commands}m/s XY; "
        f"primary={training_config.ppo.evaluation_forward_command_m_s:.2f}m/s"
    )
    print(
        "  Gait:         "
        f"{training_config.environment.gait_cycle_s:.2f}s cycle, "
        f"{training_config.environment.swing_height_m:.3f}m swing"
    )
    zmp = environment.zmp_reference_diagnostics
    print(
        "  ZMP lookup:   "
        f"{len(zmp.command_samples_xy_m_s)} command x "
        f"{zmp.lookup_length if paper_recipe else training_config.environment.zmp_reference.phase_samples} frames; "
        f"IK residual={zmp.max_ik_position_residual_m * 1000.0:.2f}mm"
    )
    print(f"  ZMP preflight: {_format_zmp_preflight(zmp_preflight)}")
    print(f"  Reference Z:   root={zmp.root_height_m:.3f}m CoM={zmp.com_height_m:.3f}m")
    print(
        "  Network:      "
        f"policy={list(training_config.network.policy_hidden_layer_sizes)} "
        f"value={list(training_config.network.value_hidden_layer_sizes)} "
        f"activation={training_config.network.activation} "
        f"distribution={training_config.network.distribution_type}"
    )
    print(
        "  Optimizer:    "
        f"lr={training_config.ppo.learning_rate:.2e} "
        f"schedule={training_config.ppo.learning_rate_schedule} "
        f"desired_kl={training_config.ppo.desired_kl:.3f}"
    )
    print(f"  Randomized:   {training_config.environment.randomization.enabled}")
    if training_config.ppo.mirror_loss_coeff > 0.0:
        print(
            "  Mirror loss:  actor-only "
            f"coeff={training_config.ppo.mirror_loss_coeff:g}; no data augmentation"
        )
    if args.restore_checkpoint is not None:
        print(f"  Restore:      {args.restore_checkpoint.resolve()}")
    print("=" * 72)
    effective_config = training_config_to_dict(training_config)
    (output / "training_config.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8"
    )
    run_config = {
        "provenance": _training_provenance(),
        "run_id": run_id,
        "output": str(output),
        "source_config": str(args.config.resolve()),
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "training_config": effective_config,
        "actuator_model": environment.robot.config["actuators"]["htd45hServo"],
        "policy_home_action": np.asarray(environment.home_action).tolist(),
        "policy_home_motor_position_rad": np.asarray(environment._home_ctrl).tolist(),
        "policy_action_scale_rad": (
            training_config.environment.policy_action_scale_rad
        ),
        "restore_checkpoint_sha256": (
            hashlib.sha256(args.restore_checkpoint.read_bytes()).hexdigest()
            if args.restore_checkpoint is not None and args.restore_checkpoint.is_file()
            else None
        ),
        "physics": {
            "solver_iterations": int(environment.sys.opt.iterations),
            "line_search_iterations": int(environment.sys.opt.ls_iterations),
            "timestep_s": float(environment.sys.opt.timestep),
            "control_substeps": environment._n_frames,
            "model_sha256": hashlib.sha256(
                (environment.robot.directory / "wr2_mjx.xml").read_bytes()
            ).hexdigest(),
        },
        "zmp_reference_validation": [asdict(result) for result in zmp_preflight],
        "zmp_reference_diagnostics": asdict(zmp),
        "restore_semantics": "parameters_only_new_Adam_and_step_counter" if training_config.ppo.backend == "brax" else "RSL_checkpoint",
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = output / "training_metrics.jsonl"
    metrics_path.write_text("")
    rollout_metrics_path = output / "rollout_metrics.jsonl"
    rollout_metrics_path.write_text("")

    training_started = time.monotonic()
    xy_training = training_config.environment.command_lateral_range_m_s != (0.0, 0.0)
    velocity_error_metric = (
        "episode_xy_velocity_error_m_s_per_step" if xy_training
        else "episode_forward_velocity_error_m_s_per_step"
    )
    progress_index = 0
    latest_params: dict[str, object] = {}
    best_selection_score = -math.inf
    pending_checkpoints: dict[int, dict[str, object]] = {}
    paper_evaluator = None
    paper_policy_factory = None
    best_paper_error = math.inf

    def save_best_checkpoint(params, candidate: dict[str, object]) -> None:
        nonlocal best_selection_score
        selection_score = float(candidate["selection_score"])
        if not math.isfinite(selection_score) or selection_score <= best_selection_score:
            return
        best_selection_score = selection_score
        if training_config.ppo.backend == "rsl_rl":
            import torch

            torch.save(params, output / "best_params.pt")
        else:
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
        nonlocal paper_evaluator, paper_policy_factory, best_paper_error
        step = int(step)
        latest_params["step"] = step
        latest_params["value"] = params
        if paper_recipe:
            from wr2.locomotion.paper_evaluation import PaperEvaluator, selection_summary

            if any(not np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(params)):
                raise RuntimeError(f"Nonfinite PPO parameters at step {step}")
            paper_policy_factory = _make_policy
            if paper_evaluator is None:
                paper_evaluator = PaperEvaluator(
                    evaluation_environment, _make_policy, training_config.ppo.num_eval_envs,
                    randomization_fn if training_config.ppo.evaluation_randomization else None,
                )
            print(f"Paper comparison at {step:,} steps: 15 conditions, seed 12102...", flush=True)
            report = paper_evaluator.evaluate(params)
            summary = selection_summary(report)
            record = {"step": step, "summary": summary, **report}
            evaluation_dir = output / "paper_evaluations"
            evaluation_dir.mkdir(exist_ok=True)
            (evaluation_dir / f"{step:012d}.json").write_text(json.dumps(record, allow_nan=False) + "\n")
            with (output / "paper_evaluation_history.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **summary}, allow_nan=False) + "\n")
            print(
                f"  paper: falls={summary['fall_count']}/{summary['episodes']} "
                f"nonfinite={summary['nonfinite_count']} "
                f"fixed walking XY MAE={summary['walking_xy_error_m_s']}m/s "
                f"TB-equivalent={summary['walking_xy_error_TB_equivalent_m_s']}m/s",
                flush=True,
            )
            if summary["nonfinite_count"]:
                model.save_params(output / f"nonfinite_eval_params_{step}", params)
                raise RuntimeError("Paper evaluation found nonfinite transitions; see its per-episode report")
            if step == 0:
                model.save_params(output / "initial_params", params)
            elif training_config.checkpoints.keep_best and summary["eligible"] and summary["walking_xy_error_m_s"] < best_paper_error:
                best_paper_error = summary["walking_xy_error_m_s"]
                model.save_params(output / "best_params", params)
                candidate = {
                    "step": step, "selection_metric": "fixed_walking_xy_error_with_survival_gate",
                    "selection_seed": 12102, "summary": summary,
                    "confirmation_seed": 22102, "held_out_confirmation_required": True,
                    "evaluation_report": str(evaluation_dir / f"{step:012d}.json"),
                }
                (output / "best_checkpoint.json").write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")
                print(f"  saved candidate at {step:,}; held-out confirmation follows training", flush=True)
            return
        candidate = pending_checkpoints.pop(step, None)
        if candidate is not None:
            save_best_checkpoint(params, candidate)

    def progress(step: int, metrics) -> None:
        nonlocal best_selection_score, progress_index
        elapsed_s = time.monotonic() - training_started
        step = int(step)
        if paper_recipe:
            serialized = {name: np.asarray(value).tolist() for name, value in metrics.items()}
            invalid = [name for name, value in metrics.items() if not np.isfinite(np.asarray(value)).all()]
            if invalid:
                raise RuntimeError(f"Nonfinite PPO metrics at step {step}: {invalid[:5]}")
            if any(float(np.asarray(v)) > 0 for name, v in metrics.items() if name.endswith("nonfinite_state")):
                raise RuntimeError(f"Nonfinite simulation state at PPO step {step}")
            path = metrics_path if "eval/avg_episode_length" in metrics else rollout_metrics_path
            with path.open("a") as stream:
                stream.write(json.dumps({"step": step, "elapsed_s": elapsed_s, "metrics": serialized}, allow_nan=False) + "\n")
            print(
                f"PPO [{_format_duration(elapsed_s)}] steps={step:,} "
                f"reward={serialized.get('eval/episode_reward', 'n/a')} "
                f"loss={serialized.get('training/total_loss', 'n/a')}", flush=True,
            )
            return
        target_steps = training_config.ppo.num_timesteps
        progress_fraction = min(max(step / target_steps, 0.0), 1.0)

        def metric(name: str) -> float | None:
            value = metrics.get(name)
            if value is None:
                return None
            array = np.asarray(value)
            if array.size != 1:
                return None
            result = float(array.item())
            return result if math.isfinite(result) else None

        def qualification_for(prefix, command, *, transitions=False):
            goal = training_config.checkpoints.qualification_goal
            if goal is None:
                return None
            scoped = {
                "eval/" + name.removeprefix(prefix): value
                for name, value in metrics.items() if name.startswith(prefix)
            }
            try:
                return evaluate_xy_goal(
                    scoped, goal, command_xy_m_s=command,
                    target_episode_length=training_config.environment.episode_length,
                    velocity_sigma_m_s=training_config.environment.velocity_tracking_sigma,
                    transitions=transitions,
                ).to_dict()
            except ValueError as error:
                return {"passed": False, "error": str(error)}

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

            def rollout_show_loss(value: float | None) -> str:
                return "n/a" if value is None else format(value, ".4f")

            left_contact = metric("episode/left_foot_contact_per_step")
            right_contact = metric("episode/right_foot_contact_per_step")
            minimum_swing = (
                None
                if left_contact is None or right_contact is None
                else minimum_foot_swing_fraction(left_contact, right_contact)
            )
            minimum_swing_text = (
                "n/a" if minimum_swing is None else format(minimum_swing, ".1%")
            )
            xy_tracking_text = (
                f"XY_MAE={rollout_show('xy_velocity_error_m_s_per_step')} "
                if xy_training else ""
            )
            print(
                f"Rollout [{_format_duration(elapsed_s)}] steps={step:,} | "
                f"ep_len={rollout_show('length', '.0f')} "
                f"vx={rollout_show('forward_velocity_m_s_per_step')} "
                f"{xy_tracking_text}"
                f"progress={rollout_show('forward_progress_per_step')} "
                f"contact={rollout_show('contact_phase_match_per_step', '.1%')} "
                f"min_swing={minimum_swing_text} "
                f"sat={rollout_show('action_saturation_fraction_per_step', '.1%')} "
                f"near={rollout_show('action_near_boundary_fraction_per_step', '.1%')} "
                f"|residual|={rollout_show('action_abs_mean_per_step')}",
                flush=True,
            )
            learning_rate = metric("training/learning_rate")
            if learning_rate is not None:
                mirror_loss = metric("training/mirror_loss")
                mirror_text = "" if mirror_loss is None else f"mirror={mirror_loss:.4f} "
                cumulative_steps = metric("training/total_steps")
                cumulative_steps_text = (
                    "n/a"
                    if cumulative_steps is None
                    else format(cumulative_steps, ",.0f")
                )
                print(
                    "  └─ ppo   : "
                    f"loss={rollout_show_loss(metric('training/total_loss'))} "
                    f"policy={rollout_show_loss(metric('training/policy_loss'))} "
                    f"value={rollout_show_loss(metric('training/v_loss'))} "
                    f"{mirror_text}"
                    f"kl={rollout_show_loss(metric('training/kl_mean'))} "
                    f"std={rollout_show_loss(metric('training/mean_noise_std'))} "
                    f"lr={learning_rate:.2e} "
                    f"total_steps={cumulative_steps_text}",
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
        velocity_error = metric(f"eval/{velocity_error_metric}")
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
        endpoint_evaluations = []
        for endpoint_index, command in enumerate(evaluation_commands):
            prefix = f"eval_endpoint_{endpoint_index}/"

            def endpoint_metric(name: str) -> float | None:
                return metric(f"{prefix}{name}")

            endpoint_episode_length = endpoint_metric("avg_episode_length")
            endpoint_velocity_error = endpoint_metric(velocity_error_metric)
            endpoint_contact_match = endpoint_metric(
                "episode_contact_phase_match_per_step"
            )
            endpoint_double_support = endpoint_metric("episode_double_support_per_step")
            endpoint_walking_score = None
            if command != (0.0, 0.0) and all(
                value is not None
                for value in (
                    endpoint_episode_length,
                    endpoint_velocity_error,
                    endpoint_contact_match,
                    endpoint_double_support,
                )
            ):
                endpoint_walking_score = calculate_walking_score(
                    episode_length=endpoint_episode_length,
                    target_episode_length=(training_config.environment.episode_length),
                    velocity_error_m_s=endpoint_velocity_error,
                    velocity_sigma_m_s=(
                        training_config.environment.velocity_tracking_sigma
                    ),
                    contact_match=endpoint_contact_match,
                    double_support=endpoint_double_support,
                )
            endpoint_forward_velocity = endpoint_metric(
                "episode_forward_velocity_m_s_per_step"
            )
            endpoint_lateral_velocity = endpoint_metric("episode_lateral_velocity_m_s_per_step")
            command_speed = math.hypot(*command)
            projected_velocity = (
                (command[0] * endpoint_forward_velocity + command[1] * endpoint_lateral_velocity)
                / command_speed
                if command_speed > 0
                and endpoint_forward_velocity is not None
                and endpoint_lateral_velocity is not None
                else None
            )
            endpoint_fall_rate = endpoint_metric("episode_fall")
            endpoint_action_saturation = endpoint_metric(
                "episode_action_saturation_fraction_per_step"
            )
            endpoint_nonfinite = endpoint_metric("episode_nonfinite_state")
            endpoint_peak_torque = endpoint_metric(
                "episode_actuator_torque_peak_nm_per_step"
            )
            endpoint_acquisition_score = None
            if all(
                value is not None
                for value in (
                    endpoint_episode_length,
                    endpoint_fall_rate,
                    projected_velocity,
                    endpoint_contact_match,
                    endpoint_double_support,
                    endpoint_action_saturation,
                    endpoint_nonfinite,
                    endpoint_peak_torque,
                )
            ):
                endpoint_acquisition_score = _acquisition_checkpoint_score(
                    episode_length=endpoint_episode_length,
                    target_episode_length=(training_config.environment.episode_length),
                    fall_rate=endpoint_fall_rate,
                    command_forward_m_s=command_speed,
                    forward_velocity_m_s=projected_velocity,
                    contact_match=endpoint_contact_match,
                    double_support=endpoint_double_support,
                    action_saturation=endpoint_action_saturation,
                    nonfinite_state=endpoint_nonfinite,
                    mean_step_peak_torque_nm=endpoint_peak_torque,
                )
            endpoint_evaluations.append(
                {
                    "command_forward_m_s": command[0],
                    "command_lateral_m_s": command[1],
                    "episode_length": endpoint_episode_length,
                    "fall_rate": endpoint_fall_rate,
                    "forward_velocity_m_s": endpoint_forward_velocity,
                    "lateral_velocity_m_s": endpoint_lateral_velocity,
                    "velocity_error_m_s": endpoint_velocity_error,
                    "contact_match": endpoint_contact_match,
                    "double_support": endpoint_double_support,
                    "walking_score": endpoint_walking_score,
                    "acquisition_score": endpoint_acquisition_score,
                    "qualification": qualification_for(prefix, command),
                }
            )
        qualification = None
        if training_config.checkpoints.qualification_goal is not None:
            transition_commands = _checkpoint_transition_xy_commands(training_config)
            transition_results = [
                {
                    "command_xy_m_s": list(command),
                    "qualification": qualification_for(
                        f"eval_transition_endpoint_{index}/" if len(transition_commands) > 1 else "eval_transition/",
                        command, transitions=True,
                    ),
                }
                for index, command in enumerate(transition_commands)
            ]
            qualification = {
                "passed": all(endpoint["qualification"]["passed"] for endpoint in endpoint_evaluations)
                and all(result["qualification"]["passed"] for result in transition_results),
                "transition_evaluations": transition_results,
            }
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
                        "endpoint_evaluations": endpoint_evaluations,
                        "qualification": qualification,
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
        xy_tracking_text = f"XY_MAE={show(velocity_error)} " if xy_training else ""
        print(
            "  └─ track : "
            f"cmd_vx={show(metric('eval/episode_command_forward_m_s_per_step'))} "
            f"vx={show(metric('eval/episode_forward_velocity_m_s_per_step'))} "
            f"|vx-cmd|={show(metric('eval/episode_forward_velocity_error_m_s_per_step'))} "
            f"{xy_tracking_text}"
            f"progress={show(metric('eval/episode_forward_progress_per_step'))} "
            f"vy={show(metric('eval/episode_lateral_velocity_m_s_per_step'))} "
            f"yaw_err={show(metric('eval/episode_yaw_rate_error_rad_s_per_step'))}",
            flush=True,
        )
        heading_error = metric("eval/episode_heading_error_rad_per_step")
        print(
            "  └─ path  : "
            f"heading_err={show(None if heading_error is None else math.degrees(heading_error), '.1f')}deg "
            f"forward={show(metric('eval/episode_path_forward_displacement_m'))}m "
            f"lateral={show(metric('eval/episode_path_lateral_displacement_m'))}m",
            flush=True,
        )
        phase_errors = []
        for index in range(4):
            fraction = metric(f"eval/episode_phase{index}_sample_count")
            error_sum = metric(f"eval/episode_phase{index}_velocity_error_sum")
            phase_errors.append(
                show(
                    error_sum / fraction if fraction and error_sum is not None else None
                )
            )
        print(f"  └─ phase : speed_MAE_by_quarter={phase_errors}m/s", flush=True)
        if "eval_transition/avg_episode_length" in metrics:
            text = []
            error_name = "xy_velocity_error" if xy_training else "velocity_error"
            for category in ("standing", "walking", "start", "stop"):
                fraction = metric(
                    f"eval_transition/episode_{category}_sample_count"
                )
                error_sum = metric(
                    f"eval_transition/episode_{category}_{error_name}_sum"
                )
                text.append(
                    f"{category}={show(error_sum / fraction if fraction and error_sum is not None else None)}"
                )
            print(
                "  └─ switch: "
                f"ep_len={show(metric('eval_transition/avg_episode_length'), '.0f')} "
                f"fall={show(metric('eval_transition/episode_fall'), '.1%')} "
                f"{'XY_MAE' if xy_training else 'speed_MAE'} {' '.join(text)}m/s",
                flush=True,
            )
        if endpoint_evaluations:
            endpoint_text = " | ".join(
                f"({endpoint['command_forward_m_s']:.2f},{endpoint['command_lateral_m_s']:.2f}):"
                f"vx={show(endpoint['forward_velocity_m_s'])},"
                f"vy={show(endpoint['lateral_velocity_m_s'])},"
                f"MAE={show(endpoint['velocity_error_m_s'])},"
                f"fall={show(endpoint['fall_rate'], '.1%')},"
                f"score={show(endpoint['walking_score'])}"
                for endpoint in endpoint_evaluations
            )
            print(f"  └─ endpoints: {endpoint_text}", flush=True)
        tilt_rad = metric("eval/episode_torso_tilt_rad_per_step")
        left_contact = metric("eval/episode_left_foot_contact_per_step")
        right_contact = metric("eval/episode_right_foot_contact_per_step")
        minimum_swing = (
            None
            if left_contact is None or right_contact is None
            else minimum_foot_swing_fraction(left_contact, right_contact)
        )
        print(
            "  └─ state : "
            f"height={show(metric('eval/episode_torso_height_m_per_step'))}m "
            f"tilt={show(None if tilt_rad is None else math.degrees(tilt_rad), '.1f')}deg "
            f"fall={show(metric('eval/episode_fall'), '.1%')} "
            f"feet=L{show(left_contact, '.0%')}"
            f"/R{show(right_contact, '.0%')}",
            flush=True,
        )
        print(
            "  └─ gait  : "
            f"phase_score={show(metric('eval/episode_feet_phase_tracking_per_step'))} "
            f"contact_match={show(metric('eval/episode_contact_phase_match_per_step'), '.1%')} "
            f"double_support={show(metric('eval/episode_double_support_per_step'), '.1%')} "
            f"min_swing={show(minimum_swing, '.1%')} "
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
            f"|residual|={show(metric('eval/episode_action_abs_mean_per_step'))} "
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
            mirror_loss = metric("training/mirror_loss")
            mirror_text = "" if mirror_loss is None else f"mirror={mirror_loss:.4f} "
            print(
                "  └─ ppo   : "
                f"loss={show(total_loss, '.4f')} "
                f"policy={show(metric('training/policy_loss'), '.4f')} "
                f"value={show(metric('training/v_loss'), '.4f')} "
                f"entropy={show(metric('training/entropy_loss'), '.4f')} "
                f"{mirror_text}"
                f"kl={show(metric('training/kl_mean'), '.5f')} "
                f"std={show(metric('training/mean_noise_std'), '.3f')} "
                f"total_steps={show(metric('training/total_steps'), ',.0f')}",
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
        selection_metric = training_config.checkpoints.selection_metric
        endpoint_selection_score = _worst_endpoint_selection_score(
            [endpoint for endpoint in endpoint_evaluations if (
                endpoint["command_forward_m_s"], endpoint["command_lateral_m_s"]
            ) != (0.0, 0.0)],
            selection_metric,
        )
        primary_selection_score = (
            acquisition_score if selection_metric == "acquisition" else walking_score
        )
        # Native training requires every configured endpoint. A missing/bad
        # endpoint must not promote a candidate using its primary command.
        selection_uses_endpoints = training_config.ppo.backend == "rsl_rl" or any(
            name.startswith("eval_endpoint_") for name in metrics
        )
        selection_score = (
            endpoint_selection_score if selection_uses_endpoints else primary_selection_score
        )
        if qualification is not None:
            print(f"  └─ XY qualification: {'passed' if qualification['passed'] else 'not passed'}", flush=True)
        if (
            training_config.checkpoints.keep_best and selection_score is not None
            and (qualification is None or qualification["passed"])
        ):
            candidate = {
                "step": step,
                "eval_episode_reward": episode_return,
                "walking_score": walking_score,
                "acquisition_score": acquisition_score,
                "endpoint_evaluations": endpoint_evaluations,
                "selection_metric": selection_metric,
                "selection_scope": (
                    "worst_endpoint"
                    if selection_uses_endpoints
                    else "primary_command"
                ),
                "selection_score": selection_score,
                "qualification": qualification,
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
    if ppo_config.backend == "rsl_rl":
        from wr2.locomotion.rsl_rl_training import train_rsl_ppo

        train_rsl_ppo(
            environment,
            evaluation_environments,
            training_config,
            output=output,
            restore_checkpoint=args.restore_checkpoint,
            progress_fn=progress,
            policy_params_fn=receive_policy_params,
            randomization_fn=randomization_fn,
            allow_cpu=args.allow_cpu,
        )
        return

    network_factory = make_network_factory(training_config.network)
    from brax.training.agents.ppo.optimizer import LRSchedule

    learning_rate_schedule = {
        "fixed": LRSchedule.NONE,
        "adaptive": LRSchedule.ADAPTIVE_KL,
    }[ppo_config.learning_rate_schedule]
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
        normalize_advantage=ppo_config.normalize_advantage,
        # WR2's coefficient multiplies MSE; Brax applies an additional 0.5.
        vf_loss_coefficient=2.0 * ppo_config.value_loss_coef,
        desired_kl=ppo_config.desired_kl,
        learning_rate_schedule=learning_rate_schedule,
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
    if paper_recipe and (output / "best_params").exists():
        from wr2.locomotion.paper_evaluation import PaperEvaluator, selection_summary

        print("Confirming the selected candidate on held-out seed 22102...", flush=True)
        selected = model.load_params(output / "best_params")
        selected_step = json.loads((output / "best_checkpoint.json").read_text())["step"]
        confirmation = PaperEvaluator(
            evaluation_environment, paper_policy_factory, ppo_config.num_eval_envs,
            randomization_fn if ppo_config.evaluation_randomization else None, seed=22102,
        ).evaluate(selected)
        (output / "paper_confirmation_22102.json").write_text(
            json.dumps({"step": selected_step, "checkpoint": "best_params",
                        "checkpoint_sha256": hashlib.sha256((output / "best_params").read_bytes()).hexdigest(),
                        "summary": selection_summary(confirmation), **confirmation}, allow_nan=False) + "\n"
        )


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
    if args.backend is not None:
        ppo_overrides["backend"] = args.backend
    ppo_config = replace(training_config.ppo, **ppo_overrides)
    if training_config.environment.recipe == "tb_paper" and ppo_config.backend != "brax":
        raise ValueError("the paper recipe requires the native Brax backend")
    if ppo_config.mirror_loss_coeff > 0.0 and ppo_config.backend != "rsl_rl":
        raise ValueError("ppo.mirror_loss_coeff requires the rsl_rl backend")

    environment = training_config.environment
    if args.episode_length is not None:
        environment = replace(environment, episode_length=args.episode_length)
    if args.solver_iterations is not None:
        if args.solver_iterations < 1:
            raise ValueError("--solver-iterations must be positive")
        environment = replace(environment, solver_iterations=args.solver_iterations)
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
    parser.add_argument(
        "--backend",
        choices=("rsl_rl", "brax"),
        default=None,
        help="Override ppo.backend from the YAML",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-evals", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument(
        "--solver-iterations", type=int, default=None,
        help="Override the Newton solver iteration ceiling for a matched physics A/B test",
    )
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
        help="Resume from an RSL-RL .pt file or Brax checkpoint directory",
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
