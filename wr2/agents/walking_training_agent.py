"""Drive bounded WR2 walking train/evaluate/promote cycles locally or over SSH."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.walking_metrics import (
    WALKING_GATE_NAMES,
    WalkingGoal,
    WalkingGoalResult,
    evaluate_walking_goal,
    walking_goal_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_CONFIG = REPO_ROOT / "wr2/locomotion/configs/walking_agent.yaml"
_SAFE_OVERRIDES = {
    "domain_randomization.enabled",
    "environment.command_forward_range_m_s",
    "environment.zero_command_probability",
}


class WalkingAgentError(RuntimeError):
    """Actionable walking-supervisor failure."""


@dataclass(frozen=True)
class CycleConfig:
    num_timesteps: int
    num_evals: int
    confirmation_commands_m_s: tuple[float, ...]
    confirmation_seeds: tuple[int, ...]
    confirmation_num_envs: int


@dataclass(frozen=True)
class HardSafetyConfig:
    max_mean_step_peak_torque_nm: float
    max_action_saturation_fraction: float
    max_nonfinite_state: float


@dataclass(frozen=True)
class StageConfig:
    name: str
    max_cycles: int
    evaluation_commands_m_s: tuple[float, ...]
    required_gates: tuple[str, ...]
    overrides: dict[str, Any]
    goal: WalkingGoal


@dataclass(frozen=True)
class WalkingAgentConfig:
    version: str
    base_training_config: Path
    agent_output_root: Path
    cycle: CycleConfig
    hard_safety: HardSafetyConfig
    stages: tuple[StageConfig, ...]


@dataclass(frozen=True)
class Candidate:
    evaluation: int
    step: int
    checkpoint: Path
    goal_result: WalkingGoalResult
    required_gates: tuple[str, ...]
    metrics: dict[str, Any]

    @property
    def required_passed(self) -> bool:
        return self.goal_result.passes(self.required_gates)

    @property
    def rank(
        self,
    ) -> tuple[bool, int, float, float, float, float, float, float, float, float]:
        values = self.goal_result.values
        return (
            self.required_passed,
            sum(self.goal_result.gates[name] for name in self.required_gates),
            values["episode_length"],
            -values["fall_rate"],
            -values["forward_velocity_error_m_s"],
            values["forward_velocity_ratio"],
            values["contact_phase_match"],
            values["minimum_foot_swing_fraction"],
            -values["action_saturation_fraction"],
            self.goal_result.score,
        )


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    if result <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{label} must be a positive integer")
    return result


def _repo_path(value: str, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository") from exc
    return resolved


def load_agent_config(path: str | Path = DEFAULT_AGENT_CONFIG) -> WalkingAgentConfig:
    """Load and strictly validate the walking-agent campaign."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("walking agent config must be a mapping")
    _require_keys(
        raw,
        {
            "version",
            "base_training_config",
            "agent_output_root",
            "cycle",
            "hard_safety",
            "stages",
        },
        "walking agent config",
    )
    base_config = _repo_path(str(raw["base_training_config"]), "base_training_config")
    training = load_training_config(base_config)

    cycle_raw = raw["cycle"]
    if not isinstance(cycle_raw, dict):
        raise ValueError("cycle must be a mapping")
    _require_keys(
        cycle_raw,
        {
            "num_timesteps",
            "num_evals",
            "confirmation_commands_m_s",
            "confirmation_seeds",
            "confirmation_num_envs",
        },
        "cycle",
    )
    seeds_raw = cycle_raw["confirmation_seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("cycle.confirmation_seeds must be a non-empty list")
    seeds = tuple(int(seed) for seed in seeds_raw)
    if len(set(seeds)) != len(seeds):
        raise ValueError("cycle.confirmation_seeds must be unique")
    commands_raw = cycle_raw["confirmation_commands_m_s"]
    if not isinstance(commands_raw, list) or not commands_raw:
        raise ValueError("cycle.confirmation_commands_m_s must be a non-empty list")
    commands = tuple(float(command) for command in commands_raw)
    if any(command <= 0.0 for command in commands) or len(set(commands)) != len(
        commands
    ):
        raise ValueError(
            "cycle.confirmation_commands_m_s must contain unique positive values"
        )
    cycle = CycleConfig(
        num_timesteps=_positive_int(cycle_raw["num_timesteps"], "cycle.num_timesteps"),
        num_evals=_positive_int(cycle_raw["num_evals"], "cycle.num_evals"),
        confirmation_commands_m_s=commands,
        confirmation_seeds=seeds,
        confirmation_num_envs=_positive_int(
            cycle_raw["confirmation_num_envs"], "cycle.confirmation_num_envs"
        ),
    )
    if cycle.num_evals < 2:
        raise ValueError("cycle.num_evals must include an initial and final evaluation")

    hard_raw = raw["hard_safety"]
    if not isinstance(hard_raw, dict):
        raise ValueError("hard_safety must be a mapping")
    hard_fields = set(HardSafetyConfig.__dataclass_fields__)
    _require_keys(hard_raw, hard_fields, "hard_safety")
    hard_safety = HardSafetyConfig(
        **{name: float(hard_raw[name]) for name in hard_fields}
    )
    if hard_safety.max_mean_step_peak_torque_nm <= 0.0:
        raise ValueError("hard_safety.max_mean_step_peak_torque_nm must be positive")
    if not 0.0 <= hard_safety.max_action_saturation_fraction <= 1.0:
        raise ValueError("hard_safety.max_action_saturation_fraction must be in [0, 1]")
    if hard_safety.max_nonfinite_state < 0.0:
        raise ValueError("hard_safety.max_nonfinite_state must be non-negative")

    stages_raw = raw["stages"]
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("stages must be a non-empty list")
    stages: list[StageConfig] = []
    names: set[str] = set()
    for index, stage_raw in enumerate(stages_raw):
        label = f"stages[{index}]"
        if not isinstance(stage_raw, dict):
            raise ValueError(f"{label} must be a mapping")
        _require_keys(
            stage_raw,
            {
                "name",
                "max_cycles",
                "evaluation_commands_m_s",
                "required_gates",
                "overrides",
                "goal",
            },
            label,
        )
        name = str(stage_raw["name"])
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError(f"{label}.name must be a unique snake-case name")
        names.add(name)
        overrides = stage_raw["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(f"{label}.overrides must be a mapping")
        forbidden = sorted(set(overrides) - _SAFE_OVERRIDES)
        if forbidden:
            raise ValueError(
                f"{label}.overrides contains unsafe fields: {', '.join(forbidden)}"
            )
        evaluation_commands_raw = stage_raw["evaluation_commands_m_s"]
        if not isinstance(evaluation_commands_raw, list) or not evaluation_commands_raw:
            raise ValueError(f"{label}.evaluation_commands_m_s must be non-empty")
        evaluation_commands = tuple(
            float(command) for command in evaluation_commands_raw
        )
        if any(command <= 0.0 for command in evaluation_commands) or len(
            set(evaluation_commands)
        ) != len(evaluation_commands):
            raise ValueError(
                f"{label}.evaluation_commands_m_s must be unique and positive"
            )
        required_gates_raw = stage_raw["required_gates"]
        if not isinstance(required_gates_raw, list) or not required_gates_raw:
            raise ValueError(f"{label}.required_gates must be a non-empty list")
        required_gates = tuple(str(gate) for gate in required_gates_raw)
        if len(set(required_gates)) != len(required_gates):
            raise ValueError(f"{label}.required_gates must be unique")
        unknown_gates = sorted(set(required_gates) - WALKING_GATE_NAMES)
        if unknown_gates:
            raise ValueError(
                f"{label}.required_gates contains unknown gates: "
                f"{', '.join(unknown_gates)}"
            )
        stage_training_range = overrides.get(
            "environment.command_forward_range_m_s",
            list(training.environment.command_forward_range_m_s),
        )
        if not all(
            float(stage_training_range[0]) <= command <= float(stage_training_range[1])
            for command in evaluation_commands
        ):
            raise ValueError(
                f"{label}.evaluation_commands_m_s must stay inside its training range"
            )
        goal_raw = stage_raw["goal"]
        if not isinstance(goal_raw, dict):
            raise ValueError(f"{label}.goal must be a mapping")
        stages.append(
            StageConfig(
                name=name,
                max_cycles=_positive_int(
                    stage_raw["max_cycles"], f"{label}.max_cycles"
                ),
                evaluation_commands_m_s=evaluation_commands,
                required_gates=required_gates,
                overrides=copy.deepcopy(overrides),
                goal=WalkingGoal.from_mapping(goal_raw, f"{label}.goal"),
            )
        )

    final_range = stages[-1].overrides.get(
        "environment.command_forward_range_m_s",
        list(training.environment.command_forward_range_m_s),
    )
    if not all(
        float(final_range[0]) <= command <= float(final_range[1])
        for command in commands
    ):
        raise ValueError(
            "cycle.confirmation_commands_m_s must stay inside the final training range"
        )
    output_root = _repo_path(str(raw["agent_output_root"]), "agent_output_root")
    return WalkingAgentConfig(
        version=str(raw["version"]),
        base_training_config=base_config,
        agent_output_root=output_root,
        cycle=cycle,
        hard_safety=hard_safety,
        stages=tuple(stages),
    )


def _set_nested(root: dict[str, Any], dotted_name: str, value: Any) -> None:
    current = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_name}: {part} is not a mapping")
        current = child
    if parts[-1] not in current:
        raise ValueError(f"Cannot set unknown config field {dotted_name}")
    current[parts[-1]] = copy.deepcopy(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _run_streamed(command: Sequence[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def _cycle_payload(
    config: WalkingAgentConfig,
    stage: StageConfig,
    *,
    seed: int,
    stage_cycle: int,
) -> dict[str, Any]:
    payload = yaml.safe_load(config.base_training_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base training config must be a mapping")
    payload["seed"] = seed
    payload["version_name"] = (
        f"{payload['version_name']} | agent_stage={stage.name} cycle={stage_cycle}"
    )
    payload["ppo"]["num_timesteps"] = config.cycle.num_timesteps
    payload["ppo"]["num_evals"] = config.cycle.num_evals
    payload["ppo"]["evaluation_forward_command_m_s"] = stage.evaluation_commands_m_s[
        (stage_cycle - 1) % len(stage.evaluation_commands_m_s)
    ]
    for name, value in stage.overrides.items():
        _set_nested(payload, name, value)
    return payload


def _hard_safe(result: WalkingGoalResult, hard: HardSafetyConfig) -> bool:
    values = result.values
    return bool(
        values["mean_step_peak_torque_nm"] <= hard.max_mean_step_peak_torque_nm
        and values["action_saturation_fraction"] <= hard.max_action_saturation_fraction
        and values["nonfinite_state"] <= hard.max_nonfinite_state
    )


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WalkingAgentError(
                f"Invalid metrics JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
            raise WalkingAgentError(f"Invalid metrics row at {path}:{line_number}")
        rows.append(row)
    return rows


def select_cycle_candidate(
    run_directory: Path,
    stage: StageConfig,
    hard_safety: HardSafetyConfig,
    *,
    checkpoint_root: Path | None = None,
    require_local_checkpoint: bool = True,
) -> Candidate:
    """Select the highest walking-score checkpoint satisfying hard invariants."""
    training = load_training_config(run_directory / "training_config.yaml")
    checkpoint_root = checkpoint_root or run_directory / "checkpoints"
    checkpoint_suffix = ".pt" if training.ppo.backend == "rsl_rl" else ""
    candidates: list[Candidate] = []
    rejected: list[str] = []
    for row in _read_metric_rows(run_directory / "training_metrics.jsonl"):
        step = int(row.get("step", 0))
        if step <= 0:
            continue
        checkpoint = checkpoint_root / f"{step:012d}{checkpoint_suffix}"
        if require_local_checkpoint and not checkpoint.exists():
            rejected.append(f"step {step}: checkpoint missing")
            continue
        try:
            goal_result = evaluate_walking_goal(
                row["metrics"],
                stage.goal,
                target_episode_length=training.environment.episode_length,
                velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
            )
        except ValueError as exc:
            rejected.append(f"step {step}: {exc}")
            continue
        if not _hard_safe(goal_result, hard_safety):
            rejected.append(f"step {step}: hard servo/state invariant failed")
            continue
        candidates.append(
            Candidate(
                evaluation=int(row["evaluation"]),
                step=step,
                checkpoint=checkpoint,
                goal_result=goal_result,
                required_gates=stage.required_gates,
                metrics=dict(row["metrics"]),
            )
        )
    if not candidates:
        details = "; ".join(rejected[-5:]) or "no post-training evaluations"
        raise WalkingAgentError(
            f"No promotable checkpoint in {run_directory}: {details}"
        )
    return max(candidates, key=lambda candidate: candidate.rank)


def _failed_gates(result: WalkingGoalResult, gates: Sequence[str]) -> list[str]:
    return [name for name in gates if not result.gates[name]]


def _confirmation(
    *,
    checkpoint: Path,
    cycle_config: Path,
    goal: WalkingGoal,
    required_gates: Sequence[str],
    agent_config: WalkingAgentConfig,
    output: Path,
    allow_cpu: bool,
    dry_run: bool,
) -> tuple[bool, dict[str, Any]]:
    seeds = ",".join(str(seed) for seed in agent_config.cycle.confirmation_seeds)
    commands = ",".join(
        str(command) for command in agent_config.cycle.confirmation_commands_m_s
    )
    command = [
        sys.executable,
        "-m",
        "wr2.locomotion.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(cycle_config),
        "--seeds",
        seeds,
        "--commands",
        commands,
        "--num-envs",
        str(agent_config.cycle.confirmation_num_envs),
        "--output",
        str(output),
    ]
    if allow_cpu:
        command.append("--allow-cpu")
    print(f"Confirmation: {shlex.join(command)}")
    if dry_run:
        return False, {"dry_run": True, "command": command}
    return_code = _run_streamed(command, log_path=output.with_suffix(".log"))
    if return_code:
        raise WalkingAgentError(f"Confirmation evaluation exited with {return_code}")
    report = json.loads(output.read_text(encoding="utf-8"))
    training = load_training_config(cycle_config)
    decisions = []
    for command_result in report["command_results"]:
        for seed_result in command_result["seed_results"]:
            result = evaluate_walking_goal(
                seed_result["metrics"],
                goal,
                target_episode_length=training.environment.episode_length,
                velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
            )
            decisions.append(
                {
                    "command_forward_m_s": command_result["command_forward_m_s"],
                    "seed": seed_result["seed"],
                    **result.to_dict(),
                    "hard_safe": _hard_safe(result, agent_config.hard_safety),
                    "required_passed": result.passes(required_gates)
                    and _hard_safe(result, agent_config.hard_safety),
                    "p1_warnings": _failed_gates(
                        result,
                        tuple(sorted(WALKING_GATE_NAMES - set(required_gates))),
                    ),
                }
            )
    report["goal"] = walking_goal_to_dict(goal)
    report["required_gates"] = list(required_gates)
    report["goal_results"] = decisions
    report["passed"] = all(decision["required_passed"] for decision in decisions)
    _write_json_atomic(output, report)
    return bool(report["passed"]), report


def _new_agent_id() -> str:
    return f"wr2_walk_agent_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"


def _safe_agent_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError("agent ID may contain only letters, digits, '_' and '-'")
    return value


def run_local(args: argparse.Namespace, config: WalkingAgentConfig) -> int:
    """Run the bounded curriculum on the current machine."""
    agent_id = _safe_agent_id(args.agent_id or _new_agent_id())
    agent_root = config.agent_output_root / agent_id
    if agent_root.exists():
        raise WalkingAgentError(f"Agent output already exists: {agent_root}")
    configs_root = agent_root / "configs"
    logs_root = agent_root / "logs"
    agent_root.mkdir(parents=True)

    base_training = load_training_config(config.base_training_config)
    current_checkpoint = (
        Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    )
    if current_checkpoint is not None and not current_checkpoint.exists():
        raise WalkingAgentError(
            f"Initial checkpoint does not exist: {current_checkpoint}"
        )
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "agent_id": agent_id,
        "agent_config_version": config.version,
        "base_training_config": str(config.base_training_config),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "current_checkpoint": str(current_checkpoint) if current_checkpoint else None,
        "cycles": [],
    }
    _write_json_atomic(agent_root / "agent_state.json", state)

    global_cycle = 0
    for stage_index, stage in enumerate(config.stages):
        stage_best: Candidate | None = None
        for stage_cycle in range(1, stage.max_cycles + 1):
            global_cycle += 1
            seed = base_training.seed + global_cycle - 1
            cycle_payload = _cycle_payload(
                config, stage, seed=seed, stage_cycle=stage_cycle
            )
            cycle_config = configs_root / f"{global_cycle:02d}_{stage.name}.yaml"
            cycle_config.parent.mkdir(parents=True, exist_ok=True)
            cycle_config.write_text(
                yaml.safe_dump(cycle_payload, sort_keys=False), encoding="utf-8"
            )
            load_training_config(cycle_config)
            run_id = f"{agent_id}_{global_cycle:02d}_{stage.name}_seed{seed}"
            training_output = Path(cycle_payload["output"]["root"])
            if not training_output.is_absolute():
                training_output = REPO_ROOT / training_output
            run_directory = training_output / run_id
            command = [
                sys.executable,
                "-m",
                "wr2.locomotion.train",
                "--config",
                str(cycle_config),
                "--run-id",
                run_id,
            ]
            if current_checkpoint is not None:
                command.extend(["--restore-checkpoint", str(current_checkpoint)])
            if args.allow_cpu:
                command.append("--allow-cpu")
            print("=" * 72)
            print(
                f"Agent {agent_id}: stage {stage_index + 1}/{len(config.stages)} "
                f"{stage.name}, cycle {stage_cycle}/{stage.max_cycles}"
            )
            print(f"Training: {shlex.join(command)}")
            state["active_cycle"] = {
                "global_cycle": global_cycle,
                "stage": stage.name,
                "stage_cycle": stage_cycle,
                "seed": seed,
                "run_id": run_id,
                "config": str(cycle_config),
                "command": command,
            }
            _write_json_atomic(agent_root / "agent_state.json", state)
            if args.dry_run:
                state.update(status="dry_run", next_command=command)
                _write_json_atomic(agent_root / "agent_state.json", state)
                return 0
            return_code = _run_streamed(
                command, log_path=logs_root / f"cycle_{global_cycle:02d}.log"
            )
            if return_code:
                state.update(status="failed", error=f"training exited {return_code}")
                _write_json_atomic(agent_root / "agent_state.json", state)
                return return_code

            candidate = select_cycle_candidate(run_directory, stage, config.hard_safety)
            if stage_best is None or candidate.rank > stage_best.rank:
                stage_best = candidate
            current_checkpoint = stage_best.checkpoint
            cycle_record = {
                "global_cycle": global_cycle,
                "stage": stage.name,
                "stage_cycle": stage_cycle,
                "seed": seed,
                "run_id": run_id,
                "run_directory": str(run_directory),
                "selected_evaluation": candidate.evaluation,
                "selected_step": candidate.step,
                "selected_checkpoint": str(candidate.checkpoint),
                "selected_goal": candidate.goal_result.to_dict(),
                "selected_required_passed": candidate.required_passed,
                "selected_p1_warnings": _failed_gates(
                    candidate.goal_result,
                    tuple(sorted(WALKING_GATE_NAMES - set(stage.required_gates))),
                ),
                "stage_best_checkpoint": str(current_checkpoint),
            }
            state["cycles"].append(cycle_record)
            state.pop("active_cycle", None)
            state["current_checkpoint"] = str(current_checkpoint)
            state["stage"] = stage.name
            _write_json_atomic(agent_root / "agent_state.json", state)
            print(
                f"Selected step {candidate.step:,}: "
                f"walking_score={candidate.goal_result.score:.3f}; "
                f"P0_failed={_failed_gates(candidate.goal_result, stage.required_gates)}; "
                "P1_warnings="
                f"{_failed_gates(candidate.goal_result, tuple(sorted(WALKING_GATE_NAMES - set(stage.required_gates))))}"
            )

            if not candidate.required_passed:
                continue
            if stage_index < len(config.stages) - 1:
                print(f"Stage {stage.name} passed; advancing curriculum.")
                break

            confirmation_path = agent_root / "confirmation.json"
            passed, report = _confirmation(
                checkpoint=current_checkpoint,
                cycle_config=cycle_config,
                goal=stage.goal,
                required_gates=stage.required_gates,
                agent_config=config,
                output=confirmation_path,
                allow_cpu=args.allow_cpu,
                dry_run=False,
            )
            state["confirmation"] = report
            if passed:
                state.update(
                    status="complete",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    final_checkpoint=str(current_checkpoint),
                )
                _write_json_atomic(agent_root / "agent_state.json", state)
                print("Walking target passed on every confirmation seed.")
                print(f"Final checkpoint: {current_checkpoint}")
                return 0
            print("Confirmation failed; continuing robust training.")
        else:
            state.update(
                status="goal_not_met",
                stopped_at=datetime.now(timezone.utc).isoformat(),
                failed_stage=stage.name,
            )
            _write_json_atomic(agent_root / "agent_state.json", state)
            print(f"Stage {stage.name} exhausted its cycle budget without passing.")
            return 2

    state.update(status="goal_not_met", error="curriculum ended without confirmation")
    _write_json_atomic(agent_root / "agent_state.json", state)
    return 2


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _remote_command(args: argparse.Namespace, agent_id: str) -> tuple[list[str], str]:
    user = args.user
    host = args.host
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", host):
        raise ValueError("--host contains unsafe characters")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", user):
        raise ValueError("--user contains unsafe characters")
    target = f"{user}@{host}"
    remote_repo = PurePosixPath(args.remote_repo)
    if not remote_repo.is_absolute() or ".." in remote_repo.parts:
        raise ValueError("--remote-repo must be an absolute path without '..'")
    local_relative = args.agent_config.resolve().relative_to(REPO_ROOT).as_posix()
    remote_arguments = [
        "uv",
        "run",
        "--extra",
        "training",
        "python",
        "-m",
        "wr2.agents.walking_training_agent",
        "--agent-config",
        local_relative,
        "--agent-id",
        agent_id,
        "--local-worker",
    ]
    if args.resume_checkpoint:
        remote_arguments.extend(["--resume-checkpoint", args.resume_checkpoint])
    if args.allow_cpu:
        remote_arguments.append("--allow-cpu")
    if args.dry_run:
        remote_arguments.append("--dry-run")
    shell_command = (
        f"cd {shlex.quote(str(remote_repo))} && {shlex.join(remote_arguments)}"
    )
    ssh = ["ssh"]
    if args.port is not None:
        ssh.extend(["-p", str(args.port)])
    ssh.extend([target, shell_command])
    return ssh, target


def run_remote(args: argparse.Namespace) -> int:
    """Verify an exact checkout and stream the same agent on the GPU host."""
    dirty = _git_output("status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise WalkingAgentError(
            "Remote execution requires a clean committed worktree; commit and push first"
        )
    local_sha = _git_output("rev-parse", "HEAD")
    agent_id = _safe_agent_id(args.agent_id or _new_agent_id())
    command, target = _remote_command(args, agent_id)
    verify = ["ssh"]
    if args.port is not None:
        verify.extend(["-p", str(args.port)])
    verify.extend([target, f"git -C {shlex.quote(args.remote_repo)} rev-parse HEAD"])
    remote_sha = subprocess.run(
        verify, check=True, capture_output=True, text=True
    ).stdout.strip()
    if remote_sha != local_sha:
        raise WalkingAgentError(
            f"Remote HEAD {remote_sha[:12]} does not match local {local_sha[:12]}; "
            "push and pull the same commit before starting"
        )
    print(f"Launching {agent_id} on {target} at commit {local_sha[:12]}")
    print(shlex.join(command))
    if args.dry_run:
        return 0
    return int(subprocess.run(command, cwd=REPO_ROOT, check=False).returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--agent-id")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", help="Run the same supervisor on this SSH host")
    parser.add_argument("--user", default="leeygang")
    parser.add_argument("--port", type=int)
    parser.add_argument("--remote-repo", default="/home/leeygang/projects/WildRobot2")
    parser.add_argument("--local-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.agent_config = args.agent_config.resolve()
    config = load_agent_config(args.agent_config)
    if args.host and not args.local_worker:
        return run_remote(args)
    return run_local(args, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WalkingAgentError, ValueError) as exc:
        print(f"WalkingAgentError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
