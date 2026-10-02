"""Mac-owned train, analyze, improve, commit, and redeploy walking loop."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import selectors
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from wr2.agents.walking_training_agent import (
    DEFAULT_AGENT_CONFIG,
    Candidate,
    StageConfig,
    WalkingAgentConfig,
    WalkingAgentError,
    _cycle_payload,
    _failed_gates,
    _hard_safe,
    _read_metric_rows,
    _safe_agent_id,
    _write_json_atomic,
    load_agent_config,
    select_cycle_candidate,
)
from wr2.locomotion.configs import load_training_config
from wr2.locomotion.walking_metrics import (
    WALKING_GATE_NAMES,
    evaluate_walking_goal,
    walking_goal_to_dict,
)
from wr2.sim.robot import RobotDescription


REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_SCHEMA = Path(__file__).with_name("autonomous_decision.schema.json")
CODEX_INSTRUCTIONS = Path(__file__).with_name("AUTONOMOUS_CODEX_PROMPT.md")
INTERVENTION_FAMILIES = frozenset(
    {
        "reward_shaping",
        "gait_timing",
        "command_curriculum",
        "ppo_optimization",
        "domain_randomization",
        "environment_dynamics",
        "infrastructure",
    }
)
DECISION_FIELDS = frozenset(
    {
        "decision",
        "summary",
        "failure_mode",
        "hypothesis",
        "intervention_family",
        "expected_outcome",
        "falsification_condition",
        "toddlerbot_alignment",
        "config",
        "start_mode",
        "verification",
    }
)
FORBIDDEN_AUTONOMOUS_CHANGES = (
    "wr2/agents/",
    "wr2/descriptions/",
    "wr2/sim/",
    "wr2/actuation/",
    "wr2/sensing/",
    "pyproject.toml",
    "uv.lock",
)
ALLOWED_AUTONOMOUS_CHANGES = (
    "wr2/locomotion/",
    "wr2/reference/",
    "tests/",
    "docs/",
    "README.md",
)
TRAINING_RELEVANT_CHANGES = (
    "wr2/locomotion/",
    "wr2/reference/",
)
REMOTE_RESULT_FILES = (
    "training_config.yaml",
    "training_metrics.jsonl",
    "run_config.json",
)


class AutonomousWalkingError(WalkingAgentError):
    """Failure that requires operator review before autonomous work continues."""


@dataclass(frozen=True)
class RemoteContext:
    host: str
    user: str
    port: int | None
    repository: PurePosixPath

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_prefix(self) -> list[str]:
        command = ["ssh", *_ssh_connection_options()]
        if self.port is not None:
            command.extend(["-p", str(self.port)])
        command.append(self.target)
        return command

    def scp_prefix(self) -> list[str]:
        command = ["scp", *_ssh_connection_options()]
        if self.port is not None:
            command.extend(["-P", str(self.port)])
        return command


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_agent_id() -> str:
    return f"wr2_auto_walk_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"


def _ssh_connection_options() -> list[str]:
    return [
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=600",
        "-o",
        "ControlPath=/tmp/wr2-walking-agent-%C",
    ]


def _git(
    *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _git_output(*arguments: str) -> str:
    return _git(*arguments).stdout.strip()


def _require_clean_branch(branch: str) -> str:
    current = _git_output("branch", "--show-current")
    if current != branch:
        raise AutonomousWalkingError(
            f"Autonomous training requires branch {branch!r}, not {current!r}"
        )
    dirty = _git_output("status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise AutonomousWalkingError(
            "Autonomous training requires a clean Git worktree"
        )
    return _git_output("rev-parse", "HEAD")


def _validate_remote(args: argparse.Namespace) -> RemoteContext:
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", args.host):
        raise ValueError("--host contains unsafe characters")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.user):
        raise ValueError("--user contains unsafe characters")
    if args.port is not None and not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", args.remote_repo):
        raise ValueError(
            "--remote-repo must be an absolute path without spaces or '..'"
        )
    repository = PurePosixPath(args.remote_repo)
    if ".." in repository.parts:
        raise ValueError("--remote-repo must not contain '..'")
    return RemoteContext(args.host, args.user, args.port, repository)


def _run_streamed(
    command: Sequence[str],
    *,
    log_path: Path,
    timeout_s: int | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
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
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                if timeout_s is not None and time.monotonic() - started > timeout_s:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise AutonomousWalkingError(
                        f"Command timed out after {timeout_s}s: {shlex.join(command)}"
                    )
                for key, _ in selector.select(timeout=1.0):
                    line = key.fileobj.readline()
                    if line:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
            remainder = process.stdout.read()
            if remainder:
                print(remainder, end="", flush=True)
                log.write(remainder)
            return int(process.wait())
        finally:
            selector.close()


def _run_checked(command: Sequence[str], label: str) -> None:
    result = subprocess.run(
        list(command), cwd=REPO_ROOT, check=False, text=True, capture_output=True
    )
    if result.returncode:
        output = (result.stderr or result.stdout).strip()
        raise AutonomousWalkingError(f"{label} failed: {output}")


def _remote_output(context: RemoteContext, shell_command: str) -> str:
    result = subprocess.run(
        [*context.ssh_prefix(), shell_command],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AutonomousWalkingError(
            f"Remote command failed: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def _remote_sync_checkout(
    context: RemoteContext, *, branch: str, expected_sha: str, log_path: Path
) -> None:
    repository = shlex.quote(str(context.repository))
    dirty = _remote_output(
        context,
        f"git -C {repository} status --porcelain --untracked-files=normal",
    )
    if dirty:
        raise AutonomousWalkingError(
            "GPU repository has uncommitted or untracked files: "
            + ", ".join(dirty.splitlines())
        )
    remote_branch = _remote_output(
        context, f"git -C {repository} branch --show-current"
    )
    if remote_branch != branch:
        raise AutonomousWalkingError(
            f"GPU repository is on branch {remote_branch!r}, not {branch!r}"
        )
    command = (
        f"git -C {repository} pull --ff-only origin {shlex.quote(branch)}"
    )
    return_code = _run_streamed(
        [*context.ssh_prefix(), command], log_path=log_path
    )
    if return_code:
        raise AutonomousWalkingError(
            f"GPU checkout could not fast-forward (exit {return_code})"
        )
    actual_sha = _remote_output(
        context, f"git -C {repository} rev-parse HEAD"
    )
    if actual_sha != expected_sha:
        raise AutonomousWalkingError(
            f"GPU HEAD {actual_sha[:12]} does not match Mac HEAD "
            f"{expected_sha[:12]}"
        )


def _remote_mkdir(context: RemoteContext, directory: PurePosixPath) -> None:
    _remote_output(context, f"mkdir -p {shlex.quote(str(directory))}")


def _copy_to_remote(
    context: RemoteContext, local_path: Path, remote_path: PurePosixPath
) -> None:
    command = [
        *context.scp_prefix(),
        str(local_path),
        f"{context.target}:{remote_path}",
    ]
    _run_checked(command, f"copying {local_path.name} to the GPU")


def _copy_from_remote(
    context: RemoteContext, remote_path: PurePosixPath, local_path: Path
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *context.scp_prefix(),
        f"{context.target}:{remote_path}",
        str(local_path),
    ]
    _run_checked(command, f"copying {remote_path.name} from the GPU")


def _push(branch: str) -> None:
    result = _git("push", "origin", f"HEAD:{branch}", check=False)
    if result.returncode:
        raise AutonomousWalkingError(
            f"git push failed: {(result.stderr or result.stdout).strip()}"
        )


def _relative_repo_path(path: Path, label: str) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise AutonomousWalkingError(f"{label} must be inside the repository") from exc


def _remote_repo_path(
    context: RemoteContext, relative_path: Path
) -> PurePosixPath:
    return context.repository.joinpath(*relative_path.parts)


def _walking_env_contract_fingerprints() -> dict[str, str]:
    path = REPO_ROOT / "wr2/locomotion/walking_env.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    environment = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WR2WalkingEnv"
    )
    semantic_helpers = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "normalized_action_rate_cost",
            "support_contact_active",
            "feet_lateral_distance",
        }
    }
    if len(semantic_helpers) != 3:
        raise AutonomousWalkingError(
            "Could not locate the complete walking semantic helpers"
        )
    methods = {
        node.name: node
        for node in environment.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    observation = methods["_observation"]
    step = methods["step"]
    action_prefix = []
    for statement in step.body:
        action_prefix.append(statement)
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pipeline_state"
            for target in statement.targets
        ):
            break
    if not action_prefix or not isinstance(action_prefix[-1], ast.Assign):
        raise AutonomousWalkingError(
            "Could not locate the frozen action-to-physics boundary"
        )

    def digest(node: ast.AST) -> str:
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    zmp_path = REPO_ROOT / "wr2/reference/walk_zmp.py"
    zmp_tree = ast.parse(
        zmp_path.read_text(encoding="utf-8"), filename=str(zmp_path)
    )
    return {
        "observation_method_sha256": digest(observation),
        "normalized_action_rate_cost_sha256": digest(
            semantic_helpers["normalized_action_rate_cost"]
        ),
        "support_contact_active_sha256": digest(
            semantic_helpers["support_contact_active"]
        ),
        "feet_lateral_distance_sha256": digest(
            semantic_helpers["feet_lateral_distance"]
        ),
        "action_pipeline_prefix_sha256": digest(
            ast.Module(body=action_prefix, type_ignores=[])
        ),
        "zmp_reference_sha256": digest(zmp_tree),
    }


def _metric_acceptance_fingerprint() -> str:
    path = REPO_ROOT / "wr2/locomotion/walking_metrics.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    protected_names = {
        "WALKING_GATE_NAMES",
        "WalkingGoal",
        "WalkingGoalResult",
        "_METRIC_NAMES",
        "_scalar",
        "evaluate_walking_goal",
    }
    protected_nodes: list[ast.stmt] = []
    for node in tree.body:
        name = getattr(node, "name", None)
        assigned_names = {
            target.id
            for target in getattr(node, "targets", [])
            if isinstance(target, ast.Name)
        }
        if name in protected_names or assigned_names & protected_names:
            protected_nodes.append(node)
    if len(protected_nodes) != len(protected_names):
        raise AutonomousWalkingError(
            "Could not locate the complete walking acceptance implementation"
        )
    normalized = ast.dump(
        ast.Module(body=protected_nodes, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _contract_snapshot(training_config: Path) -> dict[str, Any]:
    training = load_training_config(training_config)
    robot = RobotDescription.load(include_local_calibration=False)
    return {
        "actor_observation": robot.config["observation"],
        "action": robot.config["action"],
        "actuator_order": list(robot.actuator_names),
        "servo_model": robot.config["actuators"]["htd45hServo"],
        "active_groups": list(training.environment.active_groups),
        "action_rate_semantics": "sum_squared_consecutive_normalized_policy_actions",
        "privileged_linear_velocity_scale": (
            training.environment.privileged_linear_velocity_scale
        ),
        "privileged_actuator_force_scale": (
            training.environment.privileged_actuator_force_scale
        ),
        "zmp_reference": asdict(training.environment.zmp_reference),
        "normalize_observations": training.ppo.normalize_observations,
        "output": asdict(training.output),
        "walking_env_source": _walking_env_contract_fingerprints(),
        "walking_metric_acceptance_sha256": _metric_acceptance_fingerprint(),
    }


def _acceptance_contract(config: WalkingAgentConfig) -> dict[str, Any]:
    """Return final success, hard-safety, and campaign-budget invariants."""
    return {
        "base_training_config": _relative_repo_path(
            config.base_training_config, "base training config"
        ).as_posix(),
        "agent_output_root": _relative_repo_path(
            config.agent_output_root, "agent output root"
        ).as_posix(),
        "num_timesteps_per_cycle": config.cycle.num_timesteps,
        "confirmation_commands_m_s": list(
            config.cycle.confirmation_commands_m_s
        ),
        "confirmation_seeds": list(config.cycle.confirmation_seeds),
        "confirmation_num_envs": config.cycle.confirmation_num_envs,
        "hard_safety": asdict(config.hard_safety),
        "stages": [
            {
                "name": stage.name,
                "max_cycles": stage.max_cycles,
                "required_gates": list(stage.required_gates),
                "goal": walking_goal_to_dict(stage.goal),
            }
            for stage in config.stages
        ],
    }


def _training_compatibility(config: WalkingAgentConfig) -> dict[str, Any]:
    """Identify changes that invalidate the existing Brax parameter tree."""
    training = load_training_config(config.base_training_config)
    ppo_source = (REPO_ROOT / "wr2/locomotion/ppo.py").read_bytes()
    return {
        "network": asdict(training.network),
        "ppo_source_sha256": hashlib.sha256(ppo_source).hexdigest(),
    }


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "evaluation": candidate.evaluation,
        "step": candidate.step,
        "checkpoint": str(candidate.checkpoint),
        "required_passed": candidate.required_passed,
        "walking_score": candidate.goal_result.score,
        "failed_p0_gates": _failed_gates(
            candidate.goal_result, candidate.required_gates
        ),
        "p1_warnings": _failed_gates(
            candidate.goal_result,
            tuple(sorted(WALKING_GATE_NAMES - set(candidate.required_gates))),
        ),
        "goal_result": candidate.goal_result.to_dict(),
        "metrics": candidate.metrics,
    }


def _analyze_cycle(
    run_directory: Path,
    remote_checkpoint_root: PurePosixPath,
    stage: StageConfig,
    config: WalkingAgentConfig,
) -> tuple[Candidate | None, dict[str, Any]]:
    training = load_training_config(run_directory / "training_config.yaml")
    evaluations: list[dict[str, Any]] = []
    for row in _read_metric_rows(run_directory / "training_metrics.jsonl"):
        step = int(row.get("step", 0))
        if step <= 0:
            continue
        record: dict[str, Any] = {
            "evaluation": int(row.get("evaluation", 0)),
            "step": step,
        }
        try:
            result = evaluate_walking_goal(
                row["metrics"],
                stage.goal,
                target_episode_length=training.environment.episode_length,
                velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
            )
        except ValueError as exc:
            record.update(promotable=False, error=str(exc))
        else:
            record.update(
                promotable=_hard_safe(result, config.hard_safety),
                required_passed=result.passes(stage.required_gates),
                failed_p0_gates=_failed_gates(result, stage.required_gates),
                p1_warnings=_failed_gates(
                    result,
                    tuple(sorted(WALKING_GATE_NAMES - set(stage.required_gates))),
                ),
                goal_result=result.to_dict(),
            )
        evaluations.append(record)

    candidate: Candidate | None
    selection_error: str | None = None
    try:
        candidate = select_cycle_candidate(
            run_directory,
            stage,
            config.hard_safety,
            checkpoint_root=Path(str(remote_checkpoint_root)),
            require_local_checkpoint=False,
        )
    except WalkingAgentError as exc:
        candidate = None
        selection_error = str(exc)
    report = {
        "schema_version": 1,
        "stage": stage.name,
        "required_gates": list(stage.required_gates),
        "goal": walking_goal_to_dict(stage.goal),
        "hard_safety": asdict(config.hard_safety),
        "run_directory": str(run_directory),
        "remote_checkpoint_root": str(remote_checkpoint_root),
        "evaluations": evaluations,
        "selected": _candidate_payload(candidate) if candidate else None,
        "selection_error": selection_error,
    }
    return candidate, report


def _score_confirmation(
    report: dict[str, Any],
    *,
    config_path: Path,
    stage: StageConfig,
    agent_config: WalkingAgentConfig,
) -> dict[str, Any]:
    training = load_training_config(config_path)
    decisions = []
    for command_result in report["command_results"]:
        for seed_result in command_result["seed_results"]:
            result = evaluate_walking_goal(
                seed_result["metrics"],
                stage.goal,
                target_episode_length=training.environment.episode_length,
                velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
            )
            decisions.append(
                {
                    "command_forward_m_s": command_result["command_forward_m_s"],
                    "seed": seed_result["seed"],
                    **result.to_dict(),
                    "hard_safe": _hard_safe(result, agent_config.hard_safety),
                    "required_passed": result.passes(stage.required_gates)
                    and _hard_safe(result, agent_config.hard_safety),
                    "failed_p0_gates": _failed_gates(
                        result, stage.required_gates
                    ),
                    "p1_warnings": _failed_gates(
                        result,
                        tuple(
                            sorted(WALKING_GATE_NAMES - set(stage.required_gates))
                        ),
                    ),
                }
            )
    report["goal"] = walking_goal_to_dict(stage.goal)
    report["required_gates"] = list(stage.required_gates)
    report["goal_results"] = decisions
    report["passed"] = bool(decisions) and all(
        decision["required_passed"] for decision in decisions
    )
    return report


def _validate_changed_files(changed_files: Sequence[str]) -> None:
    forbidden = [
        path
        for path in changed_files
        if any(
            path == prefix or path.startswith(prefix)
            for prefix in FORBIDDEN_AUTONOMOUS_CHANGES
        )
    ]
    if forbidden:
        raise AutonomousWalkingError(
            "Codex modified frozen campaign files: " + ", ".join(forbidden)
        )
    outside_allowlist = [
        path
        for path in changed_files
        if not any(
            path == prefix or path.startswith(prefix)
            for prefix in ALLOWED_AUTONOMOUS_CHANGES
        )
    ]
    if outside_allowlist:
        raise AutonomousWalkingError(
            "Codex modified files outside the autonomous allowlist: "
            + ", ".join(outside_allowlist)
        )
    if not any(
        path == prefix or path.startswith(prefix)
        for path in changed_files
        for prefix in TRAINING_RELEVANT_CHANGES
    ):
        raise AutonomousWalkingError(
            "Codex commit does not contain a training-relevant change"
        )


def _validate_decision_shape(
    decision: Mapping[str, Any], expected_config: Path
) -> None:
    actual_fields = set(decision)
    if actual_fields != DECISION_FIELDS:
        missing = sorted(DECISION_FIELDS - actual_fields)
        unknown = sorted(actual_fields - DECISION_FIELDS)
        raise AutonomousWalkingError(
            f"Invalid Codex decision fields; missing={missing}, unknown={unknown}"
        )
    if decision["decision"] != "continue":
        raise AutonomousWalkingError("Codex decision must be 'continue'")
    for field in DECISION_FIELDS - {"decision", "verification"}:
        if not isinstance(decision[field], str) or not decision[field].strip():
            raise AutonomousWalkingError(
                f"Codex decision field {field!r} must be non-empty"
            )
    if decision["intervention_family"] not in INTERVENTION_FAMILIES:
        raise AutonomousWalkingError("Codex returned an unknown intervention family")
    if decision["start_mode"] not in {"warm_start", "cold_start"}:
        raise AutonomousWalkingError(
            "Codex start_mode must be 'warm_start' or 'cold_start'"
        )
    verification = decision["verification"]
    if not isinstance(verification, list) or not verification or not all(
        isinstance(item, str) and item.strip() for item in verification
    ):
        raise AutonomousWalkingError(
            "Codex verification must contain at least one command/result"
        )
    requested_config = Path(str(decision["config"]))
    if requested_config.is_absolute() or ".." in requested_config.parts:
        raise AutonomousWalkingError(
            "Codex config must be a repository-relative tracked path"
        )
    config = (REPO_ROOT / requested_config).resolve()
    if config != expected_config.resolve():
        raise AutonomousWalkingError(
            "Codex must keep using the campaign's tracked base training config"
        )


def _validate_codex_commit(
    *,
    before_sha: str,
    branch: str,
    decision: Mapping[str, Any],
    agent_config_path: Path,
    base_training_config: Path,
    frozen_contract: Mapping[str, Any],
    frozen_acceptance: Mapping[str, Any],
    before_training_compatibility: Mapping[str, Any],
    verification_log: Path,
) -> tuple[str, WalkingAgentConfig, bool]:
    after_sha = _require_clean_branch(branch)
    if after_sha == before_sha:
        raise AutonomousWalkingError("Codex did not create the required commit")
    if _git(
        "merge-base", "--is-ancestor", before_sha, after_sha, check=False
    ).returncode:
        raise AutonomousWalkingError("Codex rewrote Git history")
    commit_count = int(_git_output("rev-list", "--count", f"{before_sha}..{after_sha}"))
    if commit_count != 1:
        raise AutonomousWalkingError(
            f"Codex must create exactly one commit; created {commit_count}"
        )
    parents = _git_output("rev-list", "--parents", "-n", "1", after_sha).split()
    if len(parents) != 2:
        raise AutonomousWalkingError("Codex commit must not be a merge commit")
    changed_files = _git_output(
        "diff", "--name-only", before_sha, after_sha
    ).splitlines()
    _validate_changed_files(changed_files)
    _validate_decision_shape(decision, base_training_config)
    current_contract = _contract_snapshot(base_training_config)
    if current_contract != frozen_contract:
        raise AutonomousWalkingError(
            "Codex changed the frozen action, observation, metric-gate, or servo "
            "contract"
        )
    updated_agent_config = load_agent_config(agent_config_path)
    if _acceptance_contract(updated_agent_config) != frozen_acceptance:
        raise AutonomousWalkingError(
            "Codex changed final P0 gates, hard safety, confirmation, or the "
            "bounded campaign budget"
        )
    after_training_compatibility = _training_compatibility(updated_agent_config)
    requires_cold_start = after_training_compatibility != before_training_compatibility
    if requires_cold_start and decision["start_mode"] != "cold_start":
        raise AutonomousWalkingError(
            "The policy parameter contract changed; Codex must request cold_start"
        )
    test_command = [
        shutil.which("uv") or "uv",
        "run",
        "python",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
    ]
    return_code = _run_streamed(test_command, log_path=verification_log)
    if return_code:
        raise AutonomousWalkingError(
            f"Post-Codex unit tests failed with exit code {return_code}"
        )
    if _git_output("status", "--porcelain", "--untracked-files=normal"):
        raise AutonomousWalkingError("Verification left the Git worktree dirty")
    return after_sha, updated_agent_config, decision["start_mode"] == "cold_start"


def _codex_prompt(
    *,
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    confirmation: Mapping[str, Any] | None,
    base_training_config: Path,
    toddlerbot_repo: Path,
) -> str:
    instructions = CODEX_INSTRUCTIONS.read_text(encoding="utf-8")
    context = {
        "git_sha": state["git_sha"],
        "agent_id": state["agent_id"],
        "global_cycle": state["global_cycle"],
        "stage": state["stage"],
        "stage_cycle": state["stage_cycle"],
        "base_training_config": _relative_repo_path(
            base_training_config, "base training config"
        ).as_posix(),
        "selected_champion_checkpoint": state.get("current_checkpoint"),
        "cycle_analysis": analysis,
        "confirmation": confirmation,
        "experiment_history": state.get("experiment_history", []),
        "frozen_deployment_contract": state["frozen_contract"],
        "frozen_acceptance_contract": state["frozen_acceptance"],
        "current_training_compatibility": state["training_compatibility"],
        "toddlerbot_source": str(toddlerbot_repo),
        "wr1_training_agent_reference": str(
            Path.home()
            / "projects/wildrobot/wildrobot/agents/autonomous_training_loop.py"
        ),
    }
    return instructions + "\n\nIteration context:\n" + json.dumps(
        context, indent=2, sort_keys=True
    )


def _invoke_codex(
    args: argparse.Namespace,
    *,
    state: dict[str, Any],
    analysis: Mapping[str, Any],
    confirmation: Mapping[str, Any] | None,
    base_training_config: Path,
    cycle_root: Path,
) -> dict[str, Any]:
    codex_path = shutil.which(args.codex_path)
    if codex_path is None:
        raise AutonomousWalkingError(
            f"Codex executable not found on the Mac: {args.codex_path}"
        )
    decision_path = cycle_root / "codex_decision.json"
    log_path = cycle_root / "codex_exec.log"
    decision_path.unlink(missing_ok=True)
    command = [
        codex_path,
        "exec",
        "--approve-for-me",
        "--ephemeral",
        "--output-schema",
        str(DECISION_SCHEMA),
        "--output-last-message",
        str(decision_path),
        "--cd",
        str(REPO_ROOT),
    ]
    if args.codex_model:
        command.extend(["--model", args.codex_model])
    command.append(
        _codex_prompt(
            state=state,
            analysis=analysis,
            confirmation=confirmation,
            base_training_config=base_training_config,
            toddlerbot_repo=args.toddlerbot_repo,
        )
    )
    print("Mac: asking Codex for one bounded improvement...", flush=True)
    return_code = _run_streamed(
        command,
        log_path=log_path,
        timeout_s=args.codex_timeout_minutes * 60,
    )
    if return_code:
        raise AutonomousWalkingError(
            f"codex exec exited with {return_code}; see {log_path}"
        )
    if not decision_path.is_file():
        raise AutonomousWalkingError("Codex did not write its structured decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not isinstance(decision, dict):
        raise AutonomousWalkingError("Codex decision must be a JSON object")
    return decision


def _cycle_paths(
    *,
    config: WalkingAgentConfig,
    context: RemoteContext,
    agent_root: Path,
    run_id: str,
    global_cycle: int,
    stage: StageConfig,
) -> tuple[Path, PurePosixPath, Path, PurePosixPath]:
    config_relative = _relative_repo_path(config.agent_output_root, "agent output root")
    local_config = (
        agent_root / "configs" / f"{global_cycle:02d}_{stage.name}.yaml"
    )
    remote_config = _remote_repo_path(
        context,
        config_relative
        / agent_root.name
        / "configs"
        / local_config.name,
    )
    training = load_training_config(config.base_training_config)
    output_root = Path(training.output.root)
    if output_root.is_absolute() or ".." in output_root.parts:
        raise AutonomousWalkingError(
            "Training output.root must be a safe repository-relative path"
        )
    local_run = REPO_ROOT / output_root / run_id
    remote_run = _remote_repo_path(context, output_root / run_id)
    return local_config, remote_config, local_run, remote_run


def _run_remote_training(
    context: RemoteContext,
    *,
    remote_config: PurePosixPath,
    run_id: str,
    checkpoint: str | None,
    log_path: Path,
) -> int:
    arguments = [
        "uv",
        "run",
        "--extra",
        "training",
        "python",
        "-m",
        "wr2.locomotion.train",
        "--config",
        str(remote_config),
        "--run-id",
        run_id,
    ]
    if checkpoint:
        arguments.extend(["--restore-checkpoint", checkpoint])
    shell_command = (
        f"cd {shlex.quote(str(context.repository))} && {shlex.join(arguments)}"
    )
    print(f"GPU training: {shell_command}", flush=True)
    return _run_streamed(
        [*context.ssh_prefix(), shell_command], log_path=log_path
    )


def _sync_cycle_result(
    context: RemoteContext,
    *,
    remote_run: PurePosixPath,
    local_run: Path,
) -> None:
    for filename in REMOTE_RESULT_FILES:
        _copy_from_remote(
            context, remote_run / filename, local_run / filename
        )


def _run_remote_confirmation(
    context: RemoteContext,
    *,
    config: WalkingAgentConfig,
    remote_config: PurePosixPath,
    checkpoint: str,
    remote_output: PurePosixPath,
    local_output: Path,
    log_path: Path,
) -> dict[str, Any]:
    arguments = [
        "uv",
        "run",
        "--extra",
        "training",
        "python",
        "-m",
        "wr2.locomotion.evaluate",
        "--checkpoint",
        checkpoint,
        "--config",
        str(remote_config),
        "--seeds",
        ",".join(str(seed) for seed in config.cycle.confirmation_seeds),
        "--commands",
        ",".join(
            str(command) for command in config.cycle.confirmation_commands_m_s
        ),
        "--num-envs",
        str(config.cycle.confirmation_num_envs),
        "--output",
        str(remote_output),
    ]
    shell_command = (
        f"cd {shlex.quote(str(context.repository))} && {shlex.join(arguments)}"
    )
    print("GPU: running independent multi-speed confirmation...", flush=True)
    return_code = _run_streamed(
        [*context.ssh_prefix(), shell_command], log_path=log_path
    )
    if return_code:
        raise AutonomousWalkingError(
            f"GPU confirmation exited with {return_code}"
        )
    _copy_from_remote(context, remote_output, local_output)
    report = json.loads(local_output.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise AutonomousWalkingError("Confirmation output is not a JSON object")
    return report


def _prepare_state(
    *,
    args: argparse.Namespace,
    config: WalkingAgentConfig,
    agent_id: str,
    agent_root: Path,
) -> dict[str, Any]:
    initial_sha = _require_clean_branch(args.branch)
    state = {
        "schema_version": 1,
        "status": "active",
        "machine": "Mac",
        "agent_id": agent_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "branch": args.branch,
        "git_sha": initial_sha,
        "host": args.host,
        "user": args.user,
        "port": args.port,
        "remote_repo": args.remote_repo,
        "agent_config": _relative_repo_path(
            args.agent_config, "agent config"
        ).as_posix(),
        "agent_config_version": config.version,
        "base_training_config": _relative_repo_path(
            config.base_training_config, "base training config"
        ).as_posix(),
        "frozen_contract": _contract_snapshot(config.base_training_config),
        "frozen_acceptance": _acceptance_contract(config),
        "training_compatibility": _training_compatibility(config),
        "global_cycle": 0,
        "stage": None,
        "stage_cycle": 0,
        "current_checkpoint": args.resume_checkpoint,
        "cycles": [],
        "experiment_history": [],
    }
    _write_json_atomic(agent_root / "autonomous_state.json", state)
    return state


def _save_state(agent_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_json_atomic(agent_root / "autonomous_state.json", state)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutonomousWalkingError(f"Cannot read agent state {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise AutonomousWalkingError(f"Agent state is not a JSON object: {path}")
    return state


def _select_status_root(
    output_root: Path, *, agent_id: str | None, latest: bool
) -> Path:
    if bool(agent_id) == latest:
        raise AutonomousWalkingError(
            "status requires exactly one of --agent-id ID or --latest"
        )
    if agent_id:
        root = output_root / _safe_agent_id(agent_id)
        if not (root / "autonomous_state.json").is_file():
            raise AutonomousWalkingError(f"Autonomous agent state not found: {root}")
        return root
    candidates = [
        state_path.parent
        for state_path in output_root.glob("*/autonomous_state.json")
        if state_path.is_file()
    ]
    if not candidates:
        raise AutonomousWalkingError(
            f"No autonomous agent state found under {output_root}"
        )
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _tail_progress(path: Path, *, max_bytes: int = 131_072) -> list[str]:
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - max_bytes))
        text = stream.read().decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    progress_indices = [
        index for index, line in enumerate(lines) if re.match(r"#\d+\s", line)
    ]
    if progress_indices:
        start = progress_indices[-1]
        return lines[start : start + 6]
    return lines[-8:]


def _status_payload(agent_root: Path) -> dict[str, Any]:
    state_path = agent_root / "autonomous_state.json"
    state = _read_state(state_path)
    cycle_number = int(state.get("global_cycle") or 0)
    stage = state.get("stage")
    cycle_root = (
        agent_root / "cycles" / f"{cycle_number:02d}_{stage}"
        if cycle_number > 0 and stage
        else None
    )
    preferred_log = {
        "training": "gpu_training.log",
        "improving": "codex_exec.log",
    }.get(str(state.get("status")))
    active_log = cycle_root / preferred_log if cycle_root and preferred_log else None
    if active_log is None or not active_log.is_file():
        logs = sorted(
            agent_root.glob("cycles/*/*.log"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        active_log = logs[-1] if logs else None
    if active_log:
        try:
            active_log_display = str(active_log.relative_to(REPO_ROOT))
        except ValueError:
            active_log_display = str(active_log)
    else:
        active_log_display = None
    latest_cycle = state.get("cycles", [])[-1] if state.get("cycles") else None
    run_id = state.get("run_id")
    if run_id is None and isinstance(latest_cycle, dict):
        run_id = latest_cycle.get("run_id")
    payload = {
        "agent_id": state.get("agent_id", agent_root.name),
        "state_path": str(state_path),
        "status": state.get("status", "unknown"),
        "updated_at": state.get("updated_at"),
        "stage": stage,
        "stage_cycle": state.get("stage_cycle"),
        "global_cycle": cycle_number,
        "run_id": run_id,
        "git_sha": state.get("git_sha"),
        "current_checkpoint": state.get("current_checkpoint"),
        "final_checkpoint": state.get("final_checkpoint"),
        "confirmation_passed": (
            state.get("confirmation", {}).get("passed")
            if isinstance(state.get("confirmation"), dict)
            else None
        ),
        "last_cycle": latest_cycle,
        "last_decision": state.get("last_decision"),
        "error": state.get("error"),
        "active_log": active_log_display,
        "latest_progress": _tail_progress(active_log) if active_log else [],
    }
    return payload


def _status(args: argparse.Namespace) -> int:
    args.agent_config = args.agent_config.resolve()
    config = load_agent_config(args.agent_config)
    agent_root = _select_status_root(
        config.agent_output_root,
        agent_id=args.agent_id,
        latest=args.latest,
    )
    payload = _status_payload(agent_root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Agent:       {payload['agent_id']}")
    print(f"Status:      {payload['status']}")
    print(
        "Stage:       "
        f"{payload['stage']} (stage cycle {payload['stage_cycle']}, "
        f"global cycle {payload['global_cycle']})"
    )
    print(f"Run:         {payload['run_id'] or '-'}")
    print(f"Git:         {str(payload['git_sha'] or '-')[:12]}")
    checkpoint = payload["final_checkpoint"] or payload["current_checkpoint"]
    print(f"Checkpoint:  {checkpoint or '-'}")
    print(f"Updated:     {payload['updated_at'] or '-'}")
    if payload["active_log"]:
        print(f"Active log:  {payload['active_log']}")
    if payload["latest_progress"]:
        print("Latest progress:")
        for line in payload["latest_progress"]:
            print(f"  {line}")
    if payload["error"]:
        print(f"Error:       {payload['error']}")
    return 0


def _record_codex_improvement(
    args: argparse.Namespace,
    *,
    state: dict[str, Any],
    config: WalkingAgentConfig,
    analysis: Mapping[str, Any],
    confirmation: Mapping[str, Any] | None,
    cycle_root: Path,
    agent_root: Path,
) -> tuple[WalkingAgentConfig, bool]:
    before_sha = _require_clean_branch(args.branch)
    before_training_compatibility = _training_compatibility(config)
    state.update(
        status="improving",
        git_sha=before_sha,
        training_compatibility=before_training_compatibility,
    )
    _save_state(agent_root, state)
    decision = _invoke_codex(
        args,
        state=state,
        analysis=analysis,
        confirmation=confirmation,
        base_training_config=config.base_training_config,
        cycle_root=cycle_root,
    )
    after_sha, updated_config, requires_cold_start = _validate_codex_commit(
        before_sha=before_sha,
        branch=args.branch,
        decision=decision,
        agent_config_path=args.agent_config,
        base_training_config=config.base_training_config,
        frozen_contract=state["frozen_contract"],
        frozen_acceptance=state["frozen_acceptance"],
        before_training_compatibility=before_training_compatibility,
        verification_log=cycle_root / "post_codex_tests.log",
    )
    _push(args.branch)
    state["experiment_history"].append(
        {
            "cycle": state["global_cycle"],
            "source_git_sha": before_sha,
            "improvement_git_sha": after_sha,
            **decision,
        }
    )
    if requires_cold_start:
        state["current_checkpoint"] = None
    state.update(
        status="active",
        git_sha=after_sha,
        last_decision=decision,
        training_compatibility=_training_compatibility(updated_config),
    )
    _save_state(agent_root, state)
    print(
        f"Mac: validated and pushed Codex commit {after_sha[:12]}: "
        f"{decision['summary']}",
        flush=True,
    )
    return updated_config, requires_cold_start


def _run_campaign(args: argparse.Namespace) -> int:
    args.agent_config = args.agent_config.resolve()
    config = load_agent_config(args.agent_config)
    context = _validate_remote(args)
    agent_id = _safe_agent_id(args.agent_id or _new_agent_id())
    agent_root = config.agent_output_root / agent_id
    if agent_root.exists():
        raise AutonomousWalkingError(f"Agent output already exists: {agent_root}")
    current_checkpoint = args.resume_checkpoint
    if current_checkpoint:
        checkpoint = PurePosixPath(current_checkpoint)
        allowed_root = context.repository / "results/wr2_walking"
        if not checkpoint.is_absolute() or not checkpoint.is_relative_to(allowed_root):
            raise AutonomousWalkingError(
                "--resume-checkpoint must be an absolute path under the GPU "
                f"repository's results/wr2_walking directory: {allowed_root}"
            )
    agent_root.mkdir(parents=True)
    state = _prepare_state(
        args=args, config=config, agent_id=agent_id, agent_root=agent_root
    )
    base_training = load_training_config(config.base_training_config)
    global_cycle = 0

    try:
        for stage_index in range(len(config.stages)):
            stage = config.stages[stage_index]
            stage_best: Candidate | None = None
            for stage_cycle in range(1, stage.max_cycles + 1):
                global_cycle += 1
                seed = base_training.seed + global_cycle - 1
                run_id = (
                    f"{agent_id}_{global_cycle:02d}_{stage.name}_seed{seed}"
                )
                cycle_root = agent_root / "cycles" / f"{global_cycle:02d}_{stage.name}"
                cycle_root.mkdir(parents=True)
                local_config, remote_config, local_run, remote_run = _cycle_paths(
                    config=config,
                    context=context,
                    agent_root=agent_root,
                    run_id=run_id,
                    global_cycle=global_cycle,
                    stage=stage,
                )
                cycle_payload = _cycle_payload(
                    config, stage, seed=seed, stage_cycle=stage_cycle
                )
                local_config.parent.mkdir(parents=True, exist_ok=True)
                local_config.write_text(
                    yaml.safe_dump(cycle_payload, sort_keys=False), encoding="utf-8"
                )
                load_training_config(local_config)
                state.update(
                    status="training",
                    global_cycle=global_cycle,
                    stage=stage.name,
                    stage_index=stage_index,
                    stage_cycle=stage_cycle,
                    run_id=run_id,
                    git_sha=_require_clean_branch(args.branch),
                    current_checkpoint=current_checkpoint,
                )
                _save_state(agent_root, state)

                print("=" * 72)
                print(
                    f"Autonomous {agent_id}: {stage.name} "
                    f"cycle {stage_cycle}/{stage.max_cycles}"
                )
                print(f"Mac commit: {state['git_sha'][:12]}")
                if args.dry_run:
                    print(
                        "Dry run: would push, fast-forward the GPU, transfer "
                        f"{local_config}, and train {run_id}."
                    )
                    state.update(status="dry_run")
                    _save_state(agent_root, state)
                    return 0

                _push(args.branch)
                _remote_sync_checkout(
                    context,
                    branch=args.branch,
                    expected_sha=state["git_sha"],
                    log_path=cycle_root / "remote_git_sync.log",
                )
                if current_checkpoint:
                    checkpoint_status = _remote_output(
                        context,
                        f"test -d {shlex.quote(current_checkpoint)} && printf present",
                    )
                    if checkpoint_status != "present":
                        raise AutonomousWalkingError(
                            f"GPU checkpoint is missing: {current_checkpoint}"
                        )
                _remote_mkdir(context, remote_config.parent)
                _copy_to_remote(context, local_config, remote_config)
                return_code = _run_remote_training(
                    context,
                    remote_config=remote_config,
                    run_id=run_id,
                    checkpoint=current_checkpoint,
                    log_path=cycle_root / "gpu_training.log",
                )
                if return_code:
                    raise AutonomousWalkingError(
                        f"GPU training exited with {return_code}"
                    )

                state.update(status="analyzing")
                _save_state(agent_root, state)
                _sync_cycle_result(
                    context, remote_run=remote_run, local_run=local_run
                )
                candidate, analysis = _analyze_cycle(
                    local_run,
                    remote_run / "checkpoints",
                    stage,
                    config,
                )
                analysis.update(
                    run_id=run_id,
                    git_sha=state["git_sha"],
                    global_cycle=global_cycle,
                    stage_cycle=stage_cycle,
                )
                _write_json_atomic(cycle_root / "analysis.json", analysis)

                if candidate is not None:
                    remote_exists = _remote_output(
                        context,
                        f"test -d {shlex.quote(str(candidate.checkpoint))} && "
                        "printf present",
                    )
                    if remote_exists != "present":
                        raise AutonomousWalkingError(
                            f"Selected GPU checkpoint is missing: {candidate.checkpoint}"
                        )
                    if stage_best is None or candidate.rank > stage_best.rank:
                        stage_best = candidate
                    current_checkpoint = str(stage_best.checkpoint)
                    print(
                        f"Mac analysis: selected step {candidate.step:,}; "
                        f"score={candidate.goal_result.score:.3f}; "
                        f"P0_failed={analysis['selected']['failed_p0_gates']}; "
                        f"P1={analysis['selected']['p1_warnings']}",
                        flush=True,
                    )
                else:
                    print(
                        f"Mac analysis: no hard-safe checkpoint; "
                        f"{analysis['selection_error']}",
                        flush=True,
                    )

                cycle_record = {
                    "global_cycle": global_cycle,
                    "stage": stage.name,
                    "stage_cycle": stage_cycle,
                    "run_id": run_id,
                    "git_sha": state["git_sha"],
                    "analysis": str(
                        (cycle_root / "analysis.json").relative_to(REPO_ROOT)
                    ),
                    "selected": analysis["selected"],
                    "promoted_checkpoint": current_checkpoint,
                }
                state["cycles"].append(cycle_record)
                state["current_checkpoint"] = current_checkpoint
                state.pop("run_id", None)
                _save_state(agent_root, state)

                required_passed = bool(
                    candidate is not None and candidate.required_passed
                )
                if required_passed and stage_index < len(config.stages) - 1:
                    print(f"P0 acquisition passed; advancing past {stage.name}.")
                    break

                confirmation: dict[str, Any] | None = None
                if required_passed:
                    remote_confirmation = (
                        _remote_repo_path(
                            context,
                            _relative_repo_path(
                                config.agent_output_root, "agent output root"
                            )
                            / agent_id
                            / "confirmations"
                            / f"cycle_{global_cycle:02d}.json",
                        )
                    )
                    _remote_mkdir(context, remote_confirmation.parent)
                    local_confirmation = cycle_root / "confirmation.json"
                    raw_confirmation = _run_remote_confirmation(
                        context,
                        config=config,
                        remote_config=remote_config,
                        checkpoint=current_checkpoint,
                        remote_output=remote_confirmation,
                        local_output=local_confirmation,
                        log_path=cycle_root / "gpu_confirmation.log",
                    )
                    confirmation = _score_confirmation(
                        raw_confirmation,
                        config_path=local_config,
                        stage=stage,
                        agent_config=config,
                    )
                    _write_json_atomic(local_confirmation, confirmation)
                    state["confirmation"] = confirmation
                    _save_state(agent_root, state)
                    if confirmation["passed"]:
                        state.update(
                            status="complete",
                            completed_at=_utc_now(),
                            final_checkpoint=current_checkpoint,
                        )
                        _save_state(agent_root, state)
                        print("Walking P0 passed at 0.10, 0.15, and 0.20 m/s.")
                        print(f"Final GPU checkpoint: {current_checkpoint}")
                        return 0
                    print("Independent confirmation failed P0.", flush=True)

                if stage_cycle >= stage.max_cycles:
                    state.update(
                        status="goal_not_met",
                        stopped_at=_utc_now(),
                        failed_stage=stage.name,
                    )
                    _save_state(agent_root, state)
                    print(
                        f"Stage {stage.name} exhausted its bounded cycle budget."
                    )
                    return 2

                config, cold_start = _record_codex_improvement(
                    args,
                    state=state,
                    config=config,
                    analysis=analysis,
                    confirmation=confirmation,
                    cycle_root=cycle_root,
                    agent_root=agent_root,
                )
                stage = config.stages[stage_index]
                if cold_start:
                    current_checkpoint = None
                    stage_best = None
        state.update(
            status="goal_not_met",
            stopped_at=_utc_now(),
            error="curriculum ended without confirmation",
        )
        _save_state(agent_root, state)
        return 2
    except BaseException as exc:
        state.update(status="failed", stopped_at=_utc_now(), error=str(exc))
        _save_state(agent_root, state)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("run", "status"), default="run"
    )
    parser.add_argument("--agent-config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--agent-id")
    parser.add_argument(
        "--latest", action="store_true", help="Select the newest autonomous agent"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable status JSON"
    )
    parser.add_argument("--host", default="linux-pc.local")
    parser.add_argument("--user", default="leeygang")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--remote-repo", default="/home/leeygang/projects/WildRobot2"
    )
    parser.add_argument("--branch", default="main")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--codex-path", default="codex")
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-timeout-minutes", type=int, default=120)
    parser.add_argument(
        "--toddlerbot-repo",
        type=Path,
        default=Path.home() / "projects/toddlerbot",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.codex_timeout_minutes <= 0:
        parser.error("--codex-timeout-minutes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "status":
        return _status(args)
    if args.latest or args.json:
        raise AutonomousWalkingError("--latest and --json are status-only options")
    return _run_campaign(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AutonomousWalkingError, ValueError, json.JSONDecodeError) as exc:
        print(f"AutonomousWalkingError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
