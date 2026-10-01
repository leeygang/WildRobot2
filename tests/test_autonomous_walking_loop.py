import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path, PurePosixPath

import yaml

from wr2.agents.autonomous_walking_loop import (
    AutonomousWalkingError,
    DECISION_FIELDS,
    _acceptance_contract,
    _analyze_cycle,
    _contract_snapshot,
    _score_confirmation,
    _select_status_root,
    _status_payload,
    _tail_progress,
    _training_compatibility,
    _validate_changed_files,
    _validate_decision_shape,
    _validate_remote,
    parse_args,
)
from wr2.agents.walking_training_agent import (
    DEFAULT_AGENT_CONFIG,
    load_agent_config,
)
from wr2.locomotion.configs import (
    load_training_config,
    training_config_to_dict,
)


def _metrics(**updates):
    result = {
        "eval/avg_episode_length": 1000.0,
        "eval/episode_fall": 0.0,
        "eval/episode_command_forward_m_s_per_step": 0.15,
        "eval/episode_forward_velocity_m_s_per_step": 0.145,
        "eval/episode_forward_velocity_error_m_s_per_step": 0.005,
        "eval/episode_lateral_velocity_m_s_per_step": 0.005,
        "eval/episode_yaw_rate_error_rad_s_per_step": 0.01,
        "eval/episode_contact_phase_match_per_step": 0.90,
        "eval/episode_double_support_per_step": 0.10,
        "eval/episode_feet_phase_tracking_per_step": 1.40,
        "eval/episode_actuator_torque_rms_nm_per_step": 0.80,
        "eval/episode_actuator_torque_peak_nm_per_step": 2.50,
        "eval/episode_sustained_torque_exposure_per_step": 0.80,
        "eval/episode_action_saturation_fraction_per_step": 0.01,
        "eval/episode_nonfinite_state": 0.0,
    }
    result.update(updates)
    return result


def _decision(config: str) -> dict:
    return {
        "decision": "continue",
        "summary": "Increase contact-phase reward for the next bounded cycle.",
        "failure_mode": "Contact phase is below its P0 threshold.",
        "hypothesis": "The phase reward is too weak relative to velocity tracking.",
        "intervention_family": "reward_shaping",
        "expected_outcome": "Contact match rises without increasing saturation.",
        "falsification_condition": "Contact match does not improve next cycle.",
        "toddlerbot_alignment": (
            "Matches ToddlerBot's active phase-guided contact mechanism."
        ),
        "config": config,
        "start_mode": "warm_start",
        "verification": ["unit tests passed"],
    }


class AutonomousWalkingLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_agent_config(DEFAULT_AGENT_CONFIG)

    def test_remote_context_rejects_relative_or_parent_paths(self):
        base = dict(
            host="linux-pc.local",
            user="leeygang",
            port=None,
            remote_repo="relative/path",
        )
        with self.assertRaises(ValueError):
            _validate_remote(Namespace(**base))
        base["remote_repo"] = "/home/leeygang/../other"
        with self.assertRaises(ValueError):
            _validate_remote(Namespace(**base))

    def test_remote_context_builds_shared_ssh_connection(self):
        context = _validate_remote(
            Namespace(
                host="linux-pc.local",
                user="leeygang",
                port=2222,
                remote_repo="/home/leeygang/projects/WildRobot2",
            )
        )
        self.assertEqual(context.target, "leeygang@linux-pc.local")
        self.assertIn("ControlMaster=auto", context.ssh_prefix())
        self.assertEqual(context.repository.name, "WildRobot2")

    def test_frozen_contract_includes_deployment_and_servo_fields(self):
        contract = _contract_snapshot(self.config.base_training_config)
        self.assertEqual(contract["actor_observation"]["layout_id"], "wr2_proprio_v3")
        self.assertEqual(contract["actor_observation"]["history_frames"], 15)
        self.assertEqual(
            contract["action"]["representation"], "joint_position_residual"
        )
        self.assertEqual(contract["active_groups"], ["leg"])
        self.assertEqual(contract["servo_model"]["torque_limit_nm"], 4.0)
        self.assertEqual(
            len(contract["walking_env_source"]["observation_method_sha256"]), 64
        )
        self.assertEqual(len(contract["walking_metric_acceptance_sha256"]), 64)

    def test_acceptance_is_frozen_but_network_is_training_compatible(self):
        acceptance = _acceptance_contract(self.config)
        compatibility = _training_compatibility(self.config)
        self.assertEqual(acceptance["confirmation_commands_m_s"], [0.10, 0.15, 0.20])
        self.assertEqual(
            acceptance["stages"][-1]["required_gates"],
            [
                "episode_length",
                "fall_rate",
                "forward_velocity_ratio",
                "forward_velocity_error",
                "contact_phase",
                "action_saturation",
            ],
        )
        self.assertEqual(compatibility["network"]["distribution_type"], "tanh_normal")

    def test_codex_change_allowlist_requires_training_change(self):
        _validate_changed_files(
            [
                "wr2/locomotion/configs/ppo_walking.yaml",
                "wr2/locomotion/train.py",
                "wr2/locomotion/evaluate.py",
                "wr2/locomotion/walking_metrics.py",
                "wr2/locomotion/ppo.py",
                "wr2/locomotion/configs/training_config.py",
                "wr2/locomotion/configs/walking_agent.yaml",
                "tests/test_autonomous_walking_loop.py",
                "tests/test_training_interface.py",
                "docs/design/walking_training_agent.md",
            ]
        )
        with self.assertRaises(AutonomousWalkingError):
            _validate_changed_files(["docs/design/walking_training_agent.md"])
        with self.assertRaises(AutonomousWalkingError):
            _validate_changed_files(["wr2/agents/autonomous_walking_loop.py"])
        with self.assertRaises(AutonomousWalkingError):
            _validate_changed_files(["scripts/scp_from_remote.sh"])

    def test_status_cli_selects_latest_or_named_agent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            older = output_root / "older"
            newer = output_root / "newer"
            for root in (older, newer):
                root.mkdir()
                (root / "autonomous_state.json").write_text("{}")
            os.utime(older / "autonomous_state.json", ns=(1, 1))
            os.utime(newer / "autonomous_state.json", ns=(2, 2))

            self.assertEqual(
                _select_status_root(output_root, agent_id=None, latest=True),
                newer,
            )
            self.assertEqual(
                _select_status_root(output_root, agent_id="older", latest=False),
                older,
            )
            with self.assertRaises(AutonomousWalkingError):
                _select_status_root(output_root, agent_id=None, latest=False)

        args = parse_args(["status", "--latest", "--json"])
        self.assertEqual(args.command, "status")
        self.assertTrue(args.latest)
        self.assertTrue(args.json)

    def test_status_payload_reports_latest_gpu_progress(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            agent_root = Path(temporary_directory) / "agent-one"
            cycle_root = agent_root / "cycles/01_gait_acquisition"
            cycle_root.mkdir(parents=True)
            (agent_root / "autonomous_state.json").write_text(
                json.dumps(
                    {
                        "agent_id": "agent-one",
                        "status": "training",
                        "stage": "gait_acquisition",
                        "stage_cycle": 1,
                        "global_cycle": 1,
                        "run_id": "run-one",
                        "git_sha": "a" * 40,
                        "cycles": [],
                    }
                )
            )
            log = cycle_root / "gpu_training.log"
            log.write_text(
                "starting\n#0  [00:01:00] Steps: 1/10\n"
                "  └─ return: episode=1\n"
                "#1  [00:02:00] Steps: 2/10\n"
                "  └─ return: episode=2\n"
            )

            payload = _status_payload(agent_root)

        self.assertEqual(payload["status"], "training")
        self.assertEqual(payload["run_id"], "run-one")
        self.assertEqual(
            payload["latest_progress"],
            ["#1  [00:02:00] Steps: 2/10", "  └─ return: episode=2"],
        )

    def test_tail_progress_does_not_treat_markdown_heading_as_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log = Path(temporary_directory) / "codex.log"
            log.write_text("# heading\nline one\nline two\n")
            self.assertEqual(_tail_progress(log), ["# heading", "line one", "line two"])

    def test_structured_decision_is_strict_and_keeps_base_config(self):
        relative_config = self.config.base_training_config.relative_to(
            Path(__file__).resolve().parents[1]
        ).as_posix()
        decision = _decision(relative_config)
        self.assertEqual(set(decision), DECISION_FIELDS)
        _validate_decision_shape(decision, self.config.base_training_config)
        decision["config"] = "wr2/locomotion/configs/another.yaml"
        with self.assertRaises(AutonomousWalkingError):
            _validate_decision_shape(decision, self.config.base_training_config)
        decision = _decision(relative_config)
        decision["start_mode"] = "resume_optimizer"
        with self.assertRaises(AutonomousWalkingError):
            _validate_decision_shape(decision, self.config.base_training_config)

    def test_mac_analysis_uses_remote_checkpoint_without_downloading_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            training = load_training_config(self.config.base_training_config)
            (run / "training_config.yaml").write_text(
                yaml.safe_dump(training_config_to_dict(training), sort_keys=False),
                encoding="utf-8",
            )
            rows = [
                {"evaluation": 0, "step": 0, "metrics": _metrics()},
                {"evaluation": 1, "step": 100, "metrics": _metrics()},
            ]
            (run / "training_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            remote_root = PurePosixPath(
                "/home/leeygang/projects/WildRobot2/results/run/checkpoints"
            )
            candidate, report = _analyze_cycle(
                run, remote_root, self.config.stages[-1], self.config
            )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.checkpoint, Path(str(remote_root / "000000000100")))
        self.assertTrue(report["selected"]["required_passed"])

    def test_confirmation_requires_every_command_and_seed(self):
        command_results = []
        for command in (0.10, 0.15, 0.20):
            command_results.append(
                {
                    "command_forward_m_s": command,
                    "seed_results": [
                        {
                            "seed": seed,
                            "metrics": _metrics(
                                **{
                                    "eval/episode_command_forward_m_s_per_step": command,
                                    "eval/episode_forward_velocity_m_s_per_step": command,
                                    "eval/episode_forward_velocity_error_m_s_per_step": 0.0,
                                }
                            ),
                        }
                        for seed in (101, 202, 303)
                    ],
                }
            )
        report = _score_confirmation(
            {"command_results": command_results},
            config_path=self.config.base_training_config,
            stage=self.config.stages[-1],
            agent_config=self.config,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["goal_results"]), 9)
        report["command_results"][0]["seed_results"][0]["metrics"][
            "eval/episode_fall"
        ] = 0.02
        failed = _score_confirmation(
            {"command_results": report["command_results"]},
            config_path=self.config.base_training_config,
            stage=self.config.stages[-1],
            agent_config=self.config,
        )
        self.assertFalse(failed["passed"])

        first_metrics = report["command_results"][0]["seed_results"][0]["metrics"]
        first_metrics["eval/episode_fall"] = 0.0
        first_metrics["eval/episode_actuator_torque_peak_nm_per_step"] = 4.1
        unsafe = _score_confirmation(
            {"command_results": report["command_results"]},
            config_path=self.config.base_training_config,
            stage=self.config.stages[-1],
            agent_config=self.config,
        )
        self.assertFalse(unsafe["passed"])
        self.assertFalse(unsafe["goal_results"][0]["hard_safe"])


if __name__ == "__main__":
    unittest.main()
