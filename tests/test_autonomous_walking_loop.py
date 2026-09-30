import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path, PurePosixPath

import yaml

from wr2.agents.autonomous_walking_loop import (
    AutonomousWalkingError,
    DECISION_FIELDS,
    _analyze_cycle,
    _contract_snapshot,
    _score_confirmation,
    _validate_changed_files,
    _validate_decision_shape,
    _validate_remote,
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
        "config": config,
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
        self.assertEqual(contract["actor_observation"]["layout_id"], "wr2_proprio_v2")
        self.assertEqual(contract["action"]["representation"], "joint_position_residual")
        self.assertEqual(contract["active_groups"], ["leg"])
        self.assertEqual(contract["servo_model"]["torque_limit_nm"], 4.0)
        self.assertEqual(contract["network"]["distribution_type"], "normal")
        self.assertEqual(
            len(contract["walking_env_source"]["observation_method_sha256"]), 64
        )

    def test_codex_change_allowlist_requires_training_change(self):
        _validate_changed_files(
            [
                "wr2/locomotion/configs/ppo_walking.yaml",
                "docs/design/walking_training_agent.md",
            ]
        )
        with self.assertRaises(AutonomousWalkingError):
            _validate_changed_files(["docs/design/walking_training_agent.md"])
        with self.assertRaises(AutonomousWalkingError):
            _validate_changed_files(["wr2/agents/autonomous_walking_loop.py"])
        with self.assertRaises(AutonomousWalkingError):
            _validate_changed_files(["scripts/scp_from_remote.sh"])

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
