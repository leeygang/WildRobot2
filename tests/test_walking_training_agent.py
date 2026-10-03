import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from wr2.agents.walking_training_agent import (
    DEFAULT_AGENT_CONFIG,
    _cycle_payload,
    load_agent_config,
    run_local,
    select_cycle_candidate,
)
from wr2.locomotion.configs import (
    load_training_config,
    training_config_to_dict,
)
from wr2.locomotion.walking_metrics import evaluate_walking_goal


def _good_metrics(**updates):
    metrics = {
        "eval/avg_episode_length": 1000.0,
        "eval/episode_fall": 0.0,
        "eval/episode_command_forward_m_s_per_step": 0.15,
        "eval/episode_forward_velocity_m_s_per_step": 0.145,
        "eval/episode_forward_velocity_error_m_s_per_step": 0.005,
        "eval/episode_lateral_velocity_m_s_per_step": 0.005,
        "eval/episode_yaw_rate_error_rad_s_per_step": 0.01,
        "eval/episode_contact_phase_match_per_step": 0.90,
        "eval/episode_left_foot_contact_per_step": 0.65,
        "eval/episode_right_foot_contact_per_step": 0.65,
        "eval/episode_double_support_per_step": 0.10,
        "eval/episode_feet_phase_tracking_per_step": 1.40,
        "eval/episode_actuator_torque_rms_nm_per_step": 0.80,
        "eval/episode_actuator_torque_peak_nm_per_step": 2.50,
        "eval/episode_sustained_torque_exposure_per_step": 0.80,
        "eval/episode_action_saturation_fraction_per_step": 0.01,
        "eval/episode_nonfinite_state": 0.0,
    }
    metrics.update(updates)
    return metrics


class WalkingTrainingAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_config = load_agent_config(DEFAULT_AGENT_CONFIG)

    def test_campaign_uses_high_emergency_cycle_ceiling(self):
        self.assertEqual(
            [stage.max_cycles for stage in self.agent_config.stages],
            [2000, 2000],
        )
        self.assertEqual(
            [stage.name for stage in self.agent_config.stages],
            ["gait_acquisition", "robust_walking"],
        )

    def test_stage_override_does_not_change_policy_contract(self):
        base = load_training_config(self.agent_config.base_training_config)
        payload = _cycle_payload(
            self.agent_config,
            self.agent_config.stages[0],
            seed=4,
            stage_cycle=1,
        )
        self.assertEqual(payload["seed"], 4)
        self.assertFalse(payload["domain_randomization"]["enabled"])
        self.assertEqual(payload["ppo"]["num_timesteps"], 100_000_000)
        self.assertEqual(payload["ppo"]["evaluation_forward_command_m_s"], 0.10)
        self.assertEqual(
            payload["network"]["policy_hidden_layer_sizes"],
            list(base.network.policy_hidden_layer_sizes),
        )
        self.assertNotIn("action_rate_reference_rad", payload["environment"])
        self.assertEqual(
            payload["environment"]["privileged_linear_velocity_scale"], 2.0
        )
        self.assertEqual(payload["environment"]["privileged_actuator_force_scale"], 0.1)

    def test_robust_stage_rotates_requested_evaluation_commands(self):
        robust = self.agent_config.stages[-1]
        observed = [
            _cycle_payload(
                self.agent_config,
                robust,
                seed=cycle,
                stage_cycle=cycle,
            )["ppo"]["evaluation_forward_command_m_s"]
            for cycle in range(1, 7)
        ]
        self.assertEqual(observed, [0.10, 0.15, 0.20, 0.10, 0.15, 0.20])

    def test_final_velocity_gate_scales_with_command(self):
        training = load_training_config(self.agent_config.base_training_config)
        result = evaluate_walking_goal(
            _good_metrics(
                **{
                    "eval/episode_command_forward_m_s_per_step": 0.20,
                    "eval/episode_forward_velocity_m_s_per_step": 0.18,
                    "eval/episode_forward_velocity_error_m_s_per_step": 0.02,
                }
            ),
            self.agent_config.stages[-1].goal,
            target_episode_length=training.environment.episode_length,
            velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
        )
        self.assertTrue(result.gates["forward_velocity_ratio"])
        self.assertTrue(result.gates["forward_velocity_error"])
        self.assertAlmostEqual(result.values["forward_velocity_ratio"], 0.9)

    def test_final_goal_rejects_stationary_double_support(self):
        training = load_training_config(self.agent_config.base_training_config)
        metrics = _good_metrics(
            **{
                "eval/episode_forward_velocity_m_s_per_step": 0.0,
                "eval/episode_forward_velocity_error_m_s_per_step": 0.15,
                "eval/episode_contact_phase_match_per_step": 0.5,
                "eval/episode_double_support_per_step": 1.0,
            }
        )
        result = evaluate_walking_goal(
            metrics,
            self.agent_config.stages[-1].goal,
            target_episode_length=training.environment.episode_length,
            velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.gates["walking_score"])
        self.assertFalse(result.gates["forward_velocity_ratio"])
        self.assertFalse(result.gates["double_support"])

    def test_final_goal_rejects_one_planted_foot(self):
        training = load_training_config(self.agent_config.base_training_config)
        result = evaluate_walking_goal(
            _good_metrics(
                **{
                    "eval/episode_left_foot_contact_per_step": 0.40,
                    "eval/episode_right_foot_contact_per_step": 0.999,
                }
            ),
            self.agent_config.stages[-1].goal,
            target_episode_length=training.environment.episode_length,
            velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
        )
        self.assertFalse(result.gates["bilateral_foot_use"])
        self.assertAlmostEqual(result.values["minimum_foot_swing_fraction"], 0.001)

    def test_p1_diagnostics_do_not_block_required_stability_gates(self):
        training = load_training_config(self.agent_config.base_training_config)
        stage = self.agent_config.stages[-1]
        result = evaluate_walking_goal(
            _good_metrics(
                **{
                    "eval/episode_lateral_velocity_m_s_per_step": 0.05,
                    "eval/episode_yaw_rate_error_rad_s_per_step": 0.20,
                    "eval/episode_double_support_per_step": 0.40,
                    "eval/episode_feet_phase_tracking_per_step": 0.90,
                    "eval/episode_actuator_torque_rms_nm_per_step": 1.10,
                    "eval/episode_actuator_torque_peak_nm_per_step": 3.20,
                    "eval/episode_sustained_torque_exposure_per_step": 1.20,
                }
            ),
            stage.goal,
            target_episode_length=training.environment.episode_length,
            velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.passes(stage.required_gates))

    def test_candidate_selection_rejects_hard_torque_violation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            training = load_training_config(self.agent_config.base_training_config)
            (run / "training_config.yaml").write_text(
                yaml.safe_dump(training_config_to_dict(training), sort_keys=False),
                encoding="utf-8",
            )
            rows = [
                {
                    "evaluation": 1,
                    "step": 100,
                    "metrics": _good_metrics(),
                },
                {
                    "evaluation": 2,
                    "step": 200,
                    "metrics": _good_metrics(
                        **{
                            "eval/episode_actuator_torque_peak_nm_per_step": 4.1,
                            "eval/episode_forward_velocity_error_m_s_per_step": 0.0,
                        }
                    ),
                },
            ]
            (run / "training_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            for step in (100, 200):
                (run / "checkpoints" / f"{step:012d}").mkdir(parents=True)

            candidate = select_cycle_candidate(
                run,
                self.agent_config.stages[-1],
                self.agent_config.hard_safety,
            )

        self.assertEqual(candidate.step, 100)
        self.assertTrue(candidate.goal_result.passed)

    def test_candidate_selection_prioritizes_p0_gate_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            training = load_training_config(self.agent_config.base_training_config)
            (run / "training_config.yaml").write_text(
                yaml.safe_dump(training_config_to_dict(training), sort_keys=False),
                encoding="utf-8",
            )
            almost_p0 = _good_metrics(
                **{
                    "eval/episode_forward_velocity_m_s_per_step": 0.131,
                    "eval/episode_forward_velocity_error_m_s_per_step": 0.019,
                    "eval/episode_contact_phase_match_per_step": 0.79,
                    "eval/episode_double_support_per_step": 0.20,
                }
            )
            high_score_but_two_p0_failures = _good_metrics(
                **{
                    "eval/episode_fall": 0.02,
                    "eval/episode_forward_velocity_m_s_per_step": 0.15,
                    "eval/episode_forward_velocity_error_m_s_per_step": 0.0,
                    "eval/episode_contact_phase_match_per_step": 0.99,
                    "eval/episode_double_support_per_step": 0.0,
                    "eval/episode_action_saturation_fraction_per_step": 0.03,
                }
            )
            rows = [
                {"evaluation": 1, "step": 100, "metrics": almost_p0},
                {
                    "evaluation": 2,
                    "step": 200,
                    "metrics": high_score_but_two_p0_failures,
                },
            ]
            (run / "training_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            for step in (100, 200):
                (run / "checkpoints" / f"{step:012d}").mkdir(parents=True)

            candidate = select_cycle_candidate(
                run,
                self.agent_config.stages[-1],
                self.agent_config.hard_safety,
            )

        self.assertGreater(
            evaluate_walking_goal(
                high_score_but_two_p0_failures,
                self.agent_config.stages[-1].goal,
                target_episode_length=training.environment.episode_length,
                velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
            ).score,
            evaluate_walking_goal(
                almost_p0,
                self.agent_config.stages[-1].goal,
                target_episode_length=training.environment.episode_length,
                velocity_sigma_m_s=training.environment.velocity_tracking_sigma,
            ).score,
        )
        self.assertEqual(candidate.step, 100)

    def test_local_dry_run_writes_reproducible_state_and_cycle_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            config = replace(self.agent_config, agent_output_root=temporary_root)
            result = run_local(
                SimpleNamespace(
                    agent_id="dry_run_agent",
                    resume_checkpoint=None,
                    allow_cpu=True,
                    dry_run=True,
                ),
                config,
            )
            agent_root = temporary_root / "dry_run_agent"
            state = json.loads((agent_root / "agent_state.json").read_text())
            generated = load_training_config(
                agent_root / "configs/01_gait_acquisition.yaml"
            )

        self.assertEqual(result, 0)
        self.assertEqual(state["status"], "dry_run")
        self.assertEqual(generated.ppo.num_timesteps, 100_000_000)
        self.assertFalse(generated.environment.randomization.enabled)


if __name__ == "__main__":
    unittest.main()
