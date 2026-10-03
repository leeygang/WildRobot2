from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from wr2.actuation.htd45h import (
    CMD_MOVE_STOP,
    CMD_OR_MOTOR_MODE_READ,
    Htd45hBus,
    build_packet,
    checksum,
    parse_packets,
)
from wr2.tools.servo_sysid.analysis.position import summarize_series
from wr2.tools.servo_sysid.analysis.hysteresis import (
    summarize_arrays as summarize_hysteresis,
)
from wr2.tools.servo_sysid.analyze import analyze_campaigns
from wr2.tools.servo_sysid.campaign import (
    CONDITIONS,
    capture_command,
    configured_plan,
    main as campaign_main,
    selected_conditions,
)
from wr2.tools.servo_sysid.capture import (
    VoltageWarningMonitor,
    _check_health,
    _monitored_samples_to_buffer,
    _wait_for_cooldown,
    capture_profile,
)
from wr2.tools.servo_sysid.core import (
    ProfileSegment,
    build_hysteresis_profile,
    build_profile,
    radians_to_units,
    units_to_radians,
)
from wr2.tools.servo_sysid.plan import available_plans, load_plan


class Htd45hProtocolTest(unittest.TestCase):
    def test_cooldown_reports_status_and_completion(self):
        class FakeBus:
            def read_voltage_v(self, _servo_id):
                return 12.4

            def read_temperature_c(self, _servo_id):
                return 30

        output = io.StringIO()
        with redirect_stdout(output):
            samples = _wait_for_cooldown(
                FakeBus(),
                100,
                target_c=30.0,
                timeout_s=60.0,
                poll_s=5.0,
                min_voltage_v=9.6,
            )

        self.assertEqual(len(samples), 1)
        self.assertIn("COOLDOWN STATUS", output.getvalue())
        self.assertIn("COOLDOWN COMPLETE", output.getvalue())

    def test_temperature_limit_is_inclusive(self):
        healthy = {"voltage_v": 11.8, "temperature_c": 54.0, "loaded": True}
        _check_health(healthy, min_voltage_v=9.6, max_temperature_c=55.0)

        at_limit = {"voltage_v": 11.8, "temperature_c": 55.0, "loaded": True}
        with self.assertRaisesRegex(RuntimeError, "reaches or exceeds"):
            _check_health(at_limit, min_voltage_v=9.6, max_temperature_c=55.0)

    def test_low_voltage_warns_in_yellow_until_hard_floor(self):
        monitor = VoltageWarningMonitor(9.6)
        output = io.StringIO()
        low = {"voltage_v": 8.574, "temperature_c": 35.0, "loaded": True}
        recovered = {"voltage_v": 12.2, "temperature_c": 35.0, "loaded": True}
        with redirect_stderr(output):
            _check_health(
                low,
                min_voltage_v=9.6,
                hard_min_voltage_v=5.0,
                max_temperature_c=55.0,
                voltage_warning_monitor=monitor,
            )
            _check_health(
                recovered,
                min_voltage_v=9.6,
                hard_min_voltage_v=5.0,
                max_temperature_c=55.0,
                voltage_warning_monitor=monitor,
            )

        self.assertIn("\033[33mWARNING", output.getvalue())
        self.assertIn("VOLTAGE RECOVERED", output.getvalue())
        self.assertEqual(monitor.summary()["event_count"], 1)
        self.assertAlmostEqual(
            monitor.summary()["events"][0]["minimum_voltage_v"], 8.574
        )
        with self.assertRaisesRegex(RuntimeError, "hard 5.000 V abort floor"):
            _check_health(
                {"voltage_v": 4.9, "temperature_c": 35.0, "loaded": True},
                min_voltage_v=9.6,
                hard_min_voltage_v=5.0,
                max_temperature_c=55.0,
                voltage_warning_monitor=monitor,
            )

    def test_packet_round_trip(self):
        packet = build_packet(100, 28, [0x34, 0x12])
        self.assertEqual(packet[-1], checksum(packet[:-1]))
        parsed = parse_packets(packet)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].servo_id, 100)
        self.assertEqual(parsed[0].command, 28)
        self.assertEqual(parsed[0].params, (0x34, 0x12))

    def test_servo_units_are_raw_centered(self):
        self.assertEqual(radians_to_units(0.0), 500)
        self.assertEqual(radians_to_units(math.radians(8.0)), 533)
        self.assertAlmostEqual(units_to_radians(500), 0.0)

    def test_stop_and_signed_motor_mode(self):
        class FakeTransport:
            writes = []

            def write(self, packet):
                self.writes.append(packet)

            def read_available(self, *, deadline_s, quiet_s):
                del deadline_s, quiet_s
                return build_packet(100, CMD_OR_MOTOR_MODE_READ, [0, 0, 0xFF, 0xFF])

            def reset_input_buffer(self):
                pass

            def close(self):
                pass

        transport = FakeTransport()
        bus = Htd45hBus(transport)
        self.assertEqual(bus.read_motor_mode(100), (0, -1))
        bus.move_stop(100)
        self.assertEqual(parse_packets(transport.writes[-1])[0].command, CMD_MOVE_STOP)

    def test_capture_records_position_and_health(self):
        class FakeBus:
            position = 500

            def move(self, _servo_id, position, _duration_ms):
                self.position = position

            def read_position(self, _servo_id):
                return self.position

            def read_voltage_v(self, _servo_id):
                return 11.8

            def read_temperature_c(self, _servo_id):
                return 31

            def read_loaded(self, _servo_id):
                return True

        targets = np.linspace(0.0, 0.02, 5)
        result = capture_profile(
            FakeBus(),
            100,
            (ProfileSegment("test", "chirp", targets),),
            sample_hz=1000.0,
            move_time_ms=1,
            write_deadband_units=0,
            health_poll_hz=1000.0,
            min_voltage_v=9.6,
            max_temperature_c=55.0,
            max_position_error_deg=5.0,
            max_position_error_duration_s=0.01,
        ).arrays()
        self.assertEqual(result["command_rad"].shape, (5,))
        self.assertTrue(np.all(result["loaded"] == 1.0))
        self.assertTrue(np.all(result["voltage_v"] == 11.8))

    def test_hysteresis_profile_has_matched_continuous_sweeps(self):
        profile = build_hysteresis_profile(
            center_deg=10.0,
            amplitude_deg=2.0,
            sample_hz=10.0,
            settle_s=0.2,
            sweep_rate_deg_s=1.0,
            sweep_cycles=2,
        )
        segments = {item.name: item for item in profile}
        self.assertIn("sweep_positive_cycle01", segments)
        self.assertIn("sweep_negative_cycle02", segments)
        positive = segments["sweep_positive_cycle01"].targets_rad
        negative = segments["sweep_negative_cycle01"].targets_rad
        self.assertAlmostEqual(math.degrees(positive[0]), 8.0)
        self.assertAlmostEqual(math.degrees(positive[-1]), 12.0)
        np.testing.assert_allclose(positive, negative[::-1])

    def test_hysteresis_summary_matches_command_units(self):
        command_deg = np.arange(8.0, 12.01, 0.24)
        command = np.radians(np.concatenate((command_deg, command_deg[::-1])))
        measured = np.radians(
            np.concatenate((command_deg - 0.24, command_deg[::-1] + 0.24))
        )
        segment = np.asarray(
            ["sweep_positive_cycle01"] * command_deg.size
            + ["sweep_negative_cycle01"] * command_deg.size
        )
        summary = summarize_hysteresis(
            {
                "segment_name": segment,
                "command_rad": command,
                "measured_position_rad": measured,
            },
            center_deg=10.0,
        )
        self.assertEqual(summary["status"], "measured")
        self.assertEqual(summary["aggregate"]["completed_cycles"], 1)
        self.assertAlmostEqual(
            summary["cycles"][0]["center_absolute_output_loop_deg"], 0.48
        )

    def test_preparation_samples_populate_canonical_trace(self):
        samples = [
            {
                "monotonic_time_s": 100.0,
                "wall_time_s": 1000.0,
                "segment_name": "prepare_neutral",
                "target_rad": 0.0,
                "position_rad": 0.0,
                "command_written": True,
                "command_write_duration_s": 0.001,
                "position_read_duration_s": 0.002,
                "command_age_at_read_s": 0.003,
                "voltage_v": 12.0,
                "temperature_c": 30.0,
                "loaded": True,
            },
            {
                "monotonic_time_s": 100.1,
                "wall_time_s": 1000.1,
                "segment_name": "test_hold",
                "target_rad": 0.1,
                "position_rad": 0.08,
                "command_written": True,
                "command_write_duration_s": 0.001,
                "position_read_duration_s": 0.002,
                "command_age_at_read_s": 0.003,
                "voltage_v": 11.9,
                "temperature_c": 31.0,
                "loaded": True,
            },
        ]

        arrays = _monitored_samples_to_buffer(samples).arrays()

        np.testing.assert_allclose(arrays["timestamps_s"], [0.0, 0.1])
        np.testing.assert_allclose(arrays["command_rad"], [0.0, 0.1])
        np.testing.assert_allclose(arrays["measured_position_rad"], [0.0, 0.08])
        np.testing.assert_allclose(arrays["voltage_v"], [12.0, 11.9])
        self.assertEqual(
            arrays["segment_name"].tolist(), ["prepare_neutral", "test_hold"]
        )


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is not installed")
class FixtureTest(unittest.TestCase):
    def test_fixture_and_dynamic_preflight_cover_deployment_envelope(self):
        from wr2.tools.servo_sysid.core import FixtureModel

        fixture = FixtureModel.load()
        torque, inertia = fixture.evaluate_static(np.deg2rad([0, 60, 72, -60, -72]))
        self.assertAlmostEqual(fixture.moving_mass_kg, 2.722761421)
        np.testing.assert_allclose(
            torque,
            [-0.017149, 2.757323, 3.032168, -2.774471, -3.042767],
            atol=2e-5,
        )
        np.testing.assert_allclose(inertia, 0.043018486, atol=1e-8)

        positive = fixture.dynamic_envelope(
            build_profile(center_deg=60, amplitudes_deg=[2, 5, 8]),
            sample_hz=50,
        )
        negative = fixture.dynamic_envelope(
            build_profile(center_deg=-60, amplitudes_deg=[2, 5, 8]),
            sample_hz=50,
        )
        self.assertGreater(positive["peak_total_torque_nm"], 3.0)
        self.assertGreater(negative["peak_total_torque_nm"], 3.0)
        self.assertLess(positive["peak_total_torque_nm"], 3.2)
        self.assertLess(negative["peak_total_torque_nm"], 3.2)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is not installed")
    def test_dynamics_fitter_replays_known_parameters(self):
        from wr2.tools.servo_sysid.core import DEFAULT_FIXTURE, file_sha256
        from wr2.tools.servo_sysid.fit import (
            Replay,
            ServoDynamics,
            Trace,
            fit,
            training_parameters,
        )

        time_s = np.arange(80, dtype=np.float64) * 0.02
        replays = []
        expected = ServoDynamics()
        for center_deg in (0.0, 60.0, -60.0):
            center = math.radians(center_deg)
            command = center + math.radians(2.0) * np.sin(2.0 * np.pi * 0.5 * time_s)
            seed = Trace(
                DEFAULT_FIXTURE,
                f"synthetic-{center_deg:g}",
                command,
                np.full_like(command, center),
                time_s,
                1,
                0.0,
                file_sha256(DEFAULT_FIXTURE),
            )
            measured = Replay(
                DEFAULT_FIXTURE, seed, sim_dt=0.002, force_limit_nm=4.0
            ).simulate(expected)
            trace = Trace(
                seed.path,
                seed.condition_id,
                command,
                measured,
                time_s,
                1,
                0.0,
                seed.fixture_sha256,
            )
            replays.append(
                Replay(DEFAULT_FIXTURE, trace, sim_dt=0.002, force_limit_nm=4.0)
            )

        fitted, report = fit(replays, delays=(0,), max_nfev=2)
        self.assertAlmostEqual(fitted.kp, expected.kp, places=4)
        self.assertLess(report["candidates"][0]["metrics"]["mean_rmse_deg"], 1e-6)

        mapped = training_parameters(fitted, controller_kv=0.5)
        self.assertAlmostEqual(
            mapped.damping + mapped.kv_sim,
            fitted.effective_velocity_damping,
        )
        self.assertAlmostEqual(mapped.kp_sim, fitted.kp)

    def test_dynamics_training_mapping_rejects_double_counted_damping(self):
        from wr2.tools.servo_sysid.fit import ServoDynamics, training_parameters

        with self.assertRaisesRegex(ValueError, "smaller than controller_kv"):
            training_parameters(
                ServoDynamics(effective_velocity_damping=0.4),
                controller_kv=0.5,
            )


class CampaignAnalysisTest(unittest.TestCase):
    def test_bam_plan_contains_only_low_load_conditions(self):
        plan = load_plan("bam_position")
        self.assertEqual(
            [condition.condition_id for condition in plan.conditions],
            [
                "commission_plus10",
                "commission_minus10",
                "E3_inertial_bandwidth",
            ],
        )

    def test_bam_capture_commands_enforce_low_torque_caps(self):
        args = SimpleNamespace(
            servo_id=100,
            servo_label="unit-a",
            board_port="unused",
            baudrate=115200,
            fixture_mjcf=Path("fixture.xml"),
            fixture_direction=1,
            fixture_qpos_offset_deg=0.0,
            fixture_label="bam",
            measured_weight_kg=2.650,
            measured_com_radius_m=0.1204,
            repeats=5,
            external_log_label=None,
            external_log_label_prefix="supply-run",
            cooldown_target_c=35.0,
            min_voltage_v=9.6,
            hard_min_voltage_v=5.0,
            max_temperature_c=80.0,
            max_position_error_deg=5.0,
            max_position_error_duration_s=0.15,
            max_repeatability_std_deg=0.5,
            max_static_torque_nm=3.2,
            max_predicted_torque_nm=3.2,
        )
        plan = load_plan("bam_position")
        plus = capture_command(args, plan.conditions[0], Path("plus.npz"), execute=True)
        e3 = capture_command(args, plan.conditions[2], Path("e3.npz"), execute=True)
        self.assertIn("wr2.tools.servo_sysid.capture", plus)
        self.assertEqual(plus[plus.index("--max-static-torque-nm") + 1], "0.7")
        self.assertEqual(e3[e3.index("--max-static-torque-nm") + 1], "0.2")
        self.assertEqual(e3[e3.index("--max-predicted-torque-nm") + 1], "0.6")
        self.assertEqual(plus[plus.index("--max-temperature-c") + 1], "55.0")
        self.assertEqual(plus[plus.index("--min-voltage-v") + 1], "9.6")
        self.assertEqual(plus[plus.index("--hard-min-voltage-v") + 1], "5.0")
        self.assertIn("--execute", plus)
        self.assertIn("--confirm-fixture-safe", e3)

    def test_hysteresis_plan_uses_bounded_bidirectional_profile(self):
        args = SimpleNamespace(
            servo_id=100,
            servo_label="unit-a",
            board_port="unused",
            baudrate=115200,
            fixture_mjcf=Path("fixture.xml"),
            fixture_direction=1,
            fixture_qpos_offset_deg=0.0,
            fixture_label="bam",
            measured_weight_kg=2.650,
            measured_com_radius_m=0.1204,
            external_log_label=None,
            external_log_label_prefix=None,
            cooldown_target_c=35.0,
            min_voltage_v=9.6,
            hard_min_voltage_v=5.0,
            max_temperature_c=80.0,
            max_position_error_deg=5.0,
            max_position_error_duration_s=0.15,
            max_static_torque_nm=3.2,
            max_predicted_torque_nm=3.2,
        )
        condition = load_plan("bam_hysteresis").conditions[0]
        command = capture_command(
            args, condition, Path("hysteresis.npz"), execute=True
        )
        self.assertEqual(command[command.index("--profile") + 1], "hysteresis")
        self.assertEqual(command[command.index("--sweep-cycles") + 1], "3")
        self.assertEqual(command[command.index("--move-time-ms") + 1], "250")
        self.assertEqual(command[command.index("--max-static-torque-nm") + 1], "0.75")
        negative = capture_command(
            args,
            load_plan("bam_hysteresis").conditions[1],
            Path("negative.npz"),
            execute=True,
        )
        self.assertEqual(negative[negative.index("--cooldown-target-c") + 1], "30.0")

    def test_bam_hardware_mode_requires_bounded_selection(self):
        with self.assertRaisesRegex(SystemExit, "requires --stop-after"):
            campaign_main(
                [
                    "--plan",
                    "bam_position",
                    "--servo-id",
                    "100",
                    "--servo-label",
                    "unit-a",
                    "--board-port",
                    "unused",
                    "--measured-weight-kg",
                    "2.650",
                    "--measured-com-radius-m",
                    "0.1204",
                    "--execute",
                    "--confirm-fixture-safe",
                ]
            )

    def test_bam_preflight_runs_all_plan_conditions_without_hardware(self):
        with patch("wr2.tools.servo_sysid.campaign._run", return_value=0) as run:
            result = campaign_main(
                [
                    "--plan",
                    "bam_position",
                    "--servo-id",
                    "100",
                    "--servo-label",
                    "unit-a",
                    "--board-port",
                    "unused",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assertNotIn("--execute", call.args[0])

    def test_bam_run_all_writes_completed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / "bam-suite"

            def fake_run(command):
                if "--execute" not in command:
                    return 0
                output = Path(command[command.index("--output") + 1])
                center_deg = float(command[command.index("--center-deg") + 1])
                if "--prepare-only" in command:
                    position_rad = math.radians(center_deg)
                    preparation = [
                        {
                            "elapsed_s": 0.0,
                            "position_rad": 0.0,
                            "voltage_v": 11.8,
                            "temperature_c": 29.0,
                        },
                        {
                            "elapsed_s": 1.0,
                            "position_rad": 0.0,
                            "voltage_v": 11.8,
                            "temperature_c": 29.0,
                        },
                        {
                            "elapsed_s": 0.0,
                            "position_rad": position_rad,
                            "voltage_v": 11.7,
                            "temperature_c": 30.0,
                        },
                        {
                            "elapsed_s": 1.0,
                            "position_rad": position_rad,
                            "voltage_v": 11.7,
                            "temperature_c": 31.0,
                        },
                    ]
                else:
                    preparation = []
                output.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "outcome": "completed",
                            "center_deg": center_deg,
                            "preparation": preparation,
                            "trace_summary": {},
                            "health_summary": {},
                            "timing_summary": {},
                            "predicted_envelope": {},
                        }
                    )
                )
                return 0

            with (
                patch(
                    "wr2.tools.servo_sysid.campaign._run", side_effect=fake_run
                ) as run,
                patch(
                    "wr2.tools.servo_sysid.campaign._git_state",
                    return_value={"revision": "abc123", "worktree_clean": True},
                ),
            ):
                result = campaign_main(
                    [
                        "--plan",
                        "bam_position",
                        "--servo-id",
                        "100",
                        "--servo-label",
                        "unit-a",
                        "--board-port",
                        "unused",
                        "--campaign-dir",
                        str(suite),
                        "--measured-weight-kg",
                        "2.650",
                        "--measured-com-radius-m",
                        "0.1204",
                        "--external-log-label-prefix",
                        "supply-run",
                        "--run-all",
                        "--execute",
                        "--confirm-fixture-safe",
                    ]
                )

            manifest = json.loads((suite / "campaign_manifest.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 14)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["conditions"]), 3)
        self.assertTrue(
            all(item["outcome"] == "completed" for item in manifest["conditions"])
        )

    def test_voltage_warning_does_not_stop_repeatability_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / "bam-suite"

            def fake_run(command):
                if "--execute" not in command:
                    return 0
                output = Path(command[command.index("--output") + 1])
                center_deg = float(command[command.index("--center-deg") + 1])
                output.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "outcome": "completed",
                            "center_deg": center_deg,
                            "preparation": [
                                {
                                    "elapsed_s": 0.0,
                                    "position_rad": 0.0,
                                    "voltage_v": 8.574,
                                    "temperature_c": 29.0,
                                },
                                {
                                    "elapsed_s": 1.0,
                                    "position_rad": 0.0,
                                    "voltage_v": 11.8,
                                    "temperature_c": 29.0,
                                },
                                {
                                    "elapsed_s": 0.0,
                                    "position_rad": math.radians(center_deg),
                                    "voltage_v": 11.7,
                                    "temperature_c": 30.0,
                                },
                                {
                                    "elapsed_s": 1.0,
                                    "position_rad": math.radians(center_deg),
                                    "voltage_v": 11.7,
                                    "temperature_c": 31.0,
                                },
                            ],
                            "voltage_warnings": {"event_count": 1},
                        }
                    )
                )
                return 0

            with (
                patch("wr2.tools.servo_sysid.campaign._run", side_effect=fake_run),
                patch(
                    "wr2.tools.servo_sysid.campaign._git_state",
                    return_value={"revision": "abc123", "worktree_clean": True},
                ),
            ):
                result = campaign_main(
                    [
                        "--plan",
                        "bam_position",
                        "--servo-id",
                        "100",
                        "--servo-label",
                        "unit-a",
                        "--board-port",
                        "unused",
                        "--campaign-dir",
                        str(suite),
                        "--measured-weight-kg",
                        "2.650",
                        "--measured-com-radius-m",
                        "0.1204",
                        "--start-at",
                        "commission_plus10",
                        "--stop-after",
                        "commission_plus10",
                        "--execute",
                        "--confirm-fixture-safe",
                    ]
                )

            manifest = json.loads((suite / "campaign_manifest.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(manifest["status"], "partial_with_warnings")
        self.assertEqual(
            manifest["conditions"][0]["outcome"], "completed_with_warnings"
        )
        self.assertEqual(
            manifest["conditions"][0]["summary"]["failures"],
            ["voltage is below limit"],
        )

    def test_campaign_plans_are_versioned_and_loadable(self):
        self.assertEqual(
            available_plans(),
            (
                "bam_hysteresis",
                "bam_position",
                "bam_repeatability",
                "legacy_deployment",
            ),
        )
        args = SimpleNamespace(
            plan="bam_repeatability",
            center_deg=-12.5,
            repeats=3,
            prepare_speed_deg_s=None,
            settle_s=None,
        )
        plan = configured_plan(args)
        self.assertEqual(plan.conditions[0].condition_id, "repeatability_minus12p5")
        self.assertEqual(plan.conditions[0].repeats, 3)

    def test_commission_repeatability_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            captures = []
            for index, loaded_deg in enumerate((9.5, 9.7, 9.6, 9.4, 9.8), start=1):
                path = Path(temp) / f"repeat_{index:02d}.json"
                path.write_text(
                    json.dumps(
                        {
                            "outcome": "completed",
                            "center_deg": 10.0,
                            "preparation": [
                                {
                                    "elapsed_s": 0.0,
                                    "position_rad": 0.0,
                                    "voltage_v": 11.8,
                                    "temperature_c": 29.0,
                                },
                                {
                                    "elapsed_s": 1.0,
                                    "position_rad": 0.0,
                                    "voltage_v": 11.8,
                                    "temperature_c": 29.0,
                                },
                                {
                                    "elapsed_s": 0.0,
                                    "position_rad": math.radians(loaded_deg - 1.0),
                                    "voltage_v": 11.6,
                                    "temperature_c": 30.0,
                                },
                                {
                                    "elapsed_s": 1.0,
                                    "position_rad": math.radians(loaded_deg),
                                    "voltage_v": 11.5,
                                    "temperature_c": 31.0,
                                },
                            ],
                        }
                    )
                )
                captures.append(path)
            report = summarize_series(
                captures,
                center_deg=10.0,
                expected_repeats=5,
                max_repeatability_std_deg=0.5,
                max_position_error_deg=5.0,
                min_voltage_v=9.6,
                max_temperature_c=55.0,
            )
        self.assertEqual(report["status"], "stable")
        self.assertEqual(report["aggregate"]["completed_repeats"], 5)
        self.assertLess(report["aggregate"]["loaded_position_std_deg"], 0.5)

    def test_hardware_campaign_requires_a_bounded_stage(self):
        with self.assertRaisesRegex(SystemExit, "requires --stop-after"):
            campaign_main(
                [
                    "--servo-id",
                    "100",
                    "--board-port",
                    "unused",
                    "--execute",
                    "--confirm-fixture-safe",
                ]
            )

    def test_staged_condition_selection(self):
        selected = selected_conditions(
            SimpleNamespace(start_at="E4_loaded_plus60", stop_after="E4_loaded_plus60")
        )
        self.assertEqual([item.condition_id for item in selected], ["E4_loaded_plus60"])
        self.assertEqual(selected[0].amplitudes_deg, "2,5")

    def test_staged_campaign_keeps_a_reusable_partial_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            campaign_dir = Path(temp) / "unit-a"
            with (
                patch("wr2.tools.servo_sysid.campaign._run", return_value=0),
                patch(
                    "wr2.tools.servo_sysid.campaign._git_state",
                    return_value={"revision": "abc123", "worktree_clean": True},
                ),
            ):
                result = campaign_main(
                    [
                        "--servo-id",
                        "100",
                        "--servo-label",
                        "unit-a",
                        "--board-port",
                        "unused",
                        "--campaign-dir",
                        str(campaign_dir),
                        "--measured-weight-kg",
                        "2.650",
                        "--measured-com-radius-m",
                        "0.1204",
                        "--start-at",
                        "E1_static_plus72",
                        "--stop-after",
                        "E1_static_plus72",
                        "--execute",
                        "--confirm-fixture-safe",
                    ]
                )
            manifest = json.loads((campaign_dir / "campaign_manifest.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["conditions"][0]["condition_id"], "E1_static_plus72")

    def _campaign(self, parent: Path, label: str) -> Path:
        directory = parent / label
        directory.mkdir()
        (directory / "campaign_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "servo_id": label,
                    "servo_label": label,
                    "fixture_sha256": "synthetic-fixture",
                }
            )
        )
        (directory / "external_measurements.json").write_text(
            json.dumps(
                {
                    "clock_aligned": True,
                    "minimum_voltage_v": 10.4,
                    "peak_current_a": 7.0,
                    "rms_current_a": 3.0,
                }
            )
        )
        for index, condition in enumerate(CONDITIONS, start=1):
            torque = {
                "E1_static_plus72": 3.032,
                "E2_static_minus72": -3.043,
                "E7_thermal_hold_plus60": 2.757,
            }.get(condition.condition_id, 2.75)
            capture = {
                "condition_id": condition.condition_id,
                "outcome": "completed",
                "servo_label": label,
                "fixture_sha256": "synthetic-fixture",
                "center_static_torque_nm": torque,
                "predicted_envelope": {"peak_total_torque_nm": 3.1},
                "trace_summary": {
                    "tracking_rmse_deg": 1.0,
                    "tracking_abs_p95_deg": 2.0,
                    "observed_peak_speed_rad_s": 1.0,
                },
                "health_summary": {
                    "min_voltage_v": 10.5,
                    "max_temperature_c": 48.0,
                    "temperature_slope_c_per_min_last_60s": 0.2,
                },
                "preparation": [],
                "cooldown": [],
            }
            (directory / f"{index:02d}_{condition.condition_id}.json").write_text(
                json.dumps(capture)
            )
        return directory

    def test_three_complete_servos_produce_conservative_spec(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [self._campaign(root, f"servo-{index}") for index in range(3)]
            report = analyze_campaigns(
                paths,
                min_servos=3,
                min_voltage_v=9.6,
                max_temperature_c=55.0,
                max_tracking_p95_deg=5.0,
                max_thermal_slope_c_per_min=0.5,
                safety_factor=1.2,
                nominal_force_cap_nm=4.0,
                require_external_measurements=True,
            )
        self.assertEqual(report["qualification_status"], "passed")
        self.assertAlmostEqual(
            report["deployment_limits"]["conservative_peak_torque_nm"],
            3.032 / 1.2,
        )
        self.assertAlmostEqual(
            report["deployment_limits"]["conservative_continuous_torque_nm"],
            2.757 / 1.2,
        )

    def test_one_servo_does_not_claim_population_qualification(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self._campaign(Path(temp), "servo-a")
            report = analyze_campaigns(
                [path],
                min_servos=3,
                min_voltage_v=9.6,
                max_temperature_c=55.0,
                max_tracking_p95_deg=5.0,
                max_thermal_slope_c_per_min=0.5,
                safety_factor=1.2,
                nominal_force_cap_nm=4.0,
                require_external_measurements=True,
            )
        self.assertEqual(report["qualification_status"], "incomplete_or_failed")
        self.assertTrue(any("distinct servo" in item for item in report["failures"]))


if __name__ == "__main__":
    unittest.main()
