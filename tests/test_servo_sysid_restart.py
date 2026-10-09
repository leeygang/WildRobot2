from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wr2.tools.servo_sysid.campaign import (
    _campaign_lock,
    _new_manifest,
    _parser,
    configured_plan,
    main,
)
from wr2.tools.servo_sysid.core import file_sha256


class CampaignRestartTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / "unit-c"
        self.arguments = [
            "--plan",
            "bam_low_load_qualification",
            "--servo-id",
            "100",
            "--servo-label",
            "unit-c",
            "--board-port",
            "unused",
            "--run-dir",
            str(self.directory),
            "--measured-weight-kg",
            "2.650",
            "--measured-com-radius-m",
            "0.1204",
            "--stop-after",
            "Q1_repeatability_plus10",
        ]
        self.git = {"revision": "abc123", "worktree_clean": True}
        args = _parser().parse_args(self.arguments)
        self.manifest = _new_manifest(
            args, configured_plan(args), args.fixture_mjcf.resolve(), dict(self.git)
        )
        self.directory.mkdir()
        self.metadata = self.directory / "campaign_manifest.json"
        self.metadata.write_text(json.dumps(self.manifest))
        self.condition = self.directory / "01_Q1_repeatability_plus10"
        self.condition.mkdir()
        (self.condition / "repeat_01.json").write_text(
            json.dumps({"outcome": "aborted", "error": "KeyboardInterrupt:"})
        )
        (self.condition / "repeat_01.npz").write_bytes(b"original raw capture")

    def run_campaign(self, extra=(), *, execute=True, git=None):
        def completed(_args, condition, _output):
            return 0, {"condition_id": condition.condition_id, "outcome": "completed"}

        flags = ["--execute", "--confirm-fixture-safe"] if execute else []
        with (
            redirect_stdout(io.StringIO()),
            patch("wr2.tools.servo_sysid.campaign._run", return_value=0),
            patch(
                "wr2.tools.servo_sysid.campaign._git_state",
                return_value=git or self.git,
            ),
            patch(
                "wr2.tools.servo_sysid.campaign._run_repeatability_condition",
                side_effect=completed,
            ) as capture,
        ):
            result = main([*self.arguments, *flags, *extra])
        return result, capture.call_count

    def test_default_keeps_aborted_output_and_rejects_overwrite(self):
        original_hash = file_sha256(self.metadata)
        with self.assertRaisesRegex(SystemExit, "incomplete condition output"):
            self.run_campaign()
        self.assertEqual(file_sha256(self.metadata), original_hash)
        self.assertTrue((self.condition / "repeat_01.npz").is_file())

    def test_default_resume_skips_completed_stage(self):
        self.manifest["status"] = "partial"
        self.manifest["conditions"] = [
            {"condition_id": "Q1_repeatability_plus10", "outcome": "completed"}
        ]
        self.metadata.write_text(json.dumps(self.manifest))
        result, captures = self.run_campaign()
        self.assertEqual((result, captures), (0, 0))
        self.assertTrue((self.condition / "repeat_01.npz").is_file())
        self.assertEqual(list(self.root.glob("unit-c.archived-*")), [])

    def test_resume_rejects_changed_cooldown_target(self):
        original_hash = file_sha256(self.metadata)
        with self.assertRaisesRegex(SystemExit, "different safety limits"):
            self.run_campaign(["--cooldown-target-c", "32"])
        self.assertEqual(file_sha256(self.metadata), original_hash)
        self.assertTrue((self.condition / "repeat_01.npz").is_file())

    def test_restart_records_changed_cooldown_target_without_changing_archive(self):
        result, captures = self.run_campaign(["--restart", "--cooldown-target-c", "32"])
        self.assertEqual((result, captures), (0, 1))
        manifest = json.loads(self.metadata.read_text())
        archive = Path(manifest["restarted_from"]["directory"])
        original = json.loads((archive / "campaign_manifest.json").read_text())
        self.assertEqual(original["safety_limits"]["cooldown_target_c"], 35.0)
        self.assertEqual(manifest["safety_limits"]["cooldown_target_c"], 32.0)

    def test_restart_preserves_artifacts_and_records_archive_provenance(self):
        self.manifest["git"]["revision"] = "previous-revision"
        self.metadata.write_text(json.dumps(self.manifest))
        original_metadata = self.metadata.read_bytes()
        original_hash = file_sha256(self.metadata)
        result, captures = self.run_campaign(["--restart"])
        self.assertEqual((result, captures), (0, 1))
        manifest = json.loads(self.metadata.read_text())
        archive = Path(manifest["restarted_from"]["directory"])
        self.assertEqual(archive.parent, self.directory.parent)
        self.assertEqual(
            (archive / "campaign_manifest.json").read_bytes(), original_metadata
        )
        self.assertEqual(manifest["restarted_from"]["manifest_sha256"], original_hash)
        self.assertEqual(
            (archive / "01_Q1_repeatability_plus10" / "repeat_01.npz").read_bytes(),
            b"original raw capture",
        )
        self.assertEqual(manifest["git"]["revision"], "abc123")
        self.assertEqual(manifest["status"], "partial")
        self.assertFalse(self.condition.exists())

    def test_preflight_restart_does_not_move_existing_artifacts(self):
        original_hash = file_sha256(self.metadata)
        result, captures = self.run_campaign(["--restart"], execute=False)
        self.assertEqual((result, captures), (0, 0))
        self.assertEqual(file_sha256(self.metadata), original_hash)
        self.assertEqual(list(self.root.glob("unit-c.archived-*")), [])

    def test_active_runner_blocks_restart(self):
        with _campaign_lock(self.directory):
            with self.assertRaisesRegex(SystemExit, "already running"):
                self.run_campaign(["--restart"])
        self.assertTrue((self.condition / "repeat_01.npz").is_file())
        self.assertEqual(list(self.root.glob("unit-c.archived-*")), [])

    def test_dirty_worktree_blocks_restart(self):
        with self.assertRaisesRegex(SystemExit, "clean Git worktree"):
            self.run_campaign(["--restart"], git={**self.git, "worktree_clean": False})
        self.assertTrue(self.metadata.is_file())
        self.assertEqual(list(self.root.glob("unit-c.archived-*")), [])

    def test_failed_preflight_blocks_restart(self):
        with (
            redirect_stdout(io.StringIO()),
            patch("wr2.tools.servo_sysid.campaign._run", return_value=1),
        ):
            result = main(
                [*self.arguments, "--restart", "--execute", "--confirm-fixture-safe"]
            )
        self.assertEqual(result, 1)
        self.assertTrue(self.metadata.is_file())
        self.assertEqual(list(self.root.glob("unit-c.archived-*")), [])

    def test_fixture_confirmation_is_required_before_restart(self):
        with self.assertRaisesRegex(SystemExit, "requires --confirm-fixture-safe"):
            main([*self.arguments, "--restart", "--execute"])
        self.assertTrue(self.metadata.is_file())
        self.assertEqual(list(self.root.glob("unit-c.archived-*")), [])

    def test_completed_campaign_is_preserved(self):
        for status in ("completed", "completed_with_warnings"):
            with self.subTest(status=status):
                self.manifest["status"] = status
                self.metadata.write_text(json.dumps(self.manifest))
                with self.assertRaisesRegex(SystemExit, "completed campaign"):
                    self.run_campaign(["--restart"])
                self.assertTrue(self.metadata.is_file())

    def test_unrelated_directory_is_preserved(self):
        self.manifest["servo_label"] = "unit-b"
        self.metadata.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(SystemExit, "different servo"):
            self.run_campaign(["--restart"])
        self.assertTrue(self.metadata.is_file())

    def test_directory_without_manifest_is_preserved(self):
        self.metadata.unlink()
        with self.assertRaisesRegex(SystemExit, "without a manifest"):
            self.run_campaign(["--restart"])
        self.assertTrue((self.condition / "repeat_01.npz").is_file())

    def test_restart_requires_explicit_directory(self):
        args = ["--servo-id", "100", "--board-port", "unused", "--restart"]
        with self.assertRaisesRegex(SystemExit, "requires --run-dir"):
            main(args)

    def test_restart_rejects_symlink_directory(self):
        link = self.root / "alias"
        link.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(SystemExit, "symlink"):
            self.run_campaign(["--restart", "--run-dir", str(link)])
        self.assertTrue(self.metadata.is_file())

    def test_restart_rejects_repository_directory(self):
        with (
            patch("wr2.tools.servo_sysid.campaign.REPO_ROOT", self.directory),
            self.assertRaisesRegex(SystemExit, "repository or its parent"),
        ):
            self.run_campaign(["--restart"])
        self.assertTrue(self.metadata.is_file())


if __name__ == "__main__":
    unittest.main()
