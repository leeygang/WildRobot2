import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "scp_from_remote.sh"


class RemoteResultCopyTest(unittest.TestCase):
    def _copy_custom_folder(self, arguments, *, missing=False, fallback=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, remote, fake_bin = root / "repo", root / "remote", root / "bin"
            (repo / "scripts").mkdir(parents=True)
            script = repo / "scripts" / SCRIPT.name
            shutil.copy2(SCRIPT, script)
            fake_bin.mkdir()
            group = remote / "results" / "wr2_solver_ab"
            if not missing:
                for arm in ("newton1_seed0", "newton10_seed0"):
                    run = group / arm / "run_a"
                    (run / "checkpoints").mkdir(parents=True)
                    (run / "checkpoints" / "model.pt").write_bytes(b"checkpoint")
                    (run / "training_config.yaml").write_text("solver_iterations: 10\n")
            local = repo / "results" / "wr2_solver_ab"
            local.mkdir(parents=True)
            (local / "notes.local").write_text("keep me")

            (fake_bin / "ssh").write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                import os, re, sys
                from pathlib import Path
                match = re.fullmatch(r"\[ -d '(/remote/[^']*)' \]", sys.argv[-1])
                if not match:
                    raise SystemExit(1)
                path = Path(os.environ["TEST_REMOTE_ROOT"]) / match[1][len("/remote/"):]
                raise SystemExit(0 if path.is_dir() else 1)
                """))
            transfer = textwrap.dedent("""\
                #!/usr/bin/env python3
                import json, os, shutil, sys
                from pathlib import Path
                source = sys.argv[-2].split(":", 1)[1]
                path = Path(os.environ["TEST_REMOTE_ROOT"]) / source[len("/remote/"):]
                shutil.copytree(path, sys.argv[-1], dirs_exist_ok=True)
                Path(os.environ["TEST_TRANSFER_LOG"]).write_text(json.dumps({
                    "tool": Path(sys.argv[0]).name, "args": sys.argv[1:]
                }))
                """)
            for name in ("rsync", "scp"):
                (fake_bin / name).write_text(transfer)
            for executable in fake_bin.iterdir():
                executable.chmod(0o755)
            environment = dict(os.environ)
            environment.update({
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "TEST_REMOTE_ROOT": str(remote),
                "TEST_TRANSFER_LOG": str(root / "transfer.json"),
                "WR2_SSH_CONTROL_DIR": str(root / "ssh-control"),
            })
            if fallback:
                bash_env = root / "bash-env"
                bash_env.write_text(textwrap.dedent("""\
                    command() {
                        if [ "$1" = "-v" ] && [ "$2" = "rsync" ]; then return 1; fi
                        builtin command "$@"
                    }
                    """))
                environment["BASH_ENV"] = str(bash_env)
            result = subprocess.run(
                [str(script), "--host", "test-host", "--remote-base", "/remote", *arguments],
                capture_output=True, text=True, env=environment,
            )
            copied = {
                str(path.relative_to(repo)): path.read_bytes()
                for path in local.rglob("*") if path.is_file()
            }
            log = root / "transfer.json"
            return result, copied, json.loads(log.read_text()) if log.exists() else None

    def test_custom_results_folder_copies_all_nested_runs_and_preserves_local_files(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                result, copied, transfer = self._copy_custom_folder(
                    ["--results", "wr2_solver_ab/"], fallback=fallback
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(transfer["tool"], "scp" if fallback else "rsync")
                self.assertEqual(copied["results/wr2_solver_ab/notes.local"], b"keep me")
                for arm in ("newton1_seed0", "newton10_seed0"):
                    self.assertEqual(
                        copied[f"results/wr2_solver_ab/{arm}/run_a/checkpoints/model.pt"],
                        b"checkpoint",
                    )
                    self.assertIn(f"results/wr2_solver_ab/{arm}/run_a/training_config.yaml", copied)

    def test_custom_results_folder_can_copy_only_a_nested_subtree(self):
        result, copied, _ = self._copy_custom_folder(
            ["--results", "wr2_solver_ab/newton10_seed0"]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("results/wr2_solver_ab/newton10_seed0/run_a/checkpoints/model.pt", copied)
        self.assertFalse(any("newton1_seed0" in name for name in copied))

    def test_custom_results_folder_dry_run_does_not_copy(self):
        result, copied, transfer = self._copy_custom_folder(
            ["--results", "wr2_solver_ab", "--dry-run"]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Directory:", result.stdout)
        self.assertIsNone(transfer)
        self.assertEqual(copied, {"results/wr2_solver_ab/notes.local": b"keep me"})

    def test_custom_results_folder_rejects_missing_or_unsafe_paths(self):
        cases = (
            (["--results"], False, "usage:"),
            (["--results", "wr2_solver_ab"], True, "missing remote result folder"),
            (["--results", "../outside"], False, "invalid repository-relative path"),
            (["--results", "wr2_solver_ab/../../outside"], False, "invalid repository-relative path"),
            (["--results", "/remote/results/wr2_solver_ab"], False, "invalid repository-relative path"),
            (["--results", ""], False, "invalid repository-relative path"),
            (["--results", "wr2_solver_ab/newton10_seed0/run_a/training_config.yaml"], False, "missing remote result folder"),
        )
        for arguments, missing, message in cases:
            with self.subTest(arguments=arguments):
                result, _, transfer = self._copy_custom_folder(arguments, missing=missing)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertIsNone(transfer)

    def test_script_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_no_arguments_prints_usage_without_nounset_error(self):
        result = subprocess.run(
            [str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stdout)
        self.assertNotIn("unbound variable", result.stderr)

    def test_connection_option_without_action_prints_usage(self):
        result = subprocess.run(
            [str(SCRIPT), "--wrdev"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stdout)
        self.assertNotIn("unbound variable", result.stderr)

    def test_latest_sysid_treats_npz_and_json_as_one_result(self):
        with tempfile.TemporaryDirectory() as temp:
            fake_bin = Path(temp)
            fake_ssh = fake_bin / "ssh"
            fake_ssh.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import sys

                    command = sys.argv[-1]
                    if "find '/remote/results/servo_sysid'" in command:
                        print("400.0\\td\\tcampaign-new")
                        print("300.0\\tf\\tcommission_plus10.json")
                        print("299.0\\tf\\tcommission_plus10.npz")
                        print("200.0\\td\\tcampaign-old")
                        raise SystemExit(0)
                    if command == "[ -d '/remote/results/servo_sysid/campaign-new' ]":
                        raise SystemExit(0)
                    if command.startswith("[ -d '"):
                        raise SystemExit(1)
                    if command in {
                        "[ -f '/remote/results/servo_sysid/commission_plus10.json' ]",
                        "[ -f '/remote/results/servo_sysid/commission_plus10.npz' ]",
                    }:
                        raise SystemExit(0)
                    raise SystemExit(1)
                    """
                )
            )
            fake_ssh.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--host",
                    "test-host",
                    "--remote-base",
                    "/remote",
                    "--dry-run",
                    "--latest",
                    "servo_sysid",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("campaign-new", result.stdout)
        self.assertIn("commission_plus10.json", result.stdout)
        self.assertIn("commission_plus10.npz", result.stdout)
        self.assertNotIn("campaign-old", result.stdout)

    def test_rejects_parent_directory_path(self):
        result = subprocess.run(
            [str(SCRIPT), "../outside"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid repository-relative path", result.stderr)

    def test_linux_pc_profile_uses_public_environment(self):
        environment = dict(os.environ)
        environment["LINUX_PUBLIC_IP"] = "203.0.113.42"
        environment["LINUX_PUBLIC_PORT"] = "2222"
        result = subprocess.run(
            [str(SCRIPT), "--linux-pc", "../outside"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertIn("Remote: leeygang@203.0.113.42", result.stdout)
        self.assertIn("/home/leeygang/projects/WildRobot2", result.stdout)

    def test_wrdev_profile_selects_deployment_host(self):
        result = subprocess.run(
            [str(SCRIPT), "--wrdev", "../outside"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertIn("Remote: leeygang@wrdev.local", result.stdout)
        self.assertIn("/home/leeygang/projects/WildRobot2", result.stdout)


if __name__ == "__main__":
    unittest.main()
