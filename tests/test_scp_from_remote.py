import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "scp_from_remote.sh"


class RemoteResultCopyTest(unittest.TestCase):
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
