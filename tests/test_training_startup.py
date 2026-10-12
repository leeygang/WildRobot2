"""Process-start defaults must precede GPU backend imports."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingStartupTest(unittest.TestCase):
    def inspect_fresh_import(self, overrides=None):
        environment = dict(os.environ)
        environment.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        environment.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        environment.update(overrides or {})
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, os, sys; import wr2.locomotion.train; "
                "print(json.dumps({'preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'), "
                "'fraction': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'), "
                "'backend_imports': [name for name in ('jax', 'torch', 'brax') if name in sys.modules]}))",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    def test_default_disables_preallocation_before_gpu_backend_imports(self):
        result = self.inspect_fresh_import()
        self.assertEqual(result["preallocate"], "false")
        self.assertIsNone(result["fraction"])
        self.assertEqual(result["backend_imports"], [])

    def test_explicit_user_allocator_overrides_are_preserved(self):
        for preallocate in ("true", "false"):
            with self.subTest(preallocate=preallocate):
                result = self.inspect_fresh_import({
                    "XLA_PYTHON_CLIENT_PREALLOCATE": preallocate,
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.4",
                })
                self.assertEqual(result["preallocate"], preallocate)
                self.assertEqual(result["fraction"], "0.4")
                self.assertEqual(result["backend_imports"], [])


if __name__ == "__main__":
    unittest.main()
