"""Offline checks for the deployment scripts' safety boundaries on Linux."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class ScriptSafetyTests(unittest.TestCase):
    def test_stop_refuses_to_kill_unverified_processes(self):
        for name in ("minicpm-ocr.sh", "jev_server.sh", "webui.sh"):
            with self.subTest(script=name):
                sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                try:
                    result = subprocess.run(["bash", str(SCRIPTS / name), "stop"], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("cannot verify service ownership", result.stderr)
                    self.assertIsNone(sentinel.poll())
                finally:
                    sentinel.terminate()
                    sentinel.wait()

    def test_download_preserves_existing_files_and_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mock = root / "bin"
            mock.mkdir()
            curl = mock / "curl"
            curl.write_text("#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do if [ \"$1\" = -o ]; then shift; printf fake > \"$1\"; exit 0; fi; shift; done\nexit 2\n")
            stat = mock / "stat"
            stat.write_text("#!/bin/sh\nprintf '2600969816\\n'\n")
            curl.chmod(0o755)
            stat.chmod(0o755)
            env = dict(os.environ, PATH=f"{mock}:{os.environ['PATH']}")
            env["MINICPM_OCR_HOME"] = str(root / "existing")
            target = root / "existing/models/MiniCPM-OCR-1.0/config.json"
            target.parent.mkdir(parents=True)
            target.write_text("do not overwrite")
            command = ["bash", str(SCRIPTS / "01_download.sh")]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing", result.stderr)
            self.assertEqual(target.read_text(), "do not overwrite")

            env["MINICPM_OCR_HOME"] = str(root / "mismatch")
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA256 MISMATCH", result.stderr)


if __name__ == "__main__":
    unittest.main()
