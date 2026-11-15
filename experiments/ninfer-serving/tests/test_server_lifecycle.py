"""Exercise server ownership without loading a model or requiring a GPU."""

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ninfer_gb10_serve.sh"
FAKE_SERVER = """#!/usr/bin/env python3
import http.server, sys
if sys.argv[1].endswith("fail.ninfer"):
    sys.exit(1)
port = int(sys.argv[sys.argv.index("--port") + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(self.server.server_port).encode())
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def action(env, name, expected):
    result = subprocess.run(
        ["bash", str(SCRIPT), name], env=env, capture_output=True, text=True, timeout=75
    )
    assert (result.returncode == 0) == expected, (name, result.stdout, result.stderr)
    return result


def main():
    with tempfile.TemporaryDirectory(dir=os.environ.get("XL_OUT_DIR")) as temp:
        root = Path(temp)
        home = root / "home"
        home.mkdir()
        binary = root / "src" / "build-gb10-system-pkg" / "apps" / "ninfer-serve"
        binary.parent.mkdir(parents=True)
        binary.write_text(FAKE_SERVER)
        binary.chmod(0o755)
        artifact = root / "model.ninfer"
        artifact.touch()
        guard = home / "run_guarded.sh"
        guard.write_text("""#!/bin/bash
shift 2
exec python3 -c 'import subprocess,sys; p=subprocess.Popen(sys.argv[1:], start_new_session=True); sys.exit(p.wait())' "$@"
""")
        guard.chmod(0o755)
        env = os.environ.copy()
        env.pop("XL_NINFER_INSTANCE", None)
        env.update(
            HOME=str(home),
            NINFER_SRC=str(binary.parents[2]),
            NINFER_ARTIFACT=str(artifact),
            NINFER_RUN_DIR=str(root / "run"),
            NINFER_PORT=str(unused_port()),
        )
        other_port = unused_port()
        other = subprocess.Popen(
            [str(binary), str(artifact), "--port", str(other_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            action(env, "start", True)
            assert other.poll() is None, "starting interfered with an unrelated server"
            changed_port_env = dict(env, NINFER_PORT=str(other_port))
            status = action(changed_port_env, "status", True)
            assert status.stdout == env["NINFER_PORT"], "status queried an unrelated service"
            action(env, "stop", True)
            assert other.poll() is None, "stopping interfered with an unrelated server"

            # A stale/reused PID must not authorize signalling its new owner.
            run = root / "run" / "run"
            (run / "ninfer-serve.pid").write_text(str(other.pid))
            (run / "ninfer-serve-token").write_text("00000000-0000-0000-0000-000000000000")
            action(env, "stop", False)
            assert other.poll() is None, "stale PID caused an unrelated server to be stopped"
            (run / "ninfer-serve.pid").unlink()
            (run / "ninfer-serve-token").unlink()

            # A listener on the requested port must never be mistaken for ours.
            time.sleep(0.5)
            env["NINFER_PORT"] = str(other_port)
            action(env, "start", False)
            assert other.poll() is None

            failed_artifact = root / "fail.ninfer"
            failed_artifact.touch()
            env["NINFER_ARTIFACT"] = str(failed_artifact)
            env["NINFER_PORT"] = str(unused_port())
            action(env, "start", False)
            assert (
                run / "ninfer-serve.pid"
            ).exists(), "failed start must retain the record for inspection"
            assert other.poll() is None
            print("lifecycle ownership, stale PID, occupied port, failed start: passed")
        finally:
            # Stop only the processes created by this test.
            if (root / "run" / "run" / "ninfer-serve-token").exists():
                subprocess.run(
                    ["bash", str(SCRIPT), "stop"], env=env, capture_output=True, timeout=75
                )
            other.terminate()
            other.wait(timeout=10)


if __name__ == "__main__":
    main()
