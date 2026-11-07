"""Run a command with an available-memory floor (Linux only)."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def available_gib():
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable unavailable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimum-gib", type=float, default=50)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        ap.error("missing command")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    start = available_gib()
    if start < args.minimum_gib:
        raise SystemExit(f"preflight failed: MemAvailable={start:.2f} GiB")
    low = start
    aborted = False
    # Open exclusively before spawning so a log failure cannot orphan a child.
    with args.log.open("x", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, start_new_session=True)
        try:
            while process.poll() is None:
                avail = available_gib()
                low = min(low, avail)
                f.write(f"{time.time():.3f} {avail:.3f}\n")
                f.flush()
                if avail < args.minimum_gib:
                    print(f"memory floor breached: {avail:.2f} GiB", flush=True)
                    aborted = True
                    break
                time.sleep(1)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
    print(f"preflight={start:.2f} GiB minimum={low:.2f} GiB exit={process.returncode}", flush=True)
    sys.exit(2 if aborted else process.returncode)


if __name__ == "__main__":
    main()
