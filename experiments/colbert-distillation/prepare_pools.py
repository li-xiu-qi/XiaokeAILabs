"""Compatibility entry point for the shared SciFact candidate-pool tool."""

from pathlib import Path
import runpy
import sys

shared = Path(__file__).resolve().parents[2] / "utils" / "scifact-pools"
# Preserve the previously documented invocation from this experiment directory.
if "--config" in sys.argv:
    index = sys.argv.index("--config") + 1
    if index < len(sys.argv) and sys.argv[index] == "pool_config.json" and not Path(sys.argv[index]).exists():
        sys.argv[index] = str(shared / "pool_config.json")
runpy.run_path(str(shared / "prepare_pools.py"), run_name="__main__")
