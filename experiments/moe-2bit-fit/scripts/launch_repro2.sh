#!/usr/bin/env bash
# Run the local reproduction script inside the upstream plugin environment.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PLUGIN_REPO=${PLUGIN_REPO:-$HOME/deepseek-v4-flash-2bit-gb10}
cd "$PLUGIN_REPO"
[[ -f .venv-vllm/bin/activate ]] || { echo "Missing plugin environment: $PLUGIN_REPO/.venv-vllm" >&2; exit 1; }
source .venv-vllm/bin/activate
OUT_DIR=${XL_OUT_DIR:-$PLUGIN_REPO/.work}
mkdir -p "$OUT_DIR"
export CUDA_LAUNCH_BLOCKING=1
export TRITON_PRINT_AUTOTUNING=1
export VLLM_LOGGING_LEVEL=DEBUG
nohup timeout 1200 python "$SCRIPT_DIR/min_repro.py" "$@" > "$OUT_DIR/min_repro2.log" 2>&1 &
echo "launched wrapper pid: $!; log: $OUT_DIR/min_repro2.log"
