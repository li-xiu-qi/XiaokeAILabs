#!/usr/bin/env bash
# Create an isolated CUDA environment and run one minimal-model generation check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${XL_DATA_DIR:-$EXPERIMENT_DIR/.work}"
ENV_DIR="${BASELINE_ENV_DIR:-$DATA_ROOT/baseline-llm-clean-env-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${XL_OUT_DIR:-$EXPERIMENT_DIR/.work/clean-env-gpu-smoke-$(date +%Y%m%d_%H%M%S)}"
PYTORCH_SPEC="${PYTORCH_SPEC:-torch==2.15.0+cu132}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}"
MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_REVISION="${QWEN_MODEL_REVISION:-}"
if [[ -z "$MODEL_REVISION" && "$MODEL_ID" == "Qwen/Qwen2.5-0.5B-Instruct" ]]; then
  MODEL_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
fi

if [[ -e "$ENV_DIR" ]]; then
  echo "Refusing to reuse an existing environment: $ENV_DIR" >&2
  echo "Set BASELINE_ENV_DIR to a new path and retry." >&2
  exit 2
fi
mkdir -p "$DATA_ROOT" "$OUT_DIR"
python3 -m venv "$ENV_DIR"
PYTHON="$ENV_DIR/bin/python"

"$PYTHON" -m pip install "$PYTORCH_SPEC" --index-url "$PYTORCH_INDEX_URL"
"$PYTHON" -m pip install -r "$EXPERIMENT_DIR/requirements.txt"
"$PYTHON" -m pip check

export QWEN_MODEL_ID="$MODEL_ID"
if [[ -n "$MODEL_REVISION" ]]; then
  export QWEN_MODEL_REVISION="$MODEL_REVISION"
fi
export QWEN_OUTPUT_DIR="$OUT_DIR"
"$PYTHON" "$SCRIPT_DIR/gpu_smoke.py"
