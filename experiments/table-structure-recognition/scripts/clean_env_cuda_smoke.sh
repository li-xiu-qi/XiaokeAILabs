#!/usr/bin/env bash
# Fresh-environment, pinned-model CUDA smoke for SLANet+ table structure.
set -Eeuo pipefail

for name in XL_RUN_ID XL_DATA_DIR XL_OUT_DIR; do
  [[ -n "${!name:-}" ]] || { echo "FATAL: $name is required" >&2; exit 2; }
done
[[ "${XL_REQUIRE_EXTERNAL_ENV:-0}" == "1" ]] || {
  echo "FATAL: set XL_REQUIRE_EXTERNAL_ENV=1 and use external data/output directories" >&2; exit 2;
}
[[ "$XL_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "FATAL: invalid XL_RUN_ID" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}
OUT_DIR=$XL_OUT_DIR/slanet-$XL_RUN_ID
VENV_DIR=${VENV_DIR:-$XL_DATA_DIR/venvs/slanet-$XL_RUN_ID}
MODEL_DIR=${MODEL_DIR:-$XL_DATA_DIR/models/slanet-plus-e44d12b}
HF_HOME=$XL_DATA_DIR/huggingface

mkdir -p "$XL_DATA_DIR" "$(dirname "$VENV_DIR")" "$(dirname "$MODEL_DIR")"
[[ ! -e "$OUT_DIR" && ! -e "$VENV_DIR" ]] || {
  echo "FATAL: refusing to reuse an existing environment or output path" >&2; exit 2;
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check --index-url "$TORCH_INDEX_URL" \
  "torch==2.15.0+cu132" "torchvision==0.29.1+cu132"
mkdir -p "$XL_DATA_DIR/constraints"
"$PY" -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' \
  > "$XL_DATA_DIR/constraints/torch-$XL_RUN_ID.txt"
"$PY" -m pip install --disable-pip-version-check -c "$XL_DATA_DIR/constraints/torch-$XL_RUN_ID.txt" -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch, torchvision, transformers; from PIL import Image; print(torch.__version__, torchvision.__version__, transformers.__version__); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'

export HF_HOME
"$PY" - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    "PaddlePaddle/SLANet_plus_safetensors",
    revision="e44d12b3fe695792170dd46623075f63d8544461",
    local_dir=sys.argv[1],
)
PY

"$PY" "$EXP_DIR/scripts/slanet_table_smoke.py" \
  --model "$MODEL_DIR" --out "$OUT_DIR" --crop-table --device cuda

"$PY" - "$OUT_DIR/result.json" <<'PY'
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
import torch
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
data = {
    "date": "2026-11-15",
    "result": "passed",
    "platform": platform.system(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": metadata.version("torchvision"),
    "transformers": metadata.version("transformers"),
    "pillow": metadata.version("pillow"),
    "pip_check": "passed",
    "gpu": torch.cuda.get_device_name(0),
    "model_revision": "e44d12b3fe695792170dd46623075f63d8544461",
    "model_sha256": result["model_sha256"],
    "input_sha256": result["input_sha256"],
    "device": result["device"],
    "first_forward_seconds": result["first_forward_s"],
    "rows": result["result"]["structure"].count("<tr>"),
    "cells": result["result"]["structure"].count("<td></td>"),
    "structure_score": result["result"]["structure_score"],
    "limitations": ["One synthetic table only; not an OCR or real-table accuracy evaluation."],
}
Path(sys.argv[1]).with_name("clean-env-summary.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(data, ensure_ascii=False))
PY
