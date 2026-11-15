#!/usr/bin/env bash
# Fresh dependency install and synthetic GPU workflow smoke.
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
TORCH_PACKAGE=${TORCH_PACKAGE:-torch==2.15.0+cu132}
RUN_DIR=$XL_DATA_DIR/embedding-distill-clean-$XL_RUN_ID
VENV_DIR=$XL_DATA_DIR/venvs/embedding-distill-clean-$XL_RUN_ID
HF_HOME=$XL_DATA_DIR/huggingface-distill-$XL_RUN_ID
OUT_DIR=$XL_OUT_DIR/embedding-distill-clean-$XL_RUN_ID

for path in "$RUN_DIR" "$VENV_DIR" "$HF_HOME" "$OUT_DIR"; do
  [[ ! -e "$path" ]] || { echo "FATAL: refusing to reuse existing path: $path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR" "$(dirname "$VENV_DIR")" "$OUT_DIR"

if [[ "${XL_CLEAR_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check --index-url "$TORCH_INDEX_URL" "$TORCH_PACKAGE"
"$PY" -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' > "$RUN_DIR/torch-constraint.txt"
"$PY" -m pip install --disable-pip-version-check -c "$RUN_DIR/torch-constraint.txt" -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch, numpy, sentence_transformers, transformers, huggingface_hub; print("torch",torch.__version__,"CUDA",torch.cuda.is_available(),"GPU",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
"$PY" "$EXP_DIR/scripts/synthetic_gpu_smoke.py"

"$PY" - "$OUT_DIR/clean-env-summary.json" <<'PY'
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
import torch

summary = {
    "date": "2026-11-15",
    "result": "passed",
    "os": platform.system(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "numpy": metadata.version("numpy"),
    "sentence_transformers": metadata.version("sentence-transformers"),
    "transformers": metadata.version("transformers"),
    "huggingface_hub": metadata.version("huggingface-hub"),
    "pip_check": "passed",
    "scope": "synthetic GPU scoring and two-step updates for supervised and distilled student losses",
    "limits": [
        "No external model weights or real dataset were used.",
        "This checks the synthetic workflow and dependency setup, not the real-data teacher gate or full experiment."
    ]
}
Path(sys.argv[1]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
