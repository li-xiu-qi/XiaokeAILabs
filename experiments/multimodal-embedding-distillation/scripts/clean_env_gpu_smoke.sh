#!/usr/bin/env bash
# Fresh dependency install and synthetic image-text distillation GPU smoke.
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
RUN_DIR=$XL_DATA_DIR/multimodal-embedding-distill-clean-$XL_RUN_ID
VENV_DIR=$XL_DATA_DIR/venvs/multimodal-embedding-distill-clean-$XL_RUN_ID
OUT_DIR=$XL_OUT_DIR/multimodal-embedding-distill-clean-$XL_RUN_ID
SMOKE_DIR=$RUN_DIR/synthetic-smoke

for path in "$RUN_DIR" "$VENV_DIR" "$OUT_DIR" "$SMOKE_DIR"; do
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
"$PY" -c 'import torch, transformers, sentence_transformers, peft, numpy, PIL, datasets, huggingface_hub; print("torch",torch.__version__,"CUDA",torch.cuda.is_available(),"BF16",torch.cuda.is_bf16_supported()); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
"$PY" "$EXP_DIR/scripts/synthetic_gpu_smoke.py" --out "$SMOKE_DIR"

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
    "transformers": metadata.version("transformers"),
    "tokenizers": metadata.version("tokenizers"),
    "sentence_transformers": metadata.version("sentence-transformers"),
    "peft": metadata.version("peft"),
    "numpy": metadata.version("numpy"),
    "pillow": metadata.version("pillow"),
    "datasets": metadata.version("datasets"),
    "huggingface_hub": metadata.version("huggingface-hub"),
    "pip_check": "passed",
    "cuda": True,
    "bfloat16": torch.cuda.is_bf16_supported(),
    "scope": "fresh environment install and synthetic image-text student GPU smoke",
    "limits": [
        "No Flickr30k images, teacher model, or student checkpoint were downloaded.",
        "This does not reproduce the real-data teacher gate or historical experiment results."
    ]
}
Path(sys.argv[1]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
