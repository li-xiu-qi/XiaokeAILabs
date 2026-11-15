#!/usr/bin/env bash
# Fresh-environment CUDA smoke check for the actual CLIP training loop.
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
TORCH_SPEC=${TORCH_SPEC:-torch==2.15.0}
TORCHVISION_SPEC=${TORCHVISION_SPEC:-torchvision==0.29.1}
VENV_DIR=${VENV_DIR:-$XL_DATA_DIR/venvs/clip-smoke-$XL_RUN_ID}
OUT_DIR=$XL_OUT_DIR/clip-smoke-$XL_RUN_ID

mkdir -p "$XL_DATA_DIR" "$XL_OUT_DIR" "$(dirname "$VENV_DIR")"
[[ ! -e "$VENV_DIR" && ! -e "$OUT_DIR" ]] || {
  echo "FATAL: refusing to reuse an existing environment or output path" >&2; exit 2;
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check "$TORCH_SPEC" "$TORCHVISION_SPEC" --index-url "$TORCH_INDEX_URL"
"$PY" -m pip install --disable-pip-version-check -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch, transformers, datasets, matplotlib, PIL; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available(),"transformers",transformers.__version__); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'

mkdir -p "$OUT_DIR"
XL_OUT_DIR=$OUT_DIR export XL_OUT_DIR
cd "$EXP_DIR"
"$PY" tests/smoke_cuda_train.py

"$PY" - "$OUT_DIR" <<'PY'
import json, os, platform, sys
import torch, torchvision, transformers, datasets, matplotlib, PIL
out = sys.argv[1]
summary = {
    "date": "2026-11-15",
    "purpose": "Fresh-environment CUDA smoke check for the CLIP fine-tuning loop; not a Flickr30k reproduction.",
    "hardware": torch.cuda.get_device_name(0),
    "environment": {
        "platform": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "transformers": transformers.__version__,
        "datasets": datasets.__version__, "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__, "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "bfloat16_supported": torch.cuda.is_bf16_supported(),
        "pip_check": "passed"
    },
    "model": "random-initialized tiny CLIP created by the test harness; no pretrained checkpoint was downloaded",
    "data": "synthetic image-text pairs; four samples per split",
    "check": {"optimizer_steps": 2, "weights_changed": True,
              "validation_test_and_save": True, "result": "passed"},
    "limits": [
        "Processor, model loading, and data preparation were substituted in the smoke harness.",
        "The run used FP32; it does not verify BF16 or pretrained-weight loading.",
        "It does not validate Flickr30k processing or retrieval quality and does not reproduce the real-data run."
    ]
}
with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
