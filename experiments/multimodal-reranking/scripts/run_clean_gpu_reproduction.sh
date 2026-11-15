#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

: "${XL_DATA_DIR:?Set XL_DATA_DIR to an external data directory}"
: "${XL_OUT_DIR:?Set XL_OUT_DIR to an external output directory}"
: "${XL_RUN_ID:?Set XL_RUN_ID to a unique run identifier}"
: "${BGE_MODEL:?Set BGE_MODEL to a local BGE-VL-base model directory}"
: "${QWEN_MODEL:?Set QWEN_MODEL to a local Qwen3-VL-Reranker-2B model directory}"

case "$XL_RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "XL_RUN_ID may contain only letters, numbers, dot, underscore, and dash" >&2; exit 2 ;;
esac

VENV_DIR="$XL_DATA_DIR/$XL_RUN_ID/venv"
if [[ -e "$VENV_DIR" ]]; then
  echo "Refusing to reuse an existing environment: $VENV_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname -- "$VENV_DIR")" "$XL_OUT_DIR"
if [[ -e "$XL_OUT_DIR/summary.json" ]]; then
  echo "Refusing to overwrite: $XL_OUT_DIR/summary.json" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"

"$PY" -m pip install pip==24.0
"$PY" -m pip install torch==2.14.0 torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check

"$PY" - "$XL_OUT_DIR" <<'PY'
import json
import platform
import sys
import datasets
import huggingface_hub
import numpy
import torch
import torchvision
import transformers
import sentence_transformers
import qwen_vl_utils
from PIL import Image
import scipy

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU does not report BF16 support")

record = {
    "python": platform.python_version(),
    "architecture": platform.machine(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "qwen_vl_utils": getattr(qwen_vl_utils, "__version__", "0.0.14"),
    "pillow": Image.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "datasets": datasets.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "gpu": torch.cuda.get_device_name(0),
}
with open(sys.argv[1] + "/environment.json", "x", encoding="utf-8") as f:
    json.dump(record, f, indent=2)
    f.write("\n")
print(json.dumps(record, indent=2))
PY

FIXTURE_DIR="$XL_OUT_DIR/synthetic"
mkdir -p "$FIXTURE_DIR/images"
"$PY" - "$FIXTURE_DIR" <<'PY'
import csv
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
colors = ["red", "green", "blue", "yellow", "purple"]
with (root / "annotations.csv").open("x", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "raw", "filename"])
    writer.writeheader()
    for i, color in enumerate(colors):
        name = f"{i}.png"
        Image.new("RGB", (96, 96), color).save(root / "images" / name)
        writer.writerow({"split": "test", "raw": repr([f"a solid {color} square"]), "filename": name})
PY

"$PY" "$SCRIPT_DIR/candidates.py" \
  --annotations "$FIXTURE_DIR/annotations.csv" \
  --images "$FIXTURE_DIR/images" \
  --model "$BGE_MODEL" \
  --sample-size 5 \
  --output "$XL_OUT_DIR/candidates.json"
"$PY" "$SCRIPT_DIR/rerank.py" \
  --candidates "$XL_OUT_DIR/candidates.json" \
  --images "$FIXTURE_DIR/images" \
  --model "$QWEN_MODEL" \
  --output "$XL_OUT_DIR/reranked.jsonl"
"$PY" "$SCRIPT_DIR/anonymize.py" \
  --candidates "$XL_OUT_DIR/candidates.json" \
  --reranked "$XL_OUT_DIR/reranked.jsonl" \
  --expected-count 5 \
  --output "$XL_OUT_DIR/anonymous.json"
"$PY" "$SCRIPT_DIR/summarize.py" "$XL_OUT_DIR/anonymous.json" \
  --expected-count 5 > "$XL_OUT_DIR/metrics.json"

"$PY" - "$XL_OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "environment.json").open(encoding="utf-8") as f:
    environment = json.load(f)
with (root / "metrics.json").open(encoding="utf-8") as f:
    metrics = json.load(f)
if metrics.get("n") != 5:
    raise SystemExit("Expected five completed synthetic queries")
with (root / "summary.json").open("x", encoding="utf-8") as f:
    json.dump({"environment": environment, "metrics": metrics}, f, indent=2)
    f.write("\n")
PY

echo "Clean-environment GPU reproduction completed: $XL_OUT_DIR/summary.json"
