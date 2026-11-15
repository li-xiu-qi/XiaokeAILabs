#!/usr/bin/env bash
# Build an isolated CUDA environment, obtain verified fixed model revisions,
# and reproduce the bundled synthetic-table detection + recognition sample.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
: "${XL_DATA_DIR:?Set XL_DATA_DIR to an external data directory}"
: "${XL_OUT_DIR:?Set XL_OUT_DIR to an external output directory}"
: "${XL_RUN_ID:?Set XL_RUN_ID to a unique run identifier}"

VENV_DIR="$XL_DATA_DIR/venvs/ppocr-clean-$XL_RUN_ID"
MODEL_DIR="$XL_DATA_DIR/models/ppocr-v5-$XL_RUN_ID"
OUT_JSON="$XL_OUT_DIR/ppocr_metrics.json"
mkdir -p "$(dirname "$VENV_DIR")" "$MODEL_DIR" "$XL_OUT_DIR"
if [ -e "$VENV_DIR" ] || [ -e "$MODEL_DIR/det" ] || [ -e "$MODEL_DIR/rec" ] || [ -e "$OUT_JSON" ]; then
  echo "Refusing to reuse an existing environment, model directory, or output: $XL_RUN_ID" >&2
  exit 2
fi

python3.12 -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"
"$PY" -m pip install pip==24.0
# These are the recorded Linux ARM64 / GB10 builds. Other devices should use
# the matching PyTorch and torchvision builds for their own platform.
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu132 'torch==2.15.0+cu132'
"$PY" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 'torchvision==0.29.1+cu130'
"$PY" -m pip install -r "$EXPERIMENT_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch, torchvision, transformers, cv2; from PIL import Image; assert torch.cuda.is_available(); print("torch", torch.__version__, "torchvision", torchvision.__version__, "transformers", transformers.__version__, "opencv", cv2.__version__, "pillow", Image.__version__, "gpu", torch.cuda.get_device_name(0))'

export HF_HOME="$XL_DATA_DIR/huggingface"
"$PY" - "$MODEL_DIR" "${PP_OCR_MODEL_SOURCE_DIR:-}" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path

root, source_value = Path(sys.argv[1]), sys.argv[2]
expected = {
    "det/model.safetensors": "06e7a44aa5c4146531e88703140e0f7329910b92f84f8bf2756bfd37cec5cb0a",
    "rec/model.safetensors": "da94c5dd42c88b00c44081942b157bee8eda9ccd0481dbd0af3824f6c241aa91",
}

if source_value:
    source = Path(source_value)
    for relative_path in expected:
        source_dir = source / relative_path.split("/", 1)[0]
        if not source_dir.is_dir():
            raise SystemExit(f"Missing cached model directory: {source_dir}")
        shutil.copytree(source_dir, root / source_dir.name, dirs_exist_ok=True)
else:
    from huggingface_hub import snapshot_download

    snapshot_download(
        "PaddlePaddle/PP-OCRv5_server_det_safetensors",
        revision="cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb",
        local_dir=root / "det",
    )
    snapshot_download(
        "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
        revision="542979d7cc3791732bb12af35313a6840952d79f",
        local_dir=root / "rec",
    )

for relative_path, expected_hash in expected.items():
    file_path = root / relative_path
    actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(f"Model hash mismatch for {relative_path}: {actual_hash}")
print("Both model files match the pinned Hub revisions")
PY

"$PY" "$EXPERIMENT_DIR/scripts/run.py" \
  --image "$EXPERIMENT_DIR/figures/synthetic-table.png" \
  --det-model "$MODEL_DIR/det" --rec-model "$MODEL_DIR/rec" \
  --det-device cuda --rec-device cuda --threads 2 \
  --output "$OUT_JSON"

"$PY" - "$EXPERIMENT_DIR" "$MODEL_DIR" "$OUT_JSON" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import cv2
import torch
import torchvision
import transformers
from PIL import Image

experiment_dir, models_dir, output_path = map(Path, sys.argv[1:])
result = json.loads(output_path.read_text(encoding="utf-8"))
texts = [item["prediction"]["text"] for item in result]
expected = ["Item", "Units", "Price", "Notebook", "12", "6.50", "Marker", "24", "2.00", "Paper", "8", "4.00"]
if texts != expected:
    raise SystemExit(f"Recognition output differs from the recorded synthetic sample: {texts!r}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

weights = {
    "det_model.safetensors": sha256(models_dir / "det" / "model.safetensors"),
    "rec_model.safetensors": sha256(models_dir / "rec" / "model.safetensors"),
}
summary = {
    "date": "2026-11-15",
    "experiment": "PP-OCRv5 clean-environment CUDA reproduction",
    "platform": {"system": platform.system(), "architecture": platform.machine(), "python": platform.python_version()},
    "device": torch.cuda.get_device_name(0),
    "packages": {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__,
        "huggingface_hub": metadata.version("huggingface-hub"),
    },
    "models": {
        "detection_revision": "cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb",
        "recognition_revision": "542979d7cc3791732bb12af35313a6840952d79f",
        "weight_sha256": weights,
        "source": "verified local cache" if __import__("os").environ.get("PP_OCR_MODEL_SOURCE_DIR") else "Hugging Face Hub",
    },
    "input": {"file": "figures/synthetic-table.png", "sha256": sha256(experiment_dir / "figures" / "synthetic-table.png")},
    "result": {"device": "cuda", "detections": len(result), "recognized_text": texts, "matches_expected": True},
}
Path(output_path).with_name("summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("Validated 12 expected table text elements on CUDA")
PY
