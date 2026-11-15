#!/usr/bin/env bash
# Fresh-environment RVM evaluation on the fixed private 30-image sample.
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
WORK_DIR=$XL_DATA_DIR/rvm-$XL_RUN_ID
OUT_DIR=$XL_OUT_DIR/rvm-$XL_RUN_ID
VENV_DIR=${VENV_DIR:-$WORK_DIR/venv}

[[ ! -e "$WORK_DIR" && ! -e "$OUT_DIR" ]] || {
  echo "FATAL: refusing to reuse existing data, environment, or output paths" >&2; exit 2;
}
mkdir -p "$WORK_DIR/weights" "$OUT_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check --index-url "$TORCH_INDEX_URL" \
  "torch==2.15.0+cu132" "torchvision==0.29.1+cu132"
mkdir -p "$WORK_DIR/constraints"
"$PY" -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' \
  > "$WORK_DIR/constraints/torch.txt"
"$PY" -m pip install --disable-pip-version-check -c "$WORK_DIR/constraints/torch.txt" -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch, torchvision, numpy, PIL; print(torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'

git clone --quiet https://github.com/PeterL1n/RobustVideoMatting.git "$WORK_DIR/rvm-src"
git -C "$WORK_DIR/rvm-src" checkout --quiet 53d74c6826735f01f4406b5ca9075eee27bec094
curl -fL --retry 3 https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth \
  -o "$WORK_DIR/weights/rvm_mobilenetv3.pth"
echo '3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8  '"$WORK_DIR"'/weights/rvm_mobilenetv3.pth' | sha256sum -c -
curl -fL --retry 3 https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -o "$WORK_DIR/PPM-100.zip"
"$PY" "$EXP_DIR/scripts/prepare_ppm.py" --archive "$WORK_DIR/PPM-100.zip" --output "$WORK_DIR/local-ppm-30"
"$PY" "$EXP_DIR/scripts/run_rvm_ppm.py" \
  --manifest "$WORK_DIR/local-ppm-30/manifest.jsonl" --output "$OUT_DIR/results" \
  --rvm-repo "$WORK_DIR/rvm-src" --weights "$WORK_DIR/weights/rvm_mobilenetv3.pth" \
  --device cuda --max-side 1024 --warmup 1 --repeats 3

"$PY" - "$OUT_DIR/results/results.json" <<'PY'
import importlib.metadata as metadata
import json
import platform
import statistics
import sys
from pathlib import Path
import torch

run = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "date": "2026-11-15",
    "result": "passed",
    "platform": platform.system(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "torchvision": metadata.version("torchvision"),
    "numpy": metadata.version("numpy"),
    "pillow": metadata.version("pillow"),
    "pip_check": "passed",
    "rvm_commit": run["repo_commit"],
    "weights_sha256": run["weights_sha256"],
    "dataset_manifest_sha256": run["manifest_sha256"],
    "sample_count": run["n"],
    "macro_mean": run["macro_mean"],
    "mean_forward_seconds": statistics.mean(
        x["mean_forward_seconds"] for x in run["samples"]
    ),
    "scope": "fixed 30-image PPM sample; still-image inference only",
    "limits": [
        "No video temporal consistency evaluation.",
        "Dataset annotation license and portrait usage rights remain separate checks."
    ]
}
Path(sys.argv[1]).with_name("clean-env-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))
PY
