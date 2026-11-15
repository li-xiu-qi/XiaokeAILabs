#!/usr/bin/env bash
# Fresh-cache, fresh-venv end-to-end DPO run on a CUDA/BF16 device.
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
OUT_DIR=$XL_OUT_DIR/dpo-clean-$XL_RUN_ID
VENV_DIR=${VENV_DIR:-$XL_DATA_DIR/venvs/dpo-clean-$XL_RUN_ID}
HF_HOME=$XL_DATA_DIR/huggingface

mkdir -p "$XL_DATA_DIR" "$(dirname "$VENV_DIR")"
[[ ! -e "$OUT_DIR" && ! -e "$VENV_DIR" && ! -e "$HF_HOME" ]] || {
  echo "FATAL: refusing to reuse an existing environment, cache, or output path" >&2; exit 2;
}
mkdir -p "$OUT_DIR" "$HF_HOME"

"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check --index-url "$TORCH_INDEX_URL" "torch==2.15.0+cu132"
mkdir -p "$XL_DATA_DIR/constraints"
"$PY" -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' \
  > "$XL_DATA_DIR/constraints/torch-$XL_RUN_ID.txt"
"$PY" -m pip install --disable-pip-version-check \
  -c "$XL_DATA_DIR/constraints/torch-$XL_RUN_ID.txt" -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
"$PY" -c 'import transformers, datasets, peft, accelerate; print("imports ok", transformers.__version__, datasets.__version__, peft.__version__, accelerate.__version__)'

export PYTHON=$PY XL_OUT_DIR=$OUT_DIR XL_DATA_DIR HF_HOME
"$PY" "$EXP_DIR/scripts/run_dpo.py" --train-size 256 --test-size 128 --steps 64 --seed 42

"$PY" - "$OUT_DIR/report.json" <<'PY'
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
import torch

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
summary = {
    "date": "2026-11-15",
    "result": "passed",
    "platform": platform.system(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": metadata.version("transformers"),
    "datasets": metadata.version("datasets"),
    "peft": metadata.version("peft"),
    "accelerate": metadata.version("accelerate"),
    "pip_check": "passed",
    "model_revision": report["model_revision"],
    "dataset_revision": report["dataset_revision"],
    "train_size": report["train_size"],
    "test_size": report["test_size"],
    "steps": report["parameters"]["steps"],
    "train_loss_first": report["train_loss_first"],
    "train_loss_last": report["train_loss_last"],
    "base_pair_accuracy": report["base_pair_accuracy"],
    "trained_pair_accuracy": report["trained_pair_accuracy"],
    "pair_accuracy_change_bootstrap_95pct": report["pair_accuracy_change_bootstrap_95pct"],
    "mean_change_in_reference_relative_margin": report["mean_change_in_reference_relative_margin"],
    "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
    "limits": [
        "Small, single-seed run; it does not establish general alignment improvement.",
        "Preference labels are model-scored; truncation and possible pretraining overlap remain limitations."
    ]
}
report_path.with_name("clean-env-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))
PY
