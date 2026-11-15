#!/usr/bin/env bash
# Clean-environment CUDA training-path check using the bundled synthetic examples.
set -Eeuo pipefail

for name in XL_RUN_ID XL_DATA_DIR XL_OUT_DIR; do
  [[ -n "${!name:-}" ]] || { echo "FATAL: $name is required" >&2; exit 2; }
done
[[ "${XL_REQUIRE_EXTERNAL_ENV:-0}" == "1" ]] || {
  echo "FATAL: set XL_REQUIRE_EXTERNAL_ENV=1 and use external data/output directories" >&2
  exit 2
}
[[ "$XL_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "FATAL: invalid XL_RUN_ID" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}
TORCH_SPEC=${TORCH_SPEC:-torch==2.15.0}
MODEL_ID=hf-internal-testing/tiny-random-bert
MODEL_REVISION=f171d7baecaf37b5da5a3616d8833b9969753535
RUN_DATE=$(date -u +%F)
HF_HOME=${HF_HOME:-$XL_DATA_DIR/huggingface}
VENV_DIR=${VENV_DIR:-$XL_DATA_DIR/venvs/sparse-$XL_RUN_ID}
OUT_DIR=$XL_OUT_DIR/sparse-retrieval-$XL_RUN_ID

export HF_HOME HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$XL_DATA_DIR" "$XL_OUT_DIR" "$(dirname "$VENV_DIR")"
[[ ! -e "$VENV_DIR" && ! -e "$OUT_DIR" ]] || {
  echo "FATAL: refusing to reuse an existing environment or output path" >&2; exit 2;
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
"$PY" -m pip install --disable-pip-version-check -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch, transformers; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available(), "transformers:", transformers.__version__); assert torch.cuda.is_available(), "CUDA GPU required"'

MODEL_DIR=$("$PY" - "$MODEL_ID" "$MODEL_REVISION" "$HF_HOME" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3]))
PY
)
[[ ! -e "$OUT_DIR" ]] || { echo "FATAL: output already exists: $OUT_DIR" >&2; exit 2; }

"$PY" "$EXP_DIR/src/train.py" --model_name_or_path "$MODEL_DIR" --synthetic_data \
  --epochs 1 --batch_size 2 --max_len_q 16 --max_len_d 32 \
  --output_dir "$OUT_DIR/training"

"$PY" - "$OUT_DIR" "$MODEL_ID" "$MODEL_REVISION" "$RUN_DATE" <<'PY'
import json, os, platform, sys
import torch, transformers
out, model_id, revision, run_date = sys.argv[1:]
with open(os.path.join(out, "training", "training_metrics.json"), encoding="utf-8") as f:
    metrics = json.load(f)
summary = {
    "date": run_date,
    "experiment": "sparse-retrieval clean-environment CUDA training check",
    "status": "completed",
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "cuda_build": torch.version.cuda,
    "packages": {"torch": torch.__version__, "transformers": transformers.__version__},
    "model": {"id": model_id, "revision": revision},
    "data": {"kind": "four bundled synthetic triplets", "examples": 4},
    "training": {"epochs": 1, "optimizer_steps": len(metrics), "batch_size": 2,
        "max_len_q": 16, "max_len_d": 32},
    "artifacts": {"weights_saved": os.path.isfile(os.path.join(out, "training", "splade_weights.pt")),
        "metrics_saved": True},
    "pip_check": "passed",
    "limitations": ["Synthetic data and a random tiny model only check installation and the CUDA training path.",
        "This does not validate retrieval quality or satisfy the experiment success criteria."],
}
with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
