#!/usr/bin/env bash
# Clean-environment GPU training check for the documented Linux aarch64 build.
set -Eeuo pipefail

for name in XL_RUN_ID XL_DATA_DIR XL_OUT_DIR; do
  [[ -n "${!name:-}" ]] || { echo "FATAL: $name is required" >&2; exit 2; }
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd "$EXP_DIR/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}
TORCH_SPEC=${TORCH_SPEC:-torch==2.15.0}
MODEL_ID=${MODEL_ID:-cross-encoder/ms-marco-TinyBERT-L2-v2}
MODEL_REVISION=${MODEL_REVISION:-81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc}
HF_HOME=${HF_HOME:-$XL_DATA_DIR/huggingface}
VENV_DIR=${VENV_DIR:-$XL_DATA_DIR/venvs/pointwise-$XL_RUN_ID}
OUT_DIR=$XL_OUT_DIR/pointwise-bge-$XL_RUN_ID

export HF_HOME HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$XL_DATA_DIR" "$XL_OUT_DIR" "$(dirname "$VENV_DIR")"
[[ ! -e "$VENV_DIR" && ! -e "$OUT_DIR" ]] || {
  echo "FATAL: refusing to reuse existing environment/output path" >&2; exit 2;
}

STAGE=create_environment
"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python

STAGE=install_pytorch
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"

STAGE=install_experiment_dependencies
"$PY" -m pip install --disable-pip-version-check -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check

STAGE=download_pinned_model
MODEL_DIR=$("$PY" - "$MODEL_ID" "$MODEL_REVISION" "$HF_HOME" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, cache = sys.argv[1:]
print(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache))
PY
)

STAGE=verify_cuda
"$PY" - "$OUT_DIR" "$MODEL_DIR" "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import json, os, platform, sys
import torch, transformers, accelerate, numpy, tqdm, matplotlib
out, model_dir, model_id, revision = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 is unavailable; choose --mixed-precision no and re-run")
os.makedirs(out, exist_ok=False)
summary = {
    "experiment": "pointwise-bge clean-environment GPU training",
    "status": "running",
    "stage": "environment_ready",
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "cuda_build": torch.version.cuda,
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "packages": {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "accelerate": accelerate.__version__, "numpy": numpy.__version__,
        "tqdm": tqdm.__version__, "matplotlib": matplotlib.__version__,
    },
    "model": {"id": model_id, "revision": revision, "local_snapshot": model_dir},
}
with open(os.path.join(out, "environment-summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

STAGE=train
DATA=$EXP_DIR/default_train_data.jsonl
DATA_SHA256=$(sha256sum "$DATA" | awk '{print $1}')
TRAIN_STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_SECONDS=$(date +%s)
CUBLAS_WORKSPACE_CONFIG=:16:8 PYTHONHASHSEED=42 "$PY" \
  "$EXP_DIR/train_bert_rerank.py" \
  --model-name-or-path "$MODEL_DIR" --train-dataset "$DATA" \
  --output-dir "$OUT_DIR/model" --epochs 1 --batch-size 4 \
  --gradient-accumulation-steps 2 --mixed-precision bf16 --max-len 128 \
  --lr 1e-5 --seed 42 2>&1 | tee "$OUT_DIR/training.log"
ELAPSED_SECONDS=$(( $(date +%s) - START_SECONDS ))

STAGE=write_summary
"$PY" - "$OUT_DIR" "$DATA" "$DATA_SHA256" "$MODEL_ID" \
  "$MODEL_REVISION" "$TRAIN_STARTED" "$ELAPSED_SECONDS" <<'PY'
import json, math, os, sys
import torch, transformers, accelerate, numpy, tqdm, matplotlib
out, data, data_sha256, model_id, revision, started, elapsed = sys.argv[1:]
examples = sum(1 for line in open(data, encoding="utf-8") if line.strip())
batch_size, accumulation, epochs = 4, 2, 1
microbatches = math.ceil(examples / batch_size)
updates = math.ceil(microbatches / accumulation) * epochs
model_dir = os.path.join(out, "model")
assert os.path.isdir(os.path.join(model_dir, "final_model")), "trained model missing"
assert os.path.isfile(os.path.join(model_dir, "loss_curve.png")), "loss curve missing"
summary = {
    "date": started[:10], "experiment": "pointwise-bge clean-environment GPU training",
    "status": "completed", "platform": __import__("platform").platform(),
    "architecture": __import__("platform").machine(), "python": __import__("platform").python_version(),
    "gpu": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "packages": {"torch": torch.__version__, "transformers": transformers.__version__,
        "accelerate": accelerate.__version__, "numpy": numpy.__version__,
        "tqdm": tqdm.__version__, "matplotlib": matplotlib.__version__},
    "model": {"id": model_id, "revision": revision},
    "data": {"path": "default_train_data.jsonl", "examples": examples,
        "sha256": data_sha256, "labels": "repository example data; not a held-out evaluation set"},
    "training": {"epochs": epochs, "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation, "optimizer_updates": updates,
        "max_length": 128, "precision": "bf16", "learning_rate": 1e-5,
        "seed": 42, "wall_seconds": int(elapsed)},
    "artifacts": {"model_saved": True, "loss_curve_saved": True},
    "pip_check": "passed",
    "limitations": ["Training flow check only; no held-out reranking quality evaluation.",
        "Repository example data and one epoch do not reproduce the historical default BGE run."],
}
with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
