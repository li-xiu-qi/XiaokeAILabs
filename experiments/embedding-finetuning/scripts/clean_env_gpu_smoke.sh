#!/usr/bin/env bash
# Clean-environment GPU smoke for the fixed T2Retrieval subset.
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
REPO_DIR=$(cd "$EXP_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}
RUN_DIR=$XL_DATA_DIR/embedding-clean-$XL_RUN_ID
VENV_DIR=$XL_DATA_DIR/venvs/embedding-clean-$XL_RUN_ID
HF_HOME=$XL_DATA_DIR/huggingface-$XL_RUN_ID
OUT_DIR=$XL_OUT_DIR/embedding-clean-$XL_RUN_ID

for path in "$RUN_DIR" "$VENV_DIR" "$HF_HOME" "$OUT_DIR"; do
  [[ ! -e "$path" ]] || { echo "FATAL: refusing to reuse existing path: $path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR" "$(dirname "$VENV_DIR")"

# The machine's inherited proxy may be stale. Clear it only for this run.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check --index-url "$TORCH_INDEX_URL" \
  torch==2.15.0+cu132
"$PY" -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' \
  > "$RUN_DIR/torch-constraint.txt"
"$PY" -m pip install --disable-pip-version-check -c "$RUN_DIR/torch-constraint.txt" \
  -r "$EXP_DIR/requirements.txt" -r "$EXP_DIR/requirements-eval.txt"
"$PY" -m pip check
"$PY" -c 'import torch, transformers, sentence_transformers, pandas, pyarrow, huggingface_hub; print("torch",torch.__version__,"CUDA",torch.cuda.is_available(),"transformers",transformers.__version__,"sentence-transformers",sentence_transformers.__version__); assert torch.cuda.is_available()'

export HF_HOME
mkdir -p "$HF_HOME"
DATASET_DIR=$RUN_DIR/dataset
MODEL_DIR=$RUN_DIR/base-model
"$VENV_DIR/bin/hf" download mteb/T2Retrieval --repo-type dataset \
  --revision 921dd3af6e78d1ae7ee0368aa8d7eaee02c8f08e --include '*.parquet' \
  --local-dir "$DATASET_DIR"
"$VENV_DIR/bin/hf" download BAAI/bge-base-zh-v1.5 \
  --revision f03589ceff5aac7111bd60cfc7d497ca17ecac65 --local-dir "$MODEL_DIR"
"$PY" "$EXP_DIR/scripts/t2retrieval_subset.py" prepare --source "$DATASET_DIR" \
  --output "$RUN_DIR/subset"
"$PY" "$EXP_DIR/scripts/t2retrieval_subset.py" evaluate \
  --data "$RUN_DIR/subset/eval.json" --model "$MODEL_DIR" --output "$RUN_DIR/baseline.json"
"$PY" "$EXP_DIR/scripts/train_embedding.py" --model_name_or_path "$MODEL_DIR" \
  --train_dataset "$RUN_DIR/subset/train.jsonl" --output_dir "$OUT_DIR/model" \
  --epochs 1 --batch_size 8
"$PY" "$EXP_DIR/scripts/t2retrieval_subset.py" evaluate \
  --data "$RUN_DIR/subset/eval.json" --model "$OUT_DIR/model/checkpoint-epoch-1" \
  --output "$RUN_DIR/finetuned.json"

"$PY" - "$RUN_DIR" "$OUT_DIR" <<'PY'
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
import torch

run_dir, out_dir = map(Path, sys.argv[1:])
baseline = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
finetuned = json.loads((run_dir / "finetuned.json").read_text(encoding="utf-8"))
summary = {
    "date": "2026-11-15",
    "result": "passed",
    "platform": platform.system(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "transformers": metadata.version("transformers"),
    "sentence_transformers": metadata.version("sentence-transformers"),
    "pandas": metadata.version("pandas"),
    "pyarrow": metadata.version("pyarrow"),
    "huggingface_hub": metadata.version("huggingface-hub"),
    "pip_check": "passed",
    "dataset_revision": "921dd3af6e78d1ae7ee0368aa8d7eaee02c8f08e",
    "model_revision": "f03589ceff5aac7111bd60cfc7d497ca17ecac65",
    "train_queries": 256,
    "test_queries": 100,
    "candidate_documents": 409,
    "baseline": {k: baseline[k] for k in ("hit_at_10", "mrr")},
    "finetuned": {k: finetuned[k] for k in ("hit_at_10", "mrr")},
    "scope": "small fixed T2Retrieval development subset; one training seed",
    "limits": [
        "Candidate pool is small and includes the labeled test positives.",
        "Semantic near-duplicates and pretraining overlap were not ruled out.",
        "This is a workflow smoke run, not evidence of general retrieval gains.",
    ],
}
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "clean-env-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))
PY
