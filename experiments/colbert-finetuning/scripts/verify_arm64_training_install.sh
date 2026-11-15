#!/usr/bin/env bash
set -Eeuo pipefail

EXP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${XL_RUN_ID:?}
DATA_DIR=${XL_DATA_DIR:?}
OUT_DIR=${XL_OUT_DIR:?}
VENV_DIR="$DATA_DIR/venvs/colbert-pylate14-train-only-$RUN_ID"
TORCH_SPEC=${XL_TORCH_SPEC:-'torch==2.15.0+cu132'}
TORCH_INDEX_URL=${XL_TORCH_INDEX_URL:-'https://download.pytorch.org/whl/cu132'}

mkdir -p "$DATA_DIR" "$OUT_DIR" "$(dirname "$VENV_DIR")"
if [[ -e "$OUT_DIR/summary.json" || -e "$VENV_DIR" ]]; then
  echo "refusing to overwrite isolated environment or results" >&2
  exit 2
fi

STAGE=venv
python3.12 -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --disable-pip-version-check pip==24.0

STAGE=torch
"$PY" -m pip install --disable-pip-version-check --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
"$PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
"$PY" -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > "$DATA_DIR/torch-constraint-$RUN_ID.txt"

STAGE=base-dependencies
"$PY" -m pip install --disable-pip-version-check -c "$DATA_DIR/torch-constraint-$RUN_ID.txt" -r "$EXP_DIR/requirements-base.txt"

STAGE=pylate
"$PY" -m pip install --disable-pip-version-check --no-deps 'pylate==1.4.0'

STAGE=dependency-audit
if "$PY" -m pip check > "$OUT_DIR/pip-check.txt" 2>&1; then
  echo "Expected fast-plaid to remain the sole missing optional-for-this-run dependency" >&2
  exit 3
fi
if ! python3.12 - "$OUT_DIR/pip-check.txt" <<'PY'
from pathlib import Path
import sys

lines = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
expected = {"pylate 1.4.0 requires fast-plaid, which is not installed."}
if set(lines) != expected:
    print("Unexpected pip check result:", lines, file=sys.stderr)
    raise SystemExit(1)
PY
then
  exit 4
fi

STAGE=imports
"$PY" -c 'import torch, numpy, datasets, sentence_transformers, pylate; from pylate import models, losses, utils, rank; print("imports passed", torch.__version__, pylate.__version__)'

STAGE=training-and-reranking
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$EXP_DIR/scripts/synthetic_gpu_smoke.py" 2>&1 | tee "$OUT_DIR/training-smoke.log"

STAGE=summary
"$PY" - "$OUT_DIR/summary.json" "$OUT_DIR/pip-check.txt" <<'PY'
import importlib.metadata as metadata
import json
import platform
import sys
import torch

summary = {
    "date": "2026-11-15",
    "experiment": "PyLate 1.4.0 ARM64 training/reranking dependency probe",
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "pylate": metadata.version("pylate"),
    "sentence_transformers": metadata.version("sentence-transformers"),
    "transformers": metadata.version("transformers"),
    "pip_check": "one missing PyLate direct dependency: fast-plaid",
    "omitted_dependency_scope": "fast-plaid indexing backend; the exercised code trains and reranks a synthetic batch and does not build or query a PLAID index",
    "smoke": "completed; see training-smoke.log",
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
PY

STAGE=complete
