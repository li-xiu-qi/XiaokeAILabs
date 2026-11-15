#!/usr/bin/env bash
set -Eeuo pipefail
EXP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${XL_RUN_ID:?}
DATA_DIR=${XL_DATA_DIR:?}
OUT_DIR=${XL_OUT_DIR:?}
VENV_DIR="$DATA_DIR/venvs/colbert-pylate14-$RUN_ID"
mkdir -p "$DATA_DIR" "$OUT_DIR" "$(dirname "$VENV_DIR")"
if [[ -e "$OUT_DIR/summary.json" || -e "$VENV_DIR" ]]; then
  echo "refusing to overwrite isolated environment or results" >&2
  exit 2
fi
STAGE=venv
python3.12 -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"
STAGE=torch
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu132 'torch==2.15.0+cu132'
"$PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
"$PY" -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > "$DATA_DIR/torch-constraint-$RUN_ID.txt"
STAGE=complete-dependencies
"$PY" -m pip install --disable-pip-version-check -c "$DATA_DIR/torch-constraint-$RUN_ID.txt" -r "$EXP_DIR/requirements.txt"
STAGE=pip-check
"$PY" -m pip check
STAGE=imports
"$PY" -c 'import torch, numpy, datasets, sentence_transformers, pylate; print("imports passed", torch.__version__, pylate.__version__)'
STAGE=summary
"$PY" - "$OUT_DIR/summary.json" <<'PY'
import json, platform, sys, torch, numpy, datasets, sentence_transformers, pylate
json.dump({"date":"2026-11-15","status":"environment_ready","purpose":"PyLate 1.4.0 complete dependency installation gate; not a training run","platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"numpy":numpy.__version__,"datasets":datasets.__version__,"sentence_transformers":sentence_transformers.__version__,"pylate":pylate.__version__,"pip_check":"passed"},open(sys.argv[1],"w"),indent=2)
PY
STAGE=complete
