#!/usr/bin/env bash
# install opencv (needed by the torch layout provider) and verify
set -uo pipefail
VENV="$HOME/minicpm-ocr/.venv-ocr"
log() { echo "[$(date '+%F %T')] $*"; }

log "=== install opencv-contrib-python (aarch64 wheel) ==="
"$VENV/bin/python" -m pip install -q "opencv-contrib-python==4.11.0.86" 2>&1 | tail -3
"$VENV/bin/python" - <<'PY' || exit 1
import cv2, numpy
print("cv2", cv2.__version__, "| numpy", numpy.__version__)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
log "=== done ==="
