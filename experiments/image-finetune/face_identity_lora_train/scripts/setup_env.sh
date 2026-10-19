#!/usr/bin/env bash
# Install the face-recognition evaluation stack and verify the ArcFace weights load.
#
#   bash scripts/setup_env.sh                      # packages only
#   bash scripts/setup_env.sh --with-models        # also place the two ONNX files
#
# The two ONNX files (det_10g.onnx, w600k_r50.onnx) are not redistributed here.
# Download them from the InsightFace buffalo_l release, then either drop them in
# $FACE_MODELS_DIR (default ~/.insightface/models/buffalo_l) or pass --models-dir.
set -euo pipefail

MODELS_DIR="${FACE_MODELS_DIR:-$HOME/.insightface/models/buffalo_l}"
PYTHON="${PYTHON:-python3}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple/}"

echo "=== install onnxruntime + insightface ==="
"$PYTHON" -m pip install --quiet --disable-pip-version-check \
    --index-url "$PIP_INDEX" onnxruntime insightface

echo "=== verify ==="
"$PYTHON" - <<'PY'
import onnxruntime, insightface
print("onnxruntime", onnxruntime.__version__)
print("insightface", insightface.__version__)
PY

mkdir -p "$MODELS_DIR"
missing=0
for f in det_10g.onnx w600k_r50.onnx; do
    if [ ! -f "$MODELS_DIR/$f" ]; then
        echo "[missing] $MODELS_DIR/$f"
        missing=1
    fi
done

if [ "$missing" -eq 1 ]; then
    cat <<EOF

ArcFace weights are missing. Download buffalo_l and place both files in:
  $MODELS_DIR
or point FACE_MODELS_DIR at wherever they live. Scripts will tell you if the
path is wrong; they do not guess.
EOF
    exit 1
fi

"$PYTHON" - <<'PY'
import os
from insightface.model_zoo import ArcFaceONNX, RetinaFace

mdir = os.environ.get("FACE_MODELS_DIR", os.path.expanduser("~/.insightface/models/buffalo_l"))
det = RetinaFace(model_file=os.path.join(mdir, "det_10g.onnx"))
rec = ArcFaceONNX(os.path.join(mdir, "w600k_r50.onnx"))
det.prepare(ctx_id=0, input_size=(640, 640))
print("models loaded, evaluation stack ready")
PY
