#!/usr/bin/env bash
# Launch LoRA training with ai-toolkit.
#
#   bash scripts/train_lora.sh                                   # 2000 steps, lr 1e-4
#   bash scripts/train_lora.sh config/face_identity_lora_regularized.yaml
#
# Expects:
#   - ai-toolkit checked out somewhere; set AI_TOOLKIT to its run.py
#   - training images at $FACE_DATA_DIR/celeb_14/train/*.jpg with matching .txt captions
#   - a base model at $SDXL_BASE (local path or HF repo id)
#
# Nothing here is machine-specific: every path comes from an environment
# variable, and the config reads the same variables.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-config/face_identity_lora.yaml}"
AI_TOOLKIT="${AI_TOOLKIT:-$HOME/ai-toolkit/run.py}"
PYTHON="${PYTHON:-python3}"

DATA_DIR="${FACE_DATA_DIR:-$PWD/data}"
TARGET_ID="${FACE_TARGET_ID:-14}"
TRAIN_DIR="$DATA_DIR/celeb_${TARGET_ID}/train"

if [ ! -f "$AI_TOOLKIT" ]; then
    echo "[error] ai-toolkit not found at $AI_TOOLKIT" >&2
    echo "        clone https://github.com/ostris/ai-toolkit and set AI_TOOLKIT=/path/to/run.py" >&2
    exit 1
fi

if [ ! -d "$TRAIN_DIR" ]; then
    echo "[error] training images not found at $TRAIN_DIR" >&2
    echo "        run: python scripts/build_dataset.py --parquet /path/to/celeba_shard0.parquet" >&2
    exit 1
fi

n_images=$(find "$TRAIN_DIR" -name '*.jpg' | wc -l | tr -d ' ')
n_captions=$(find "$TRAIN_DIR" -name '*.txt' | wc -l | tr -d ' ')
echo "=== training set: $n_images images, $n_captions captions in $TRAIN_DIR ==="
if [ "$n_images" -ne "$n_captions" ]; then
    echo "[error] image/caption count mismatch — ai-toolkit silently skips uncaptioned images" >&2
    exit 1
fi

# The dataset has to clear the four gates in check_dataset.py before any GPU
# time is spent. Set SKIP_CHECK=1 only when you know the data was audited and
# nothing changed since.
if [ "${SKIP_CHECK:-0}" != "1" ]; then
    echo "=== checking the training data ==="
    "$PYTHON" scripts/check_dataset.py
fi

echo "=== launching $CONFIG ==="
"$PYTHON" "$AI_TOOLKIT" "$CONFIG"
