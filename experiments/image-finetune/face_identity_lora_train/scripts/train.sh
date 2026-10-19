#!/usr/bin/env bash
# Launch LoRA training with ai-toolkit.
#
#   bash scripts/train.sh                    # config/face_identity_lora.yaml (v1, 2000 steps)
#   bash scripts/train.sh face_identity_lora_v2.yaml
#
# Expects:
#   - ai-toolkit checked out somewhere; set AI_TOOLKIT to its run.py
#   - training images at $FACE_DATA_DIR/celeb_14/train/*.jpg with matching .txt captions
#   - a base model at $SDXL_BASE (local path or HF repo id)
#
# The config reads paths from the environment too, so nothing here is machine-specific.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-config/face_identity_lora.yaml}"
AI_TOOLKIT="${AI_TOOLKIT:-$HOME/ai-toolkit/run.py}"
PYTHON="${PYTHON:-python3}"

DATA_DIR="${FACE_DATA_DIR:-$PWD/data}"
TRAIN_DIR="$DATA_DIR/celeb_14/train"

if [ ! -f "$AI_TOOLKIT" ]; then
    echo "[error] ai-toolkit not found at $AI_TOOLKIT" >&2
    echo "        clone https://github.com/ostris/ai-toolkit and set AI_TOOLKIT=/path/to/run.py" >&2
    exit 1
fi

if [ ! -d "$TRAIN_DIR" ]; then
    echo "[error] training images not found at $TRAIN_DIR" >&2
    echo "        run: python scripts/prepare_data.py --parquet /path/to/celeba_shard0.parquet" >&2
    exit 1
fi

n_images=$(find "$TRAIN_DIR" -name '*.jpg' | wc -l | tr -d ' ')
n_captions=$(find "$TRAIN_DIR" -name '*.txt' | wc -l | tr -d ' ')
echo "=== training set: $n_images images, $n_captions captions in $TRAIN_DIR ==="
if [ "$n_images" -ne "$n_captions" ]; then
    echo "[error] image/caption count mismatch — ai-toolkit will silently skip uncaptioned images" >&2
    exit 1
fi

echo "=== launching $CONFIG ==="
"$PYTHON" "$AI_TOOLKIT" "$CONFIG"
