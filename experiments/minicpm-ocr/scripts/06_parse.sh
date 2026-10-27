#!/usr/bin/env bash
# end-to-end parse with the SDK: layout (torch provider) + recognition service + markdown assembly
set -uo pipefail
BASE="$HOME/minicpm-ocr"
MODEL="${MODEL:-$BASE/models/MiniCPM-OCR-1.0}"
VENV="$BASE/.venv-ocr"
INPUT="${1:-$BASE/testdata/testpage_cn.jpg}"
TAG="${2:-run1}"
OUT="$BASE/runs/$TAG"
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

log "=== preflight ==="
bash "$HOME/preflight.sh" "minicpm-ocr-parse-$TAG" 12 || { log "preflight FAILED"; exit 1; }

log "=== parse $INPUT ==="
cd "$BASE"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$VENV/bin/minicpmvocr" parse "$INPUT" \
    --config "$BASE/configs/spark.yaml" \
    --server-url http://127.0.0.1:18499 \
    --model "$MODEL" \
    --device cuda:0 \
    --output "$OUT/result.md" \
    --run-tag "$TAG" 2>&1 | grep -v -E 'FutureWarning|pynvml|it/s\]' | tail -40

log "=== result ==="
ls -la "$OUT"
echo "-------- result.md --------"
cat "$OUT/result.md" 2>/dev/null
echo "-------- end --------"
log "=== done ==="
