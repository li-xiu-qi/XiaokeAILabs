#!/usr/bin/env bash
# MiniCPM-OCR on dgx-spark: stage 1 - download weights + layout model + test docs
# Source: ModelScope (hf-mirror only proxies non-LFS files now, huggingface.co is blocked on this box)
set -uo pipefail

BASE="$HOME/minicpm-ocr"
MODELS="$BASE/models"
LOG="$BASE/logs"
mkdir -p "$MODELS" "$LOG" "$BASE/sources" "$BASE/testdata"
MS="https://modelscope.cn"

log() { echo "[$(date '+%F %T')] $*"; }

ms_dl() {  # ms_dl <model_id> <file> <destdir>
    local model="$1" file="$2" dest="$3"
    local url="$MS/models/$model/resolve/master/$file"
    mkdir -p "$dest/$(dirname "$file")"
    for attempt in 1 2 3 4 5; do
        log "GET $model/$file (attempt $attempt)"
        if curl -fL --retry 3 --retry-delay 3 -C - -m 3600 -o "$dest/$file" "$url"; then
            return 0
        fi
        log "  retry after failure: $(ls -l "$dest/$file" 2>/dev/null | awk '{print $5}') bytes on disk"
        sleep 5
    done
    log "FAILED $model/$file"
    return 1
}

log "=== main recognition model: OpenBMB/MiniCPM-OCR-1.0 ==="
for f in config.json generation_config.json preprocessor_config.json \
         processor_config.json tokenizer.json tokenizer_config.json chat_template.jinja \
         model.safetensors; do
    ms_dl "OpenBMB/MiniCPM-OCR-1.0" "$f" "$MODELS/MiniCPM-OCR-1.0" || exit 1
done

log "=== layout model: PaddlePaddle/PP-DocLayoutV3_safetensors ==="
for f in config.json preprocessor_config.json inference.yml model.safetensors; do
    ms_dl "PaddlePaddle/PP-DocLayoutV3_safetensors" "$f" "$MODELS/PP-DocLayoutV3" || exit 1
done

log "=== test documents (from GitHub, reachable) ==="
for f in french_eval_contract.jpg french_eval_address.jpg; do
    ms_dl "OpenBMB/MiniCPM-OCR" "assets/minicpm-ocr/$f" "$BASE/testdata" || true
done

log "=== verify sizes ==="
EXPECT_MAIN=2600969816
ACTUAL=$(stat -c%s "$MODELS/MiniCPM-OCR-1.0/model.safetensors" 2>/dev/null || echo 0)
log "main model.safetensors: $ACTUAL / expected $EXPECT_MAIN"
[ "$ACTUAL" = "$EXPECT_MAIN" ] && log "SIZE OK" || log "SIZE MISMATCH"
log "main sha256: $(sha256sum "$MODELS/MiniCPM-OCR-1.0/model.safetensors" | awk '{print $1}')"
log "expected   : 83f4e433b044fd23f9ea7eb3acbc7d59e4fe11a1fdf3c3129b7021d0ab4f8fd4"

find "$MODELS" "$BASE/testdata" -type f -printf '%12s  %p\n' | sort -k2
log "=== stage 1 done ==="
