#!/usr/bin/env bash
# MiniCPM-OCR on dgx-spark: stage 2b - fetch sources via codeload tarballs (git clone over TLS is flaky here)
set -uo pipefail

BASE="$HOME/minicpm-ocr"
VENV="$BASE/.venv-ocr"
TF_COMMIT="549e5b410733fb50e8efbbe4aa585cb89f6478ea"
log() { echo "[$(date '+%F %T')] $*"; }

fetch() {  # fetch <url> <outfile>
    local url="$1" out="$2"
    for attempt in 1 2 3 4 5; do
        log "GET $url (attempt $attempt)"
        if curl -fL --retry 5 --retry-delay 3 -C - -m 1800 -o "$out" "$url"; then return 0; fi
        log "  on disk: $(stat -c%s "$out" 2>/dev/null || echo 0) bytes"
        sleep 5
    done
    return 1
}

log "=== MiniCPM-OCR repo tarball (SDK + configs) ==="
rm -rf "$BASE/repo"; mkdir -p "$BASE/repo"
fetch "https://codeload.github.com/OpenBMB/MiniCPM-OCR/tar.gz/refs/heads/main" "$BASE/src-minicpm-ocr.tar.gz" || exit 1
tar xzf "$BASE/src-minicpm-ocr.tar.gz" -C "$BASE/repo" --strip-components=1 || exit 1
log "repo extracted: $(du -sh "$BASE/repo" | cut -f1)"

log "=== transformers tarball @ $TF_COMMIT ==="
rm -rf "$BASE/sources/transformers"; mkdir -p "$BASE/sources/transformers"
fetch "https://codeload.github.com/huggingface/transformers/tar.gz/$TF_COMMIT" "$BASE/src-transformers.tar.gz" || exit 1
tar xzf "$BASE/src-transformers.tar.gz" -C "$BASE/sources/transformers" --strip-components=1 || exit 1
log "transformers extracted: $(du -sh "$BASE/sources/transformers" | cut -f1)"

log "=== install transformers from source (--no-deps) ==="
"$VENV/bin/python" -m pip install -q --no-deps -e "$BASE/sources/transformers" || exit 1

log "=== serving deps (transformers serve) ==="
"$VENV/bin/python" -m pip install -q --no-deps pydantic uvicorn fastapi starlette rich openai || exit 1

log "=== install SDK (no paddle; torch layout provider) ==="
"$VENV/bin/python" -m pip install -q --no-deps -e "$BASE/repo/sdk" || exit 1
"$VENV/bin/python" -m pip install -q pymupdf beautifulsoup4 pyyaml || exit 1

log "=== verify ==="
"$VENV/bin/python" - <<'PY' || exit 1
import torch, transformers
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformers", transformers.__version__)
import transformers.models.minicpm_ocr.modeling_minicpm_ocr as m
print("minicpm_ocr modeling OK:", m.MiniCPMOcrForConditionalGeneration.__name__)
import minicpmvocr
print("minicpmvocr import OK")
import fitz
print("PyMuPDF OK")
PY
log "=== stage 2b done ==="
