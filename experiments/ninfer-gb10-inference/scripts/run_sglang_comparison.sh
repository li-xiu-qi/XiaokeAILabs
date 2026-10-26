#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
: "${SGLANG_MODEL_ROOT:?set SGLANG_MODEL_ROOT to the Coder390 root}"
: "${SGLANG_PYTHON:?set SGLANG_PYTHON to the env with SGLang installed}"
export SGLANG_RUN_DIR=${SGLANG_RUN_DIR:-$ROOT}
mkdir -p "$SGLANG_RUN_DIR/artifacts"

OWNS_SERVER=0
cleanup() {
    if [[ "$OWNS_SERVER" == 1 ]]; then
        bash scripts/dgx_sglang.sh stop >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

run_case() {
    local variant=$1
    local graph=$2
    local output=$3
    local model=$4
    echo "=== variant=$variant graph=$graph ==="
    DECODE_GRAPH="$graph" bash scripts/dgx_sglang.sh start "$variant" dflash2
    OWNS_SERVER=1
    if [[ "$graph" == disabled ]]; then
      "$SGLANG_PYTHON" scripts/ninfer_openai_smoke.py \
        --base-url http://127.0.0.1:19410/v1 \
        --model "$model" --max-tokens 1536 \
        --output "$SGLANG_RUN_DIR/artifacts/smoke-${variant}-dflash2.json"
      "$SGLANG_PYTHON" scripts/quality_probe.py \
        --base-url http://127.0.0.1:19410/v1 \
        --model "$model" \
        --output "$SGLANG_RUN_DIR/artifacts/quality-${variant}-dflash2.json"
      "$SGLANG_PYTHON" scripts/verify_code_functional.py \
        --quality "$SGLANG_RUN_DIR/artifacts/quality-${variant}-dflash2.json" \
        --output "$SGLANG_RUN_DIR/artifacts/code-functional-${variant}-dflash2.json"
    fi
    "$SGLANG_PYTHON" scripts/bench_ninfer_concurrency.py \
      --base-url http://127.0.0.1:19410/v1 \
      --engine "SGLang 0.5.19 $graph" \
      --context-total 65536 \
      --enable-thinking --reasoning-effort xhigh \
      --return-spec-tokens-details \
      --model "$model" \
      --levels 1,2,4,8 --rounds 2 --max-tokens 512 \
      --output "$output"
    bash scripts/dgx_sglang.sh stop
    OWNS_SERVER=0
}

for graph in disabled full; do
    for variant in nvfp4 fp8; do
        if [[ "$variant" == nvfp4 ]]; then
            model=qwen3.8-27b-coder390-nvfp4-w4a4-dflash2
        else
            model=qwen3.8-27b-coder390-fp8-w8a8-dflash2
        fi
        if [[ "$graph" == full ]]; then
            suffix=-graph
            comparison=comparison-graph.md
        else
            suffix=
            comparison=comparison-eager.md
        fi
        run_case "$variant" "$graph" \
          "$SGLANG_RUN_DIR/artifacts/bench-${variant}-dflash2${suffix}.json" "$model"
    done
    args=(
      --nvfp4-bench "$SGLANG_RUN_DIR/artifacts/bench-nvfp4-dflash2${suffix}.json"
      --fp8-bench "$SGLANG_RUN_DIR/artifacts/bench-fp8-dflash2${suffix}.json"
      --output "$SGLANG_RUN_DIR/artifacts/$comparison"
    )
    if [[ "$graph" == disabled ]]; then
      [[ -f "$SGLANG_RUN_DIR/artifacts/quality-nvfp4-dflash2.json" ]] && \
        args+=(--nvfp4-quality "$SGLANG_RUN_DIR/artifacts/quality-nvfp4-dflash2.json")
      [[ -f "$SGLANG_RUN_DIR/artifacts/quality-fp8-dflash2.json" ]] && \
        args+=(--fp8-quality "$SGLANG_RUN_DIR/artifacts/quality-fp8-dflash2.json")
      [[ -f "$SGLANG_RUN_DIR/artifacts/smoke-nvfp4-dflash2.json" ]] && \
        args+=(--nvfp4-smoke "$SGLANG_RUN_DIR/artifacts/smoke-nvfp4-dflash2.json")
      [[ -f "$SGLANG_RUN_DIR/artifacts/smoke-fp8-dflash2.json" ]] && \
        args+=(--fp8-smoke "$SGLANG_RUN_DIR/artifacts/smoke-fp8-dflash2.json")
    fi
    "$SGLANG_PYTHON" scripts/compare_results.py "${args[@]}"
done
