#!/usr/bin/env bash
# MiniCPM-OCR 1.0 on dgx-spark —— 一键启动 / 停止 / 状态
# 部署根目录 ~/minicpm-ocr，推理服务走 transformers serve（OpenAI 兼容，端口 18499）
set -uo pipefail
BASE="$HOME/minicpm-ocr"
VENV="$BASE/.venv-ocr"
MODEL="$BASE/models/MiniCPM-OCR-1.0"
PORT=18499
ACTION="${1:-status}"

case "$ACTION" in
  start)
    if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "Port $PORT responds, but service ownership cannot be verified; inspect it before starting." >&2; exit 1
    fi
    mkdir -p "$BASE/logs"
    cd "$BASE"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid nohup "$VENV/bin/transformers" serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --device cuda --dtype bfloat16 \
        --chat-template-kwargs '{"enable_thinking": false}' \
        > "$BASE/logs/serve.log" 2>&1 < /dev/null &
    echo "launched pid $!, log: $BASE/logs/serve.log"
    ;;
  stop)
    echo "Automatic stop is unavailable: this script cannot verify service ownership." >&2
    echo "Inspect the service process and stop the intended process manually." >&2
    exit 1
    ;;
  status)
    curl -fsS -m 5 "http://127.0.0.1:$PORT/health" && echo || echo "not running"
    pgrep -af 'transformers serve' | grep -v pgrep || true
    ;;
  parse)
    shift
    INPUT="$1"; shift
    cd "$BASE"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$VENV/bin/minicpmvocr" parse "$INPUT" \
        --config "$BASE/configs/spark.yaml" \
        --server-url "http://127.0.0.1:$PORT" \
        --model "$MODEL" \
        --device cuda:0 "$@"
    ;;
esac
