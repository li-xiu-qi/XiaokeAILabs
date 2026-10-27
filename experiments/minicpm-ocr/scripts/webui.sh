#!/usr/bin/env bash
# MiniCPM-OCR WebUI（SDK 自带，纯标准库 HTTP 服务，端口 8950）
# 上传 PDF/图片 -> 布局 + 识别 -> 页面内直接看 Markdown / HTML / LaTeX 渲染结果
set -uo pipefail
BASE="$HOME/minicpm-ocr"
VENV="$BASE/.venv-ocr"
SDK="$BASE/repo/sdk"
PORT="${PORT:-8950}"
DATA="$BASE/webui-data"
ACTION="${1:-status}"

case "$ACTION" in
  start)
    if curl -fsS -m 5 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
        echo "already running on $PORT"; exit 0
    fi
    mkdir -p "$DATA" "$BASE/logs"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    cd "$SDK/webui"
    OCRWEB_CONFIG="$BASE/configs/spark.yaml" \
    OCRWEB_DATA_DIR="$DATA" \
    VLLM_URL="http://127.0.0.1:18499" \
    OCR_SERVED_NAME="$BASE/models/MiniCPM-OCR-1.0" \
    OCR_LAYOUT_MODEL_DIR="$BASE/models/PP-DocLayoutV3" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$SDK" PYTHONNOUSERSITE=1 \
    PORT="$PORT" \
    setsid nohup "$VENV/bin/python" -s server.py "ocrweb-server $PORT" \
        > "$BASE/logs/webui.log" 2>&1 < /dev/null &
    echo "launched pid $!, log: $BASE/logs/webui.log"
    ;;
  stop)
    for pid in $(pgrep -f "ocrweb-server $PORT" || true); do
        [ "$pid" = "$$" ] && continue
        kill "$pid" 2>/dev/null && echo "killed $pid"
    done
    ;;
  status)
    curl -fsS -m 5 "http://127.0.0.1:$PORT/api/health" && echo || echo "not running"
    pgrep -af "ocrweb-server" | grep -v pgrep || true
    ;;
esac
