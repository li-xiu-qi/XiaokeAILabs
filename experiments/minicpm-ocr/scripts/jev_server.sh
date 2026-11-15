#!/usr/bin/env bash
# MiniCPM-OCR-Jev 打分服务：启停 + 健康检查。
#
#   bash jev_server.sh start     # 起服务（约 35 秒加载权重），等 /healthz 就绪
#   stop is disabled until service ownership can be verified
#   bash jev_server.sh status
#
# 端口固定 18501。渲染器和打分器分别使用已准备好的 Python 环境。
set -uo pipefail
BASE="$HOME/minicpm-ocr"
PORT="${PORT:-18501}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$BASE/logs/jev_server.log"
PY="${JEV_PYTHON:-$HOME/xinfer-env/bin/python}"

pid_on_port() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1; }

case "${1:-status}" in
  start)
    if [ -n "$(pid_on_port)" ]; then
      echo "Port $PORT is in use; service ownership cannot be verified." >&2; exit 1
    fi
    if [ ! -x "$PY" ]; then
      echo "缺少打分环境: $PY；请先准备依赖并用 JEV_PYTHON 指定解释器" >&2; exit 1
    fi
    mkdir -p "$(dirname "$LOG")"
    setsid nohup "$PY" "$SCRIPT_DIR/jev_server.py" --port "$PORT" \
        > "$LOG" 2>&1 < /dev/null &
    for i in $(seq 1 60); do
      sleep 2
      if curl -sf -m 3 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
        echo "就绪 pid=$(pid_on_port)"; curl -s "http://127.0.0.1:$PORT/healthz"; echo; exit 0
      fi
    done
    echo "启动超时，日志尾部："; tail -15 "$LOG"; exit 1
    ;;
  stop)
    echo "Automatic stop is unavailable: this script cannot verify service ownership." >&2
    echo "Inspect the service process and stop the intended process manually." >&2
    exit 1
    ;;
  status)
    PID="$(pid_on_port)"
    if [ -z "$PID" ]; then echo "未运行"; exit 1; fi
    echo "运行中 pid=$PID"; curl -s "http://127.0.0.1:$PORT/healthz"; echo
    ;;
  *) echo "用法: $0 {start|stop|status}"; exit 2 ;;
esac
