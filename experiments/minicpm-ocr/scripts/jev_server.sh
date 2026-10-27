#!/usr/bin/env bash
# MiniCPM-OCR-Jev 打分服务：启停 + 健康检查。
#
#   bash jev_server.sh start     # 起服务（约 35 秒加载权重），等 /healthz 就绪
#   bash jev_server.sh stop
#   bash jev_server.sh status
#
# 端口固定 18501。渲染器在独立 venv（venv-render）里跑，打分在 xinfer-env 里跑。
set -uo pipefail
BASE="$HOME/minicpm-ocr"
PORT="${PORT:-18501}"
JEV="$BASE/jev"
LOG="$BASE/logs/jev_server.log"
PY="$HOME/xinfer-env/bin/python"

pid_on_port() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1; }

case "${1:-status}" in
  start)
    if [ -n "$(pid_on_port)" ]; then
      echo "已在运行 pid=$(pid_on_port)"; exit 0
    fi
    ( cd "$JEV" && setsid nohup "$PY" jev_server.py --port "$PORT" \
        > "$LOG" 2>&1 < /dev/null & )
    for i in $(seq 1 60); do
      sleep 2
      if curl -sf -m 3 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
        echo "就绪 pid=$(pid_on_port)"; curl -s "http://127.0.0.1:$PORT/healthz"; echo; exit 0
      fi
    done
    echo "启动超时，日志尾部："; tail -15 "$LOG"; exit 1
    ;;
  stop)
    PID="$(pid_on_port)"
    if [ -z "$PID" ]; then echo "未运行"; exit 0; fi
    kill "$PID" && echo "已停 pid=$PID"
    ;;
  status)
    PID="$(pid_on_port)"
    if [ -z "$PID" ]; then echo "未运行"; exit 1; fi
    echo "运行中 pid=$PID"; curl -s "http://127.0.0.1:$PORT/healthz"; echo
    ;;
  *) echo "用法: $0 {start|stop|status}"; exit 2 ;;
esac
