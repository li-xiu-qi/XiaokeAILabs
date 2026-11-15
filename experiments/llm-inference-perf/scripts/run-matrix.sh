#!/usr/bin/env bash
# 运行单模型配置矩阵。运行数据默认写入 Git 忽略的 .work/。
# 用法：bash scripts/run-matrix.sh <tag> <模型目录> <模型文件> <ctx> <配置文件>
# 配置每行：标签|空格分隔的参数；支持参数中的 $HOME 或 ${HOME}。
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "用法: bash scripts/run-matrix.sh <tag> <模型目录> <模型文件> <ctx> <配置文件>" >&2
  exit 2
fi
TAG=$1; MODEL_DIR=$2; MODEL=$3; CTX=$4; CFG=$5
[[ "$TAG" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "非法实验标签: $TAG" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-cuv2/bin/llama-server}"
PORT="${PORT:-8835}"
if [ "${XL_REQUIRE_EXTERNAL_ENV:-0}" = 1 ] && [ -z "${XL_OUT_DIR:-}" ]; then
  echo "严格模式需要显式设置 XL_OUT_DIR" >&2
  exit 2
fi
OUT_DIR="${XL_OUT_DIR:-$EXP_DIR/.work/matrix-$TAG}"
[ -d "$MODEL_DIR" ] || { echo "模型目录不存在: $MODEL_DIR" >&2; exit 2; }
[ -f "$CFG" ] || { echo "配置文件不存在: $CFG" >&2; exit 2; }
[ -x "$LLAMA_BIN" ] || { echo "服务程序不可执行: $LLAMA_BIN" >&2; exit 2; }
LLAMA_BIN="$(cd "$(dirname "$LLAMA_BIN")" && pwd)/$(basename "$LLAMA_BIN")"
if [[ "$PYTHON" == */* ]]; then
  [ -x "$PYTHON" ] || { echo "Python 不可执行: $PYTHON" >&2; exit 2; }
  PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"
else
  command -v "$PYTHON" >/dev/null || { echo "Python 不可用: $PYTHON" >&2; exit 2; }
fi
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
# Reject an already occupied port before starting; REUSEADDR allows an old
# completed connection in TIME_WAIT without permitting an active listener.
port_available() {
  "$PYTHON" -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$PORT"
}
port_available || { echo "端口不可用: $PORT" >&2; exit 2; }
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
OUT="$OUT_DIR/matrix_${TAG}.txt"
(set -C; : > "$OUT") 2>/dev/null || { echo "输出已存在，避免覆盖: $OUT" >&2; exit 2; }
SERVER_PID=
stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || :
    wait "$SERVER_PID" 2>/dev/null || :
    SERVER_PID=
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$MODEL_DIR"
while IFS='|' read -r label args || [ -n "${label:-}" ]; do
  [ -n "$label" ] || continue
  [[ "$label" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "非法配置标签: $label" >&2; exit 2; }
  read -r -a extra <<< "${args:-}"
  for i in "${!extra[@]}"; do
    extra[$i]="${extra[$i]//\$\{HOME\}/$HOME}"
    extra[$i]="${extra[$i]//\$HOME/$HOME}"
  done
  LOG="$OUT_DIR/server_${TAG}_${label}.log"
  [ ! -e "$LOG" ] || { echo "日志已存在，避免覆盖: $LOG" >&2; exit 2; }
  port_available || { echo "端口不可用: $PORT" >&2; exit 2; }
  echo "===== ${TAG} / ${label} =====" >> "$OUT"
  "$LLAMA_BIN" -m "$MODEL" -ngl 99 -c "$CTX" -np 1 -t 8 \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port "$PORT" \
    --jinja "${extra[@]}" > "$LOG" 2>&1 &
  SERVER_PID=$!
  ready=0
  for ((i=0; i<150; i++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    if grep -q "listening on http" "$LOG" &&
       curl -fsS -m 2 "http://127.0.0.1:${PORT}/health" > /dev/null &&
       kill -0 "$SERVER_PID" 2>/dev/null; then
      ready=1; break
    fi
    sleep 3
  done
  echo "### ${TAG}|${label}|READY=${ready}" >> "$OUT"
  [ "$ready" -eq 1 ] || { echo "服务未就绪，检查 $LOG" >&2; exit 1; }
  curl -fsS -m 120 -X POST "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' -d '{"prompt":"warmup","n_predict":4}' > /dev/null
  for task in "T1-自由生成" "T2-长文本改写" "T3-代码改写"; do
    for round in 1 2 3; do
      kill -0 "$SERVER_PID" 2>/dev/null || { echo "服务在测试期间退出" >&2; exit 1; }
      echo "### ${TAG}|${label}|${task}|round${round}" >> "$OUT"
      "$PYTHON" "$SCRIPT_DIR/bench.py" --host 127.0.0.1 --port "$PORT" --mode completion \
        --n-predict 400 --ignore-eos --prompt-file "$EXP_DIR/tasks/${task}.txt" >> "$OUT" 2>&1
    done
  done
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "服务在测试期间退出" >&2; exit 1; }
  echo "### ${TAG}|${label}|END" >> "$OUT"
  stop_server
done < "$CFG"
echo "ALL_DONE ${TAG}" >> "$OUT"
echo "结果: $OUT"
