#!/bin/bash
# 投机解码矩阵批量执行器（DGX 侧运行）v2
# 用法: bash run-matrix.sh <tag> <模型目录> <模型文件> <ctx> <配置文件>
# 配置文件每行: 标签|额外参数
#
# v2 修正：v1 用 /health 判就绪，实测发现 /health 在模型仍在加载时就返回 200，
# 导致测试请求打进未就绪的服务、大量 503。改为检测日志中的 "listening on http"。
set -u

TAG=$1; DIR=$2; MODEL=$3; CTX=$4; CFG=$5
OUT=$HOME/matrix_${TAG}.txt
PORT=8835
: > "$OUT"

bench_all() {
  local label=$1
  local task r
  for task in "T1-自由生成" "T2-长文本改写" "T3-代码改写"; do
    for r in 1 2 3; do
      echo "### ${TAG}|${label}|${task}|round${r}" >> "$OUT"
      python3 "$HOME/bench.py" --host 127.0.0.1 --port "$PORT" --mode completion \
        --n-predict 400 --ignore-eos --prompt-file "$HOME/tasks/${task}.txt" >> "$OUT" 2>&1
    done
  done
  echo "### ${TAG}|${label}|END" >> "$OUT"
}

while IFS='|' read -r label args; do
  [ -z "$label" ] && continue
  echo "===== ${TAG} / ${label} =====" >> "$OUT"
  pkill -f 'llama-serve[r]'
  sleep 10
  cd "$DIR" || exit 1
  LOG=$HOME/mx_${TAG}_${label}.log
  : > "$LOG"
  setsid nohup "$HOME/llama.cpp/build-cuv2/bin/llama-server" \
    -m "$MODEL" -ngl 99 -c "$CTX" -np 1 -t 8 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 0.0.0.0 --port "$PORT" --jinja $args \
    >> "$LOG" 2>&1 < /dev/null &

  ok=0
  for i in $(seq 1 150); do
    if grep -q "listening on http" "$LOG" 2>/dev/null; then ok=1; break; fi
    sleep 3
  done
  echo "### ${TAG}|${label}|READY=${ok}" >> "$OUT"

  # 暖机请求：吞掉首请求的初始化开销，结果不计入统计
  curl -s -m 120 -X POST "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"warmup","n_predict":4}' > /dev/null 2>&1
  sleep 2
  bench_all "$label"
done < "$CFG"

pkill -f 'llama-serve[r]'
echo "ALL_DONE ${TAG}" >> "$OUT"
