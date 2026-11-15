#!/usr/bin/env bash
# A/B 对照：使用同一数据、seed 和记录的超参，比较不同 loss。
#   rlcd    = notebook 配方（GRPO 策略梯度项 + soft CE）
#   soft-ce = 纯蒸馏（soft cross-entropy 单独）
# 串行执行（两臂同时跑会抢 GPU 并引入系统噪声，破坏对照公平性）。
#
# 用法（激活独立环境后，在实验目录下）：
#   bash scripts/run_ab_compare.sh
# 产物：runs/rlcd/ 与 runs/soft-ce/ 各自的 checkpoint + train_report.json，
#       最后打印并排对比表。
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
export HF_HUB_DISABLE_XET=1

PY="${PYTHON:-python}"
DATA_DIR="${XL_DATA_DIR:-data}"
OUT_DIR="${XL_OUT_DIR:-runs}"
DATA="$DATA_DIR/typed_decisions_train.jsonl"
EVAL="$DATA_DIR/typed_decisions_test.jsonl"
MODEL_ARGS=()
if [ -n "${MODEL_DIR:-}" ]; then
  MODEL_ARGS=(--model-dir "$MODEL_DIR")
fi
if [ -n "${MODEL_REVISION:-}" ]; then
  MODEL_ARGS+=(--model-revision "$MODEL_REVISION")
fi

mkdir -p "$OUT_DIR"
for ARM in rlcd soft-ce; do
  if [ -e "$OUT_DIR/$ARM" ] || [ -e "$OUT_DIR/${ARM}.log" ]; then
    echo "$OUT_DIR/$ARM or its log already exists; use a fresh experiment directory" >&2
    exit 1
  fi
done
MEM_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
MIN_KB=$(( ${MIN_AVAILABLE_GIB:-50} * 1024 * 1024 ))
if [ -z "$MEM_KB" ] || [ "$MEM_KB" -lt "$MIN_KB" ]; then
  echo "Insufficient available system memory for configured threshold" >&2
  exit 1
fi

for ARM in rlcd soft-ce; do
  LOSS="$ARM"
  echo "=== arm: $ARM (loss=$LOSS) ==="
  "$PY" scripts/run_finetune.py \
    --data "$DATA" --eval-data "$EVAL" "${MODEL_ARGS[@]}" \
    --output-dir "$OUT_DIR/$ARM" \
    --loss "$LOSS" --epochs 4 --seed 0 --device cuda \
    > "$OUT_DIR/${ARM}.log" 2>&1 || { echo "arm $ARM FAILED, see $OUT_DIR/${ARM}.log"; exit 1; }
done

echo
echo "=== A/B 对比（held-out 2000 题，训练配置差异为 loss）==="
"$PY" - <<'EOF'
import json

def report(arm):
    # train_report.json 是扁平结构：eval_mode/before/after 在顶层，无 train_report 外层键
    import os
    with open(os.path.join(os.environ.get("XL_OUT_DIR", "runs"), arm, "train_report.json")) as f:
        return json.load(f)

r, s = report("rlcd"), report("soft-ce")
keys = [("accuracy", "Accuracy"), ("ece", "ECE"), ("brier", "Brier"),
        ("brier_top1", "Brier top-1"), ("mean_confidence", "Mean conf")]
print(f"{'metric':<16}{'rlcd':>12}{'soft-ce':>12}{'delta':>12}")
for k, name in keys:
    a, b = r["after"].get(k), s["after"].get(k)
    d = (a - b) if (a is not None and b is not None) else None
    print(f"{name:<16}{a:>12.4f}{b:>12.4f}{d:>+12.4f}")
print(f"{'loss (final ep)':<16}{r['after']['loss']:>12.4f}{s['after']['loss']:>12.4f}")
print("\nby type (accuracy):")
for t in ("choice", "score", "noul"):
    ra = r["after"]["by_type"][t]["accuracy"]
    sa = s["after"]["by_type"][t]["accuracy"]
    print(f"  {t:<10}{ra:>12.4f}{sa:>12.4f}{ra - sa:>+12.4f}")
EOF
