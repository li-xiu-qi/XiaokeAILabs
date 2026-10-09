#!/usr/bin/env bash
# A/B 对照：同一数据、同一 seed、同一一切超参，唯一变量是 loss。
#   rlcd    = notebook 配方（GRPO 策略梯度项 + soft CE）
#   soft-ce = 纯蒸馏（soft cross-entropy 单独）
# 串行执行（两臂同时跑会抢 GPU 并引入系统噪声，破坏对照公平性）。
#
# 用法（laya-rlcd-env，实验目录下）：
#   bash scripts/run_ab_compare.sh
# 产物：runs/rlcd/ 与 runs/softce/ 各自的 checkpoint + train_report.json，
#       最后打印并排对比表。
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
export HF_HUB_DISABLE_XET=1

PY=/home/ke/laya-rlcd-env/bin/python
DATA=data/typed_decisions_train.jsonl
EVAL=data/typed_decisions_test.jsonl

bash ~/preflight.sh laya-ab-compare 50 || exit 1

for ARM in rlcd soft-ce; do
  LOSS="$ARM"
  echo "=== arm: $ARM (loss=$LOSS) ==="
  $PY scripts/run_finetune.py \
    --data "$DATA" --eval-data "$EVAL" \
    --output-dir "runs/$ARM" \
    --loss "$LOSS" --epochs 4 --seed 0 --device cuda \
    > "runs/${ARM}.log" 2>&1 || { echo "arm $ARM FAILED, see runs/${ARM}.log"; exit 1; }
done

echo
echo "=== A/B 对比（held-out 2000 题，唯一变量 loss）==="
$PY - <<'EOF'
import json

def report(arm):
    # train_report.json 是扁平结构：eval_mode/before/after 在顶层，无 train_report 外层键
    with open(f"runs/{arm}/train_report.json") as f:
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
