#!/usr/bin/env bash
# run_all.sh — 一键复现整条链路（参考 karpathy/nanochat 的全栈流程，CPU 玩具规模）
#   tokenizer -> model 自检 -> pretrain(Muon vs AdamW) -> midtrain -> SFT -> RL(GRPO) -> 生成 -> 评估
# 用法: bash run_all.sh            （默认配置，CPU 约 5.5 分钟）
#       bash run_all.sh --quick    （缩减步数，约 2-3 分钟）
# Python 解释器默认取 PATH 中的 python，可用 PYTHON 环境变量覆盖，例如
#   PYTHON=/path/to/python bash run_all.sh
set -euo pipefail

PY="${PYTHON:-python}"
cd "$(dirname "$0")"

PRETRAIN_STEPS=250
MIDTRAIN_STEPS=120
SFT_STEPS=250
RL_STEPS=80
if [[ "${1:-}" == "--quick" ]]; then
  PRETRAIN_STEPS=60
  MIDTRAIN_STEPS=40
  SFT_STEPS=80
  RL_STEPS=20
fi

echo "==> [1/8] 生成语料 + 训练 BPE 分词器（GPT-4 风格正则预分词）"
"$PY" scripts/01_train_tokenizer.py --force

echo "==> [2/8] 模型自检（QK norm / RoPE / KV cache / softcap / GQA）"
"$PY" scripts/02_model.py

echo "==> [3/8] 预训练：Muon+AdamW，附 AdamW 基线（steps=$PRETRAIN_STEPS x2）"
"$PY" scripts/03_pretrain.py --steps "$PRETRAIN_STEPS"

echo "==> [4/8] Midtrain：对话格式适配（steps=$MIDTRAIN_STEPS）"
"$PY" scripts/04_midtrain.py --steps "$MIDTRAIN_STEPS"

echo "==> [5/8] 指令微调 SFT（steps=$SFT_STEPS）"
"$PY" scripts/05_sft.py --steps "$SFT_STEPS"

echo "==> [6/8] RL：GRPO 可验证奖励（steps=$RL_STEPS）"
"$PY" scripts/06_rl_grpo.py --steps "$RL_STEPS"

echo "==> [7/8] 生成（贪心 + 温度采样 + KV cache 提速对比）"
"$PY" scripts/07_generate.py

echo "==> [8/8] 全链路评估与 report.md"
"$PY" scripts/08_eval_report.py

echo "==> 记录运行环境清单"
"$PY" - <<PYEOF
import json, platform, sys
from pathlib import Path
import torch, tokenizers
manifest = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "tokenizers": tokenizers.__version__,
    "torch_threads": torch.get_num_threads(),
    "steps": {"pretrain_per_optimizer": $PRETRAIN_STEPS, "midtrain": $MIDTRAIN_STEPS,
              "sft": $SFT_STEPS, "rl": $RL_STEPS},
}
try:
    import matplotlib
    manifest["matplotlib"] = matplotlib.__version__
except Exception:
    manifest["matplotlib"] = None
Path("results/run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False))
PYEOF

echo "==> 完成。产物见 results/ ，说明见 README.md 与 results/report.md"
