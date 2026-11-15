#!/usr/bin/env bash
# run.sh — 全链路一键入口（运行契约版）
#
# 运行契约（与 llm_common 一致）：
#   支持通过 XL_OUT_DIR / XL_DATA_DIR / XL_RUN_ID 指定输出目录、数据目录与运行标识；
#   未显式指定时默认写到实验目录下被 .gitignore 忽略的 .work/out/<run_id> 与 .work/data；
#   设置 XL_REQUIRE_EXTERNAL_ENV=1 时禁止本地回退并要求显式注入三个变量。
#
# 用法：
#   bash run.sh --config configs/smoke.json     # 冒烟档（CPU 最小步数，约 2-3 分钟）
#   bash run.sh --config configs/full.json      # 完整档（真实规模）
#   PYTHON=/path/to/python bash run.sh --config configs/smoke.json
#   NPROC=2 bash run.sh --config configs/smoke.json            # 单机双进程 DDP
#   bash run.sh --config configs/smoke.json --with-tok-eval     # 末尾追加分词器多域评测
#
# profile 定义在 configs/<name>.json：device / amp / scale / nproc /
# tokenizer.vocab_size / tokenizer.max_chars / stages.{pretrain,midtrain,sft,rl}.steps
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="configs/smoke.json"
WITH_TOK_EVAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --with-tok-eval) WITH_TOK_EVAL=1; shift ;;
    *) echo "[run] 未知参数: $1" >&2; exit 2 ;;
  esac
done

if [ -f "$CONFIG" ]; then
  CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
elif [ -f "$SCRIPT_DIR/$CONFIG" ]; then
  CONFIG="$SCRIPT_DIR/$CONFIG"
else
  echo "[run] 配置文件不存在: $CONFIG" >&2
  exit 2
fi

PY="${PYTHON:-python3}"
cd "$SCRIPT_DIR"
WORK_ROOT="$PWD/.work"

REQ_EXT="$(printf '%s' "${XL_REQUIRE_EXTERNAL_ENV:-}" | tr '[:upper:]' '[:lower:]')"
if [[ "$REQ_EXT" == "1" || "$REQ_EXT" == "true" || "$REQ_EXT" == "yes" ]]; then
  : "${XL_OUT_DIR:?XL_OUT_DIR 未设置（XL_REQUIRE_EXTERNAL_ENV=1 禁止本地回退）}"
  : "${XL_DATA_DIR:?XL_DATA_DIR 未设置（XL_REQUIRE_EXTERNAL_ENV=1 禁止本地回退）}"
  : "${XL_RUN_ID:?XL_RUN_ID 未设置（XL_REQUIRE_EXTERNAL_ENV=1 禁止本地回退）}"
else
  export XL_RUN_ID="${XL_RUN_ID:-local-$(date +%Y%m%d-%H%M%S)}"
  export XL_OUT_DIR="${XL_OUT_DIR:-$WORK_ROOT/out/$XL_RUN_ID}"
  export XL_DATA_DIR="${XL_DATA_DIR:-$WORK_ROOT/data}"
fi

# 从 profile 读参数（JSON -> shell 变量）
eval "$("$PY" - "$CONFIG" <<'PYEOF'
import json, shlex, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
st = c.get("stages", {})
tk = c.get("tokenizer", {})
print(f"PROFILE={shlex.quote(str(c.get('name', 'custom')))}")
print(f"DEVICE={shlex.quote(str(c.get('device', 'auto')))}")
print(f"AMP={shlex.quote(str(c.get('amp', 'none')))}")
print(f"SCALE={shlex.quote(str(c.get('scale', 1.0)))}")
print(f"PROFILE_NPROC={shlex.quote(str(c.get('nproc', 1)))}")
print(f"VOCAB={shlex.quote(str(tk.get('vocab_size', 2048)))}")
print(f"TOK_CHARS={shlex.quote(str(tk.get('max_chars', 2000000)))}")
print(f"PRETRAIN_STEPS={shlex.quote(str(st.get('pretrain', 60)))}")
print(f"MIDTRAIN_STEPS={shlex.quote(str(st.get('midtrain', 40)))}")
print(f"SFT_STEPS={shlex.quote(str(st.get('sft', 80)))}")
print(f"RL_STEPS={shlex.quote(str(st.get('rl', 20)))}")
PYEOF
)"

NPROC="${NPROC:-$PROFILE_NPROC}"
if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[run] nproc 必须为正整数: $NPROC" >&2
  exit 2
fi

# 02~06 在 NPROC>1 时经 torchrun 起 DDP；01/07/08 固定单进程
run_train() {
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --nproc_per_node="$NPROC" "$@"
  else
    "$PY" "$@"
  fi
}

echo "[run] profile=$PROFILE config=$CONFIG device=$DEVICE amp=$AMP scale=$SCALE nproc=$NPROC"
echo "[run] XL_OUT_DIR=$XL_OUT_DIR XL_DATA_DIR=$XL_DATA_DIR XL_RUN_ID=$XL_RUN_ID"

# 前置：缺语料时生成离线自测语料（只写 XL_DATA_DIR）
if [ ! -f "$XL_DATA_DIR/corpus.txt" ]; then
  echo "[run] [前置] 生成离线自测语料 -> $XL_DATA_DIR"
  "$PY" scripts/prepare_data.py --source toy
fi

echo "[run] [1/8] 训练 BPE 分词器（vocab=$VOCAB, max_chars=$TOK_CHARS）"
"$PY" scripts/01_train_tokenizer.py --vocab-size "$VOCAB" --max-chars "$TOK_CHARS"

echo "[run] [2/8] 模型自检（QK norm / RoPE / KV cache / softcap / GQA / AMP / DDP）"
run_train scripts/02_model.py --device "$DEVICE" --amp "$AMP"

echo "[run] [3/8] 预训练：Muon+AdamW，附 AdamW 基线（steps=$PRETRAIN_STEPS x2）"
run_train scripts/03_pretrain.py --steps "$PRETRAIN_STEPS" --device "$DEVICE" --amp "$AMP" --scale "$SCALE"

echo "[run] [4/8] Midtrain：对话格式适配（steps=$MIDTRAIN_STEPS）"
run_train scripts/04_midtrain.py --steps "$MIDTRAIN_STEPS" --device "$DEVICE" --amp "$AMP" --scale "$SCALE"

echo "[run] [5/8] 指令微调 SFT（steps=$SFT_STEPS）"
run_train scripts/05_sft.py --steps "$SFT_STEPS" --device "$DEVICE" --amp "$AMP" --scale "$SCALE"

echo "[run] [6/8] RL：GRPO 可验证奖励（steps=$RL_STEPS）"
run_train scripts/06_rl_grpo.py --steps "$RL_STEPS" --device "$DEVICE" --amp "$AMP" --scale "$SCALE"

echo "[run] [7/8] 生成（贪心 + 温度采样 + KV cache 提速对比）"
"$PY" scripts/07_generate.py --device "$DEVICE"

echo "[run] [8/8] 全链路评估与 report.md"
"$PY" scripts/08_eval_report.py --device "$DEVICE"

if [[ "$WITH_TOK_EVAL" == "1" ]]; then
  echo "[run] [附加] 分词器多域压缩率评测（fixtures 固定域切片）"
  "$PY" scripts/eval_tokenizer_domains.py \
      --tokenizer-path "$XL_OUT_DIR/tokenizer/tokenizer.json" --tokenizer-name "$PROFILE" \
      --tokenizer-bytes "$XL_OUT_DIR/tokenizer/token_bytes.pt" \
      --out "$XL_OUT_DIR/tokenizer/domain_compression_eval.json"
fi

echo "[run] 记录运行环境清单 -> $XL_OUT_DIR/run_manifest.json"
"$PY" - <<PYEOF
import json, platform, sys
from pathlib import Path
import torch, tokenizers
manifest = {
    "run_id": "$XL_RUN_ID",
    "profile": "$PROFILE",
    "config": "$CONFIG",
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nccl_available": torch.distributed.is_nccl_available(),
    "tokenizers": tokenizers.__version__,
    "torch_threads": torch.get_num_threads(),
    "device": "$DEVICE",
    "amp": "$AMP",
    "scale": "$SCALE",
    "nproc": $NPROC,
    "steps": {"pretrain_per_optimizer": $PRETRAIN_STEPS, "midtrain": $MIDTRAIN_STEPS,
              "sft": $SFT_STEPS, "rl": $RL_STEPS},
}
try:
    import matplotlib
    manifest["matplotlib"] = matplotlib.__version__
except Exception:
    manifest["matplotlib"] = None
out = Path("$XL_OUT_DIR") / "run_manifest.json"
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False))
PYEOF

echo "[run] 完成。产物见 $XL_OUT_DIR ，说明见 README.md"
