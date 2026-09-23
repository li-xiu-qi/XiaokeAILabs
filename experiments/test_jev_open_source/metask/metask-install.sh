#!/bin/bash
# metask-jev 一键安装脚本
# 建 venv、装锁定版本的依赖、下载模型、校验 A-Z 单 token 契约，
# 然后跑一个自测打分例子。
#
# 用法：curl -fsSL https://raw.githubusercontent.com/metask-ai/metask-jev/main/install.sh | bash
set -euo pipefail

MODEL_ID="wayfind/metask-jev-4b-policy-mix"
DIR="${HOME}/metask-jev"
REPO="https://github.com/metask-ai/metask-jev"

say() { printf "\n\033[1;31m[metask-jev]\033[0m %s\n" "$*"; }

# ---- 0. 探测加速器 ----
GPU_HINT="CPU (slow but works)"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_HINT="NVIDIA GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
elif [ "$(uname)" = "Darwin" ]; then
  GPU_HINT="Apple Silicon (MPS)"
fi
say "accelerator: $GPU_HINT"

# ---- 1. python ----
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  say "ERROR: python3 未找到"; exit 1
fi
say "python: $($PY --version)"

# ---- 2. venv ----
mkdir -p "$DIR" && cd "$DIR"
if [ ! -d ".venv" ]; then
  say "在 $DIR/.venv 创建虚拟环境"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 3. deps ----
say "安装 torch + transformers（可能要几分钟）"
if [ "$(uname)" = "Darwin" ]; then
  pip install -q --upgrade pip
  pip install -q "torch>=2.4" "transformers>=4.53" "accelerate" "huggingface_hub"
else
  pip install -q --upgrade pip
  pip install -q "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121 \
    || pip install -q "torch>=2.4"
  pip install -q "transformers>=4.53" "accelerate" "huggingface_hub"
fi

# ---- 4. 推理代码 ----
if [ ! -f "jev_scorer.py" ]; then
  say "从 $REPO 取推理代码"
  for f in jev_scorer.py jev_schema.py; do
    curl -fsSL "$REPO/raw/main/inference/$f" -o "$f"
  done
fi

# ---- 5. 模型下载（经 huggingface_hub snapshot）----
say "下载 $MODEL_ID（约 8.5 GB，可续传）"
python - "$MODEL_ID" << 'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1])
print("model at:", p)
with open("model_path.txt", "w") as f:
    f.write(p)
EOF
MODEL_PATH=$(cat model_path.txt)

# ---- 6. 校验 A-Z 单 token 契约 + 自测 ----
say "校验 tokenizer 契约并跑自测"
python - "$MODEL_PATH" << 'EOF'
import sys
from jev_scorer import load_model, score

model_path = sys.argv[1]
model, tok, dev = load_model(model_path)

# A-Z 每个字母都必须是单 token（候选 logit 契约）
bad = [chr(65+i) for i in range(26) if len(tok.encode(chr(65+i), add_special_tokens=False)) != 1]
assert not bad, f"tokenizer contract broken for: {bad}"
print(f"[ok] A-Z single-token contract verified on {dev}")

state = ("The store accepts returns within 30 days of purchase. "
         "This item was bought 12 days ago and is unopened.")
schema = {"decision": {
    "description": "Is the item still eligible for return?",
    "type": "boolean", "choices": [False, True],
    "choice_descriptions": {"false": "Not eligible.", "true": "Eligible."},
}}
r = score(model, tok, state, schema, temperature=2.25)  # noul temperature
print(f"[ok] self-test prediction: {r['prediction']}  probs: "
      + ", ".join(f"{k}={v:.3f}" for k, v in sorted(r['probabilities'].items(), key=lambda x: -x[1])))
EOF

say "安装完成"
cat << TIP

  Quick use:
    source $DIR/.venv/bin/activate
    cd $DIR
    python - << 'PY'
from jev_scorer import load_model, score
model, tok, dev = load_model("$MODEL_PATH")
state = "Your state text here."
schema = {"decision": {"description": "Your question?",
    "type": "boolean", "choices": [False, True],
    "choice_descriptions": {"false": "No.", "true": "Yes."}}}
print(score(model, tok, state, schema, temperature=2.25))
PY

  Temperatures: choice 1.7875 / noul 2.25 / score 2.05 (see model card).
TIP