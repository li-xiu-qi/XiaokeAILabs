#!/bin/bash
# metask-jev 一键安装脚本
# 建 venv、安装依赖、下载模型、校验 A-Z 单 token 契约，
# 然后跑一个自测打分例子。
#
# 用法：从本目录运行 bash metask-install.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="wayfind/metask-jev-4b-policy-mix"
MODEL_REVISION="${METASK_MODEL_REVISION:-ea20fe85b28733b1522721dec119bec50947a869}"
DIR="${METASK_DIR:-${HOME}/metask-jev}"
TORCH_REQUIREMENT="${TORCH_REQUIREMENT:-torch>=2.4}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
export HF_HOME="${HF_HOME:-$DIR/huggingface}"

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
  if [ -n "$TORCH_INDEX_URL" ]; then
    pip install -q "$TORCH_REQUIREMENT" --index-url "$TORCH_INDEX_URL"
  else
    pip install -q "$TORCH_REQUIREMENT"
  fi
else
  pip install -q --upgrade pip
  if [ -n "$TORCH_INDEX_URL" ]; then
    pip install -q "$TORCH_REQUIREMENT" --index-url "$TORCH_INDEX_URL"
  else
    pip install -q "$TORCH_REQUIREMENT" --index-url https://download.pytorch.org/whl/cu121 \
      || pip install -q "$TORCH_REQUIREMENT"
  fi
fi
TORCH_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("torch"))')
printf 'torch==%s\n' "$TORCH_VERSION" > "$DIR/torch-constraint.txt"
pip install -q -c "$DIR/torch-constraint.txt" -r "$SCRIPT_DIR/requirements.txt"
if ! CHECK_OUTPUT=$(pip check 2>&1); then
  # NVIDIA's PyTorch CUDA 13 wheel for ARM SBSA pins cuSPARSELt 0.8.1,
  # whose wheel tag is manylinux2014_sbsa. Current pip reports that tag as
  # unsupported on aarch64 even when installed on the matching DGX Spark.
  # Accept only this exact metadata warning after validating the library and
  # CUDA runtime. Any other dependency or platform error remains fatal.
  if [ "$(uname -m)" = "aarch64" ] \
    && [ "$TORCH_VERSION" = "2.15.0+cu132" ] \
    && [ "$CHECK_OUTPUT" = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform" ]; then
    python - "$DIR/.venv" << 'EOF'
import ctypes
import importlib.metadata as metadata
import pathlib
import sys

venv = pathlib.Path(sys.argv[1])
dist = metadata.distribution("nvidia-cusparselt-cu13")
wheel = dist.read_text("WHEEL") or ""
assert "Tag: py3-none-manylinux2014_sbsa" in wheel, wheel
assert dist.version == "0.8.1", dist.version
library = venv / "lib/python3.12/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"
assert library.is_file() and library.stat().st_size > 0, library
ctypes.CDLL(str(library))
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 1, "No CUDA device found"
print(f"[ok] accepted NVIDIA SBSA cuSPARSELt wheel; loaded {library.name}")
print(f"[ok] CUDA device: {torch.cuda.get_device_name(0)}")
EOF
    say "pip reported only the known SBSA wheel-tag mismatch; runtime checks passed"
  else
    printf '%s\n' "$CHECK_OUTPUT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CHECK_OUTPUT"
fi

# ---- 4. 推理代码 ----
say "从本目录复制固定的评分器与 schema 代码"
cp "$SCRIPT_DIR/metask-jev_scorer.py" "$DIR/jev_scorer.py"
cp "$SCRIPT_DIR/metask-jev_schema.py" "$DIR/jev_schema.py"

# ---- 5. 模型下载（经 huggingface_hub snapshot）----
say "下载 $MODEL_ID@$MODEL_REVISION（约 8.5 GB，可续传）"
python - "$MODEL_ID" "$MODEL_REVISION" "$DIR/model_path.txt" << 'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])
print("model at:", p)
with open(sys.argv[3], "w") as f:
    f.write(p)
EOF
MODEL_PATH=$(cat "$DIR/model_path.txt")

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
r = score(model, tok, state, schema, temperature=2.375)  # noul temperature
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
print(score(model, tok, state, schema, temperature=2.375))
PY

  Temperatures: choice 1.9 / noul 2.375 / score 2.3 (see metask-temperature.json).
TIP
