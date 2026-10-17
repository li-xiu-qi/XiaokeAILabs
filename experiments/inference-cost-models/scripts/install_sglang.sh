#!/bin/bash
source /home/ke/xinfer-env/bin/activate

echo "=== 创建 SGLang 独立环境 ==="
cd ~
uv venv --python 3.12 sglang-env

echo "=== 安装 PyTorch ==="
uv pip install --python /home/ke/sglang-env/bin/python torch torchvision torchaudio 2>&1 | tail -5

echo "=== 安装 SGLang ==="
uv pip install --python /home/ke/sglang-env/bin/python "sglang[all]" 2>&1 | tail -20

echo "=== 安装 ninja ==="
uv pip install --python /home/ke/sglang-env/bin/python ninja 2>&1 | tail -3

echo "=== 验证 ==="
/home/ke/sglang-env/bin/python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
import sglang
print('SGLang:', sglang.__version__)
" 2>&1
