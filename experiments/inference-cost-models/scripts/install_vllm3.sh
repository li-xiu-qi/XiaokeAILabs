#!/bin/bash
source /home/ke/xinfer-env/bin/activate

echo "=== 安装 PyTorch ==="
uv pip install --python /home/ke/vllm-env/bin/python torch torchvision torchaudio 2>&1 | tail -20
