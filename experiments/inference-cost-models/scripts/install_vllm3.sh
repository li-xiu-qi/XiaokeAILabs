#!/bin/bash
source $HOME/xinfer-env/bin/activate

echo "=== 安装 PyTorch ==="
uv pip install --python $HOME/vllm-env/bin/python torch torchvision torchaudio 2>&1 | tail -20
