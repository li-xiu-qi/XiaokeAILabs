#!/bin/bash
source ~/xinfer-env/bin/activate

echo "=== 检查网络 ==="
curl -s -o /dev/null -w "PyPI: %{http_code}\n" --max-time 10 https://pypi.org/simple/ 2>&1
curl -s -o /dev/null -w "PyTorch CUDA: %{http_code}\n" --max-time 10 https://download.pytorch.org/whl/cu130 2>&1

echo "=== 安装 PyTorch ==="
~/.xinfer-env/bin/uv pip install --python ~/vllm-env/bin/python torch torchvision torchaudio 2>&1 | tail -20
