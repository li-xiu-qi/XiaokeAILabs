#!/bin/bash
source ~/xinfer-env/bin/activate

echo "=== 创建 vLLM 独立环境 ==="
cd ~
uv venv --python 3.12 vllm-env

echo "=== 安装 PyTorch (CUDA 13.0 for GB10) ==="
~/.xinfer-env/bin/uv pip install --python ~/vllm-env/bin/python torch torchvision torchaudio 2>&1 | tail -5

echo "=== 安装 vLLM ==="
~/.xinfer-env/bin/uv pip install --python ~/vllm-env/bin/python vllm 2>&1 | tail -10

echo "=== 验证安装 ==="
~/vllm-env/bin/python -c "import vllm; print('vLLM version:', vllm.__version__)" 2>&1
~/vllm-env/bin/python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>&1
