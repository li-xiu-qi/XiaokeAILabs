#!/bin/bash
source ~/xinfer-env/bin/activate
echo "=== vLLM 可用版本 ==="
pip index versions vllm 2>&1 | head -3
echo "=== 系统架构 ==="
uname -m
echo "=== Python 版本 ==="
python --version
echo "=== PyTorch 状态 ==="
pip show torch 2>&1 | grep -E "Name|Version"
