#!/bin/bash
source /home/ke/xinfer-env/bin/activate

echo "=== 验证 vLLM 和 torch ==="
/home/ke/vllm-env/bin/python -c "
import vllm
import torch
print('vLLM version:', vllm.__version__)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print('Compute capability:', cap)
# 测试 CUDA 是否可用
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x).sum().item()
print('CUDA matmul OK:', round(y, 2))
" 2>&1
