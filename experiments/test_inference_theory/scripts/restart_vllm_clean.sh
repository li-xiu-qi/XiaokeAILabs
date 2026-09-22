#!/bin/bash
source /home/ke/xinfer-env/bin/activate

# 杀掉所有 vllm 相关进程
echo "清理所有 vllm 进程..."
pgrep -f "vllm" | while read pid; do
    kill -9 $pid 2>/dev/null
done
sleep 5

# 检查 GPU 内存是否释放
echo "GPU 状态:"
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader

# 如果还有残留，再等一会儿
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "可用内存: ${FREE_MEM} MB"

if [ "$FREE_MEM" -lt 60000 ]; then
    echo "等待 GPU 内存释放..."
    sleep 30
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
fi

# 用更低的显存利用率重启
echo "启动 vLLM（gpu-memory-utilization=0.5）..."
export PATH="/home/ke/vllm-env/bin:$PATH"
cd /home/ke/vllm-env
setsid ./bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.5 \
    --enforce-eager \
    --port 8000 \
    > /tmp/vllm_final2.log 2>&1 < /dev/null &

sleep 2
echo "vLLM 已启动（enforce-eager 模式，跳过 torch.compile）"
disown
