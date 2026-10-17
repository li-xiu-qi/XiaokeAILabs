# vLLM 在 DGX Spark（GB10）上的部署

> 实测日期：2026-09-10
> 硬件：NVIDIA GB10（统一内存 128 GB，CUDA 13.0，compute capability 12.1）
> 模型：Qwen3.8-27B-BF16（64 层混合架构，16 full attention + 48 linear attention）
> vLLM 版本：0.28.0

## 环境准备

### 硬件限制

GB10 是统一内存架构，CPU 和 GPU 共享同一块内存。这带来两个直接影响：

1. **CUDA 可见内存少于物理内存**：`nvidia-smi` 显示总内存 121.69 GB，但系统进程（Xorg、GNOME 等）已占用约 33 GB，实际可用约 88.7 GB。
2. **GPU 内存和系统内存相互挤压**：`--gpu-memory-utilization` 不能设太高，否则启动直接失败。

### 软件环境

```bash
# 创建独立环境（避免和 xinfer-env 的依赖冲突）
uv venv --python 3.12 vllm-env

# 安装 PyTorch（CUDA 13.0）
uv pip install --python vllm-env/bin/python torch torchvision torchaudio

# 安装 vLLM
uv pip install --python vllm-env/bin/python vllm

# 安装 ninja（FlashInfer JIT 编译需要）
uv pip install --python vllm-env/bin/python ninja
```

**版本注意**：vLLM 0.28.0 要求 PyTorch 2.13.0，如果先装了 2.14.0，uv 会自动降级，这是正常行为。

## 启动命令

```bash
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
    > /tmp/vllm.log 2>&1 < /dev/null &
```

### 参数说明

| 参数 | 值 | 原因 |
|------|-----|------|
| `--gpu-memory-utilization` | 0.5 | GB10 可用内存仅 88.7 GB，0.5 对应约 60 GB，留足余量 |
| `--enforce-eager` | — | 跳过 torch.compile，启动时间从 15 分钟降到 6 分钟 |
| `--max-model-len` | 1024 | 混合架构的 linear attention 层在长上下文下性能衰减严重 |
| `--dtype` | bfloat16 | 与模型权重格式一致，避免隐式转换 |

**不要用 0.85 或更高的利用率**：GB10 上会直接报 `Free memory on device cuda:0 (88.7/121.69 GiB) on startup is less than desired GPU memory utilization (0.85, 103.44 GiB)`。

## 踩坑记录

### 1. ninja 找不到

**症状**：模型加载完成后，EngineCore 崩溃，报 `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`。

**原因**：FlashInfer 的采样模块需要 JIT 编译，编译过程调用 `ninja` 命令。`uv pip install ninja` 只把 ninja 装到 Python 环境里，没有放到系统 PATH。

**解决**：启动前把 vllm-env/bin 加入 PATH：

```bash
export PATH="/home/ke/vllm-env/bin:$PATH"
```

或者找到 ninja 的实际路径手动加入：

```bash
NINJA_BIN=$(find /home/ke/vllm-env -name "ninja" -type f 2>/dev/null | head -1)
export PATH="$(dirname $NINJA_BIN):$PATH"
```

### 2. torch.compile 太慢

**症状**：用默认参数启动，模型加载完后卡在 `Compiling a graph for compile range (1, 2048) takes 24.32 s`，整个过程超过 15 分钟。

**原因**：GB10 的算力本来就弱（89 TFLOPS BF16），torch.compile 的 autotune 在这个硬件上极慢。日志里有警告 `Not enough SMs to use max_autotune_gemm mode`。

**解决**：加 `--enforce-eager` 跳过 torch.compile。代价是 prefill 性能会差一些（见下文实测数据），但启动时间大幅缩短。

### 3. 残留进程占显存

**症状**：vLLM 启动失败后，再次启动报 `Free memory on device cuda:0 (49.44/121.69 GiB) on startup is less than desired GPU memory utilization (0.5, 60.85 GiB)`。

**原因**：vLLM 的 EngineCore 子进程在崩溃后没有正确释放 GPU 内存。

**解决**：手动清理残留进程：

```bash
pgrep -f "vllm.entrypoints" | while read pid; do kill -9 $pid 2>/dev/null; done
pgrep -f "EngineCore" | while read pid; do kill -9 $pid 2>/dev/null; done
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

确认内存释放后再启动。如果 `nvidia-smi` 返回 N/A，说明 GPU 驱动有问题，等一会儿再试。

### 4. 模型加载慢

**症状**：`Loading safetensors checkpoint shards` 每个分片要 17-20 秒，18 个分片总共约 6 分钟。

**原因**：GB10 的存储 IO 带宽有限（NVMe，但不是数据中心级），加上统一内存架构下加载权重要同时走 PCIe 和内存总线。

**这是正常现象**，不是配置问题。51.75 GB 的模型在 GB10 上加载需要 5-6 分钟。

## 实测性能

### 测试方法

```python
import requests, time

url = "http://localhost:8000/v1/completions"
model = "/home/ke/models/Qwen3.8-27B-BF16"

# Decode: bs=1, prompt="你好", max_tokens=128
payload = {"model": model, "prompt": "你好", "max_tokens": 128, "temperature": 0.7}
start = time.time()
r = requests.post(url, json=payload, timeout=300)
elapsed = time.time() - start

result = r.json()
usage = result['usage']
decode_tps = usage['completion_tokens'] / elapsed

# Prefill: bs=32, prompt=["你好世界。这是一个测试。"]*32, max_tokens=1
payload2 = {"model": model, "prompt": ["你好世界。这是一个测试。"] * 32, "max_tokens": 1, "temperature": 0.7}
start = time.time()
r2 = requests.post(url, json=payload2, timeout=300)
elapsed2 = time.time() - start

result2 = r2.json()
usage2 = result2['usage']
prefill_tps = usage2['prompt_tokens'] / elapsed2
```

### 结果（GB10, seq=128）

| 指标 | vLLM 0.28.0 | llama.cpp | 模型预测 | vLLM vs 预测 |
|------|------------|-----------|----------|-------------|
| decode tok/s | 4.39 | 4.30 | 4.35 | +0.9% |
| prefill tok/s | 11.34 | 25.8 | 578.8 | +5004% |

### 结论

**Decode 性能正常**：4.39 tok/s，与 llama.cpp 的 4.30 tok/s 基本一致，模型预测 4.35 tok/s 准确（误差 0.9%）。这说明 decode 阶段是带宽瓶颈，与推理引擎无关。

**Prefill 性能差**：11.34 tok/s，远低于模型预测的 578.8 tok/s（误差 5004%），也低于 llama.cpp 的 25.8 tok/s。原因有两个：

1. **enforce-eager 模式**：跳过了 torch.compile 的 kernel 融合优化，prefill 的算力利用率低
2. **混合架构的 linear attention 层**：Qwen3.8-27B 有 48 层 linear attention，这些层在 prefill 时无法像 full attention 那样 batch，有效算力利用率进一步降低

**生产环境建议**：如果追求 prefill 性能，不要用 `--enforce-eager`，让 torch.compile 跑完（需要 15 分钟启动）。如果追求快速启动和稳定性，用 `--enforce-eager`，接受 prefill 性能的损失。

## 与 llama.cpp 的对比

| 维度 | vLLM 0.28.0 | llama.cpp |
|------|------------|-----------|
| decode tok/s | 4.39 | 4.30 |
| prefill tok/s | 11.34 | 25.8 |
| 启动时间 | 6 分钟（enforce-eager） | 3-4 分钟 |
| 显存占用 | ~60 GB（gpu_util=0.5） | ~52 GB（BF16 权重） |
| PD 分离 | 支持（需要额外配置） | 不支持 |
| 连续批处理 | 支持 | 部分支持 |

vLLM 的优势在于支持连续批处理和 PD 分离，适合生产环境的多用户场景。llama.cpp 的优势在于启动快、显存占用低、单用户 prefill 性能好。

## 下一步

- 测 Q4_K_XL 量化版，看量化对 vLLM 性能的影响
- 尝试不用 enforce-eager，看 torch.compile 完整编译后的 prefill 性能
- 测 PD 分离模式（单机双实例）
