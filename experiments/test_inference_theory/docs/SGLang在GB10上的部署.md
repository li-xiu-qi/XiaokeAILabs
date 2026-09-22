# SGLang 在 DGX Spark（GB10）上的部署

> 实测日期：2026-09-13
> 硬件：NVIDIA GB10（统一内存 128 GB，CUDA 13.0，compute capability 12.1）
> 模型：Qwen3.8-27B-BF16（64 层混合架构，16 full attention + 48 linear attention）
> SGLang 版本：0.5.19

## 环境准备

```bash
# 创建独立环境
uv venv --python 3.12 sglang-env

# 安装 PyTorch（CUDA 13.0）
uv pip install --python sglang-env/bin/python torch torchvision torchaudio

# 安装 SGLang（all extras）
uv pip install --python sglang-env/bin/python "sglang[all]"

# 安装 ninja
uv pip install --python sglang-env/bin/python ninja
```

## 启动命令

```bash
export PATH="/home/ke/sglang-env/bin:$PATH"

cd /home/ke/sglang-env
setsid ./bin/python -m sglang.launch_server \
    --model-path /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 2048 \
    --mem-fraction-static 0.7 \
    --max-running-requests 32 \
    --port 8001 \
    > /tmp/sglang.log 2>&1 < /dev/null &
```

### 参数说明

| 参数 | 值 | 原因 |
|------|-----|------|
| `--mem-fraction-static` | 0.7 | 混合架构的 Mamba cache 很大，0.5 不够 |
| `--max-running-requests` | 32 | 配合 Mamba cache 限制，避免 OOM |
| `--context-length` | 2048 | 混合架构在长上下文下性能衰减严重 |
| `--dtype` | bfloat16 | 与模型权重格式一致 |

**不要用 mem-fraction-static=0.5**：会报 `Not enough GPU memory for hybrid (mamba/linear-attention) state cache. Computed max_mamba_cache_size=-72`。

## 踩坑记录

### 1. Mamba cache 显存不够

**症状**：模型加载完后，Scheduler 崩溃，报 `Not enough GPU memory for hybrid (mamba/linear-attention) state cache. Computed max_mamba_cache_size=-72 (total_rest_memory=-21.21 GB, mamba_cache_per_req=146.81 MB)`。

**原因**：Qwen3.8-27B 的 48 层 linear attention（GDN 架构）每个序列需要 146.81 MB 的 Mamba cache。mem-fraction-static=0.5 时，剩余内存不够分配。

**解决**：提高 `--mem-fraction-static` 到 0.7，同时降低 `--max-running-requests` 到 32。

### 2. CUDA graph 捕获慢

**症状**：模型加载完后，卡在 `Capturing num tokens` 进度条，58 个 graph 每个 4-6 秒，总共约 5 分钟。

**原因**：SGLang 为不同 token 数（4 到 8192）捕获 prefill CUDA graph，共 58 个。每个 graph 的捕获需要在 GPU 上实际运行一次。

**这是正常现象**，不是配置问题。捕获完后 prefill 性能会很好。

### 3. 不能和其他引擎同时跑

**症状**：同时启动 vLLM 和 SGLang，两个引擎各加载 51 GB 模型，128 GB 统一内存瞬间耗尽，系统卡死，sshd 被 OOM killer 杀掉。

**原因**：GB10 是统一内存架构，GPU 内存和系统内存是同一个池子。两个推理引擎的 GPU 分配加起来超过物理内存时，系统进程（sshd、Xorg、GNOME）全部遭殃。

**解决**：一次只跑一个引擎。启动前确认：

```bash
# 确认没有残留进程
ps aux | grep -E 'vllm|sglang' | grep -v grep

# 确认 GPU 内存已释放
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

## 实测性能（GB10, bs=1/bs=32, seq=128）

| 引擎 | decode tok/s | prefill tok/s |
|------|-------------|---------------|
| llama.cpp | 4.30 | 25.8 |
| vLLM 0.28.0 (enforce-eager) | 4.39 | 11.34 |
| vLLM 0.28.0 (full compile) | 4.44 | 152.04 |
| SGLang 0.5.19 | 4.47 | 121.53 |
| 模型预测 | 4.35 | 578.8 |

### 结论

- **decode 与引擎无关**：SGLang 4.47 tok/s，与 vLLM（4.44）和 llama.cpp（4.30）基本一致。decode 是带宽瓶颈。
- **prefill 略低于 vLLM full**：SGLang 121.53 vs vLLM full 152.04，差 20%。但 SGLang 的 CUDA graph 捕获更细粒度（58 个 token 数 vs vLLM 的 51 个），理论上应该更好。差距可能来自混合架构的 linear attention kernel 实现差异。
- **SGLang 对混合架构的 Mamba cache 管理更严格**：mem-fraction-static=0.5 直接报错，需要 0.7。

## 与 vLLM 的对比

| 维度 | SGLang 0.5.19 | vLLM 0.28.0 (full) |
|------|--------------|-------------------|
| decode tok/s | 4.47 | 4.44 |
| prefill tok/s | 121.53 | 152.04 |
| 启动时间 | ~12 分钟（含 CUDA graph 捕获） | ~10 分钟（含 torch.compile） |
| 显存利用率 | 0.7 | 0.5 |
| Mamba cache 要求 | 高（需要 0.7） | 中（0.5 可用，但需限制 max-num-seqs） |
| CUDA graph 粒度 | 58 个 token 数 | 51 个 token 数 |

vLLM 在 prefill 上略胜一筹，且显存利用率更低（0.5 vs 0.7），在 GB10 这种内存紧张的硬件上更有优势。SGLang 的 decode 稍快一点，但差距在误差范围内。

## 下一步

- 测 Q4_K_XL 量化版，看量化对两个引擎的影响
- 测更长上下文（2048、4096）下的性能衰减
- 测更大 batch size（受 Mamba cache 限制，可能需要更高显存利用率）
