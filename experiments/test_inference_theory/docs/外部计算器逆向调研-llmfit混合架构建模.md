# 外部计算器逆向调研：llmfit 混合架构建模

> 逆向自 AlexsJones/llmfit（MIT, 35.1k star, Rust），提取核心公式，Python 重写整合到本实验。
> 日期：2026-09-10

## 为什么逆向 llmfit

LLM-Viewer 适配器在 Qwen3.8-27B（混合架构）上 KV Cache 误差 1440x 到 1650x，
prefill 误差 2852% 到 2989%。原因：LLM-Viewer 把所有 64 层都当 full attention，
Qwen3.8-27B 实际只有 16 层 full attention（`full_attention_interval: 4`），其余 48 层是 linear attention，固定大小的循环状态，不计入 per-token KV Cache。

llmfit 源码里直接写了 Qwen3.8-27B 的混合架构模板，是本实验最准确的参考实现。

## 逆向出的核心公式

### 1. KV Cache（models.rs `kv_cache_gb`）

只对 full attention 层计算：

```
kv_bytes = 2 * n_kv_heads * head_dim * ctx * dtype_bytes * n_full_layers
kv_gb = kv_bytes / 2^30
```

Qwen3.8-27B（ctx=128）：16 层 full × 4 kv_heads × 256 head_dim × 2 bytes = 0.0078 GB。
标准 Transformer 全部 64 层：0.0312 GB。差异正好 4x，这就是 LLM-Viewer 高估的倍数。

### 2. 显存（models.rs `estimate_memory_gb`）

```
total = params_b × bpp + kv_cache_gb + overhead(0.5 GB)
```

overhead 0.5 GB 包含 CUDA context 和混合架构的 fixed recurrent state。

### 3. Decode TPS（fit.rs `estimate_tps`，带宽路径）

```
raw_tps = bandwidth_GB_s / (params_b × bytes_per_param)
tps = raw_tps × efficiency × run_mode_factor
```

efficiency 默认 0.55（通用经验值），run_mode_factor GPU 模式为 1.0。

### 4. Prefill TPS（fit.rs `estimate_prefill`，算力路径）

```
flops_per_token = 2 × params
usable_flops = tflops × 1e12 × 0.35    # PREFILL_COMPUTE_UTILIZATION
prefill_tps = usable_flops / flops_per_token
```

关键：prefill 用 35% 的峰值算力（不是 100%），这比 LLM-Viewer 的 100% 假设更接近实际。
Qwen3.8-27B prefill 预测 578.8 tok/s。

**2026-09-13 更新**：这个 35% 已用 LLM-Viewer 独立交叉验证，结论是它仍偏高。GB10 实测标定值是 20.8%（`PREFILL_COMPUTE_UTILIZATION_GB10`），对应预测 344.0 tok/s；LLM-Viewer 逐层 roofline 是 730.5 tok/s（等效 utilization=44.2%）。三个档位对实测最优 152.04 tok/s 分别高估 126% / 281% / 380%，没有一档能进 10%。详见 `docs/LLM-Viewer交叉验证.md`。

## 与本实验的整合

`scripts/hybrid_architecture_model.py` 已实现上述全部公式。

### 实测验证（GB10, bs=1, seq=128）

| 指标 | BF16 预测 | BF16 实测 | 误差 | Q4 预测 | Q4 实测 | 误差 |
|------|-----------|-----------|------|---------|---------|------|
| decode tok/s | 4.35 | 4.30 | 1.2% | 13.12 | 12.10 | 8.4% |
| 权重显存 GB | 54.00 | 50.90 | 6.1% | 15.19 | 17.50 | 13.2% |

### 与 LLM-Viewer 的对比

| 指标 | LLM-Viewer | llmfit 逆向 | 实测 |
|------|-----------|------------|------|
| BF16 decode | 6.07 tok/s（+39.6%） | 4.35 tok/s（+1.2%） | 4.30 |
| Q4 decode | 22.31 tok/s（+84%） | 13.12 tok/s（+8.4%） | 12.10 |
| BF16 prefill | 730.5 tok/s（+380%） | 578.8 tok/s（+281%） | 152.04 |
| KV Cache | 1440x~1650x 高估 | 准确（只算 16 full 层） | — |

注：LLM-Viewer 两列均为 2026-09-13 按 GB10 硬件参数正确标定后的逐层 roofline 结果（此前的 6.28/15.19 是未适配混合架构的旧值）。

## 关键发现

1. **混合架构的核心是 KV Cache 只算 full attention 层**：48 层 linear attention 的固定循环状态计入 0.5 GB overhead，不计入 per-token KV Cache。
2. **Decode 的 86%/73% 效率是 GB10 实测标定值**，比 llmfit 通用 0.55 更准。
3. **Prefill 的 35% 算力利用率**是 llmfit 的经验常数，比 LLM-Viewer 的 100% 假设更合理。2026-09-13 交叉验证后按 GB10 实测重新标定为 20.8%，但**调常数仍修不干净**：20.8% 档预测 344.0 tok/s 对实测最优 152.04 仍高估 126%。prefill 的系统性偏差根因是混合架构 48 层 linear attention 无法高效 batch，加上 GB10 kernel 融合程度远低于峰值算力。
4. **Q4_K_XL 显存偏差 13.2%**：可能是因为 Q4_K_XL 的实际 bpp 略高于 0.5625（llmfit 公式里的固定值），需要从实际 GGUF 文件读取精确值。
