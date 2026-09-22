# KV Cache 显存模型

> 实证日期 2026-09-08 · Python 3.13.0 · 脚本 `scripts/kv_cache_model.py` · 用真实配置算每 token 字节

KV cache 是长上下文推理的主要内存成本，随上下文长度和 batch 线性增长。这条模型回答：给定显存，能撑多长上下文、多大并发，以及 GQA 和 KV 量化能省多少。

## 核心公式

```
KV cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes × batch
```

2 是 K 和 V 两份。关键参数是 `num_kv_heads`（KV 头数），GQA/MQA 把它压到远小于注意力头数，KV cache 同比缩小。

Qwen3.8-27B 是 GQA：注意力头 24 个，KV 头只有 4 个。算出来每 token KV cache 是 256 KB（bf16），而标准 MHA（24 个 KV 头）要 1536 KB，**GQA 省了 6 倍**（脚本 [1]）。这个 6 倍直接决定长上下文的可行性。

## 各上下文长度的显存占用（Qwen3.8-27B，bf16，batch=1）

脚本 [2] 输出：

| 上下文 | KV cache | + bf16 权重 | 总计 | 占 121GB |
|---|---|---|---|---|
| 1K | 0.27 GB | 54.6 GB | 56.9 GB | 47% |
| 8K | 2.15 GB | 54.6 GB | 58.7 GB | 49% |
| 32K | 8.59 GB | 54.6 GB | 65.2 GB | 54% |
| 128K | 34.36 GB | 54.6 GB | 91.0 GB | 75% |
| 256K | 68.72 GB | 54.6 GB | 125.3 GB | 104% |

到 256K 原生上限时，KV cache（68.7GB）已经超过权重本身（54.6GB），总显存突破 121GB 装不下。所以 Qwen3.8-27B 在 GB10 上跑满 262K 上下文需要量化权重（换 q4，权重降到 16.4GB，才有空间给 KV）。

## batch 线性缩放

KV cache 随 batch 线性增长（脚本 [3]，8K 上下文 q4 权重）：batch 1 是 2.15GB KV，batch 16 是 34.4GB，batch 64 是 137GB 直接爆显存。所以并发数受 KV cache 显存硬约束，不是想加多少加多少。

## 最大上下文（q4 权重 + bf16 KV，GB10 121GB）

脚本 [4]：Qwen3.8-27B 能撑到约 39 万 tokens（超过原生 262K，需 YaRN 外推）。如果 KV 也量化到 fp8（脚本 [5]），最大上下文翻倍到约 78 万 tokens。这就是社区跑 1M 上下文的路径：NVFP4 权重 + fp8 KV + FlashAttention 三管齐下。

## 实践判据

长上下文部署的三道约束按紧张程度排序：KV cache 显存（随长度和 batch 线性，最先爆）、注意力计算量（O(S²)，见注意力复杂度模型）、权重显存（固定）。省 KV 的两个杠杆是 GQA/MQA（架构层面，Qwen3.8-27B 已用）和 KV 量化（fp8 KV，运行时可选）。
