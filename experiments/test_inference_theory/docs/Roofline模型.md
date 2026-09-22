# Roofline 模型

> 实证日期 2026-09-08 · Python 3.13.0 · 脚本 `scripts/roofline_model.py` · 预测误差 bf16 -7%、q4 -6%

Roofline 模型（Williams et al. 2009）是判断一段计算受算力还是带宽限制的框架。把它套到 LLM 推理上，能回答两件事：生成速度由什么决定，以及换硬件、换量化后速度怎么变。

## 核心量：算术强度与机器平衡点

一段做 F FLOP、搬运 B 字节的计算，算术强度是 `I = F/B`（FLOP/byte）。硬件有峰值算力 P（FLOP/s）和内存带宽 W（byte/s），机器平衡点是 `ridge = P/W`（FLOP/byte）。

算术强度低于平衡点时，搬运数据花的时间超过计算，kernel 受带宽限制；高于时受算力限制。GB10 的平衡点约 220 FLOP/byte（60 TFLOPs FP16 ÷ 273 GB/s）。

## LLM 推理的两个阶段瓶颈不同

一次前向做约 `2×N` FLOP（N 是参数量），搬运 `N×bpp` 字节权重（bpp 是每参数字节数），算术强度 `I = 2/bpp`。代入各精度：bf16 是 1.0、fp8 是 2.0、q4 约 3.3 FLOP/byte，全都远低于 GB10 的 220。所以 **LLM 推理在 GB10 上几乎总是带宽瓶颈**，这个结论和上下文长度、batch 大小基本无关。

但 Prefill 和 Decode 的算术强度差别很大：

- **Decode（逐 token 生成）**：一次只算一个 token，`2×N` FLOP 配 `N×bpp` 字节搬运，强度低到 1~3，纯粹带宽瓶颈。
- **Prefill（prompt 处理）**：一次算 S 个 token，`2×N×S` FLOP 配同样的 `N×bpp` 字节（权重只读一遍），强度随 S 线性增长，长 prompt 时接近或越过平衡点，转为算力瓶颈。

## Decode 速度公式（含 KV cache）

Decode 每生成一个 token，要把全部权重加全部 KV cache 从内存读一遍，所以：

```
tok/s = 带宽 × 效率 / (权重字节 + KV cache 字节)
```

这条公式在 2026-09-08 修正过：早期版本只算权重字节，漏了 KV cache，导致长上下文 decode 速度被高估。KV cache 随上下文线性增长（每 token 256KB），到 128K 时约 34GB，和权重（54.6GB）同阶，decode 每步多读六成，速度掉四成。短上下文（1K 以下）KV cache 占比小可忽略，长上下文必须计入。

## 实测验证

用 Qwen3.8-27B 在 GB10 上的两个实测点验证（脚本 [1] 反推有效带宽）：

| 精度 | 权重字节 | 实测 tok/s | 有效带宽 | 峰值占比 |
|---|---|---|---|---|
| bf16 | 54.6 GB | 4.3 | 235 GB/s | 86% |
| q4_K_XL | 16.4 GB | 12.1 | 198 GB/s | 73% |

预测 bf16 误差 -7%、q4 -6%，达标。

## 论文校准：解码阶段不能用单一利用率系数

2026-09-08 从 arXiv 2512.01644（A100/Llama-3-8B，warp 级 stall 分析）补入的结构性事实，用于解释部分卸载场景 43%–63% 的误差。

**GPU 执行周期占比本来就低。** Prefill 的 GPU Execution 占 27%、Decode 占 24%，即 70%–80% 的周期在 stall。这印证了上表需要乘效率因子，但说明该因子不是常数。

**解码阶段的 stall 是三种性质不同的开销叠加，各有成因：**

| kernel | 主导 stall | 占比 | 机制 |
|---|---|---|---|
| FFN-Up | Memory Dependency | 58% | GEMV 模式，无法隐藏权重加载延迟 |
| O-Proj / FFN-Down | Execution Dependency | 51% | 130 个活跃寄存器，可调度 warp 受限 |
| LayerNorm | Synchronization | 27.7%（Prefill 11.0%） | 小批量 reduction 的 barrier 尾延迟 |
| AttnCore | Memory Dependency | 32.7% | KV cache 密集访问 |

**建模含义**：部分卸载时每层在 GPU/CPU 之间切分，三类 stall 的比例随层数分配而移动，单一系数必然失准。全 GPU 路径上比例固定，所以常数能拟合出 7% 误差；一切换 `-ngl`，误差就跳到 63%。修法是分层给 stall 权重，而不是继续调总系数。

**算术强度随上下文会跨过平衡点。** Decode kernel 的 AI 实测区间是 1–10 FLOP/byte，Prefill 是 55–100；但长上下文 Summary 场景下 Attention 的 AI 升到 319.3（Llama-3-8B）和 382.1（Qwen2.5-32B），反超 GB10 的 220 平衡点。FFN 的 AI 从 Prefill 的 95 掉到 Decode 的 8，同时 DRAM 利用率升 62%。所以「decode 恒为带宽瓶颈」只在短上下文成立，长上下文必须按 attention/FFN 分类。

**L2 局部性的退化幅度**：Prefill→Decode 的 L2 命中率中位数降 54.0%（Chat）/58.5%（Summary）；Attention kernel 的 L2 命中率掉 81.9%，DRAM 利用率最高升 76.4%。

来源：arXiv 2512.01644，原文与 markdown 在 `pkm-hub-papers/论文阅读/papers/2512.01644/`。**注意这些 stall 比例是 A100 的实测，GB10 的 SM 数、寄存器文件、L2 大小都不同，不能直接搬，需要重新标定。**

## 算术强度 2/bpp 的推导出处

`docs/` 各处用 `AI = 2/bpp`（bf16 = 1.0、fp8 = 2.0、q4 ≈ 3.3 FLOP/byte），这个式子的两步假设来自 arXiv 2402.16363（LLM Inference Unveiled）：

1. **每 MAC 计 2 ops**。原文脚注：「Each Multiply-Accumulate (MAC) operation counts two operations.」所以一次矩阵乘的 FLOP = 2 × 参数量。
2. **搬运字节 = 参数量 × 每参数字节数**。原文举例：「Llama-13b ... 13 billion weights, occupies approximately 26GB of memory in FP16 format」，即 FP16 每参数 2 字节。

两者相除得 `AI = 2N / (N × bpp) = 2/bpp`，与参数量无关。这是「decode 恒为访存受限」的数学根源。

同篇 Table 1 给出 Llama-2-7b 在 A6000（FP16）上的分层实测：prefill 阶段多数层 compute-bound，decode 阶段全部 memory-bound，与本文两阶段结论一致。

该篇的 LLM-Viewer 工具（`github.com/feifeibear/LLM-Viewer`，与论文原版 `hahnyuan/LLM-Viewer` 同源）把硬件规格 + 模型配置 + 量化位宽 + batch/序列长作为输入，逐层算性能与峰值显存。**与本实验的方法论同构**，可作为交叉验证：把它对 Qwen3.8-27B / GB10 的预测与本实验公式对比，差异超过 10% 则说明某一侧的效率假设要修。

**交叉验证已执行（2026-09-13，`docs/LLM-Viewer交叉验证.md`）**。在 Spark 上部署 LLM-Viewer，手工补入 GB10 硬件参数（bandwidth=273 GB/s、FP16=89.3 TFLOPS）与 Qwen3.8 混合架构配置，跑 bs=1 / seq=128 的逐层分析。核心结论：

| 指标 | LLM-Viewer 预测 | 本实验实测最优 | 误差 | 反推有效带宽 |
|---|---|---|---|---|
| decode tok/s | 6.07 | 4.35（四引擎均值 4.30–4.47） | +39.6% | 71.6%（带宽） |
| prefill tok/s | 730.5 | 152.04（vLLM full compile） | +380.5% | 20.8%（带宽，非算力） |

- **decode 侧：本实验的效率常数站得住**。LLM-Viewer 假设 100% 带宽（performance = bandwidth），反推 71.6%，与本实验标定的 82–86% 同一量级。差距主要来自 LLM-Viewer 只算 44.9 GB 权重（漏了 embedding 与混合架构真实投影维度），实际要读 51.1 GB；按实际字节反推是 82.4%。**不是效率常数错，是权重字节口径不同。**
- **prefill 侧：建模口径要换**。bs=1/seq=128 下所有层都是 memory bound，730.5 这个数是带宽下界而非算力上界。4.80 倍偏差分解为权重字节 1.14x × 有效带宽 4.50x，主导项是有效带宽。核心发现是 **prefill 有效带宽（60.7 GB/s，22.2% 峰值）只有 decode（224.8 GB/s，82.4% 峰值）的四分之一**，原因是 prefill 每字节配十倍计算（AI 55–100 vs 1–10），在 GB10 的低算力下 kernel 藏不住访存延迟。原来那个「35% 算力利用率」把带宽不足、激活开销、未建模算子全揉进一个乘子，属经验拟合而非机理分解。

> 一处需澄清的历史记录：本文此前有一版「2026-09-09 交叉验证」的记载，称 LLM-Viewer 反推有效带宽 272.7 GB/s（峰值 99.9%）。该日期在 `迭代日志.md` 中查不到对应条目，272.7 这个数也无法由任何一组已知输入复现（按 LLM-Viewer 自身的带宽假设它应恰好是 273）。以本条 2026-09-13 的实测为准。

> 另有一处待对齐的内部不一致：本节上文按 60 TFLOPs 算平衡点为 220 FLOP/byte，而硬件常数表已更新为实测 89.3 TFLOPS（对应平衡点 327 FLOP/byte）。两处数值未同步，涉及平衡点的结论（含 prefill 何时转入 compute-bound）应以 89.3 TFLOPS / 327 为准。

## 已弃用假设

| 精度 | 权重字节 | 实测 decode | 反推有效带宽 | 占峰值 273 GB/s |
|---|---|---|---|---|
| bf16 | 54.6 GB | 4.3 t/s | 235 GB/s | 86% |
| q4_k_xl | 16.4 GB | 12.1 t/s | 198 GB/s | 73% |

两个点反推出的有效带宽都在峰值带宽的 73%~86%，符合「统一内存实测带宽约为峰值 70~85%」的经验区间，说明 273 GB/s 峰值取值和公式都正确。（实测在 ctx=1024 下进行，KV cache 占比 0.3GB 可忽略，反推按纯权重近似。）

再用反推的效率回代预测（脚本 [2]）：bf16 预测 4.0 t/s（实测 4.3，-7%），q4 预测 11.3 t/s（实测 12.1，-6%）。误差在可接受范围，模型可用。

## decode 随上下文的下降（修正后）

修正后的公式把 KV cache 计入，decode 速度随上下文下降（脚本 [3]，bf16）：

| 上下文 | KV cache | 每步读取 | decode t/s | 瓶颈 |
|---|---|---|---|---|
| 1K | 0.3 GB | 54.9 GB | 3.98 | 带宽 |
| 8K | 2.1 GB | 56.7 GB | 3.85 | 带宽 |
| 32K | 8.6 GB | 63.2 GB | 3.46 | 带宽 |
| 128K | 34.4 GB | 89.0 GB | 2.46 | 带宽 |
| 256K | 68.7 GB | 123.3 GB | 1.77 | 带宽 |

到 256K 原生上限，decode 从 3.98 掉到 1.77 t/s，降幅 55%。这是早期只算权重的模型完全看不到的。短上下文（8K 内）降幅很小（3%），长上下文才显著。这条把 Roofline 模型和 KV cache 显存模型接上了：KV cache 不只是「占显存」，还直接拖慢 decode 速度。

## 外推：换硬件和换模型

公式可以直接外推，无需重新实测。

**换硬件**（脚本 [3]）：Qwen3.8-27B q4_k_xl 在 4090 上，带宽 1008 GB/s 是 GB10 的 3.7 倍，预测 decode 约 49 t/s。这是同模型换到高带宽卡的线性外推。

**换大模型**（脚本 [4]）：320B 的 MoE（GLM-5.3-Flash 量级）激活参数约 124B，bf16 下预测 decode 仅 0.9 t/s；但显存要装全部 320B 专家 = 640GB bf16，GB10 121GB 装不下，量化到 nvfp4 也要 176GB 仍超，得 q3 级或 DSpark 格式。这条线演示了「速度公式」和「显存公式」必须一起用，单看一个会得出错误结论。

## 对部署选型的直接指导

decode 速度 ≈ 带宽 / 权重字节，所以提升 decode 只有两条路：提高有效带宽（更好的内存子系统、更高的带宽利用率），或减小权重字节（量化）。这也是为什么 GB10 这类高带宽统一内存机器特别适合量化后的模型，量化对速度的提升几乎和量化比例成正比。

Prefill 受峰值算力限制，提升靠更强的算力或 FP8/FP4 计算。
