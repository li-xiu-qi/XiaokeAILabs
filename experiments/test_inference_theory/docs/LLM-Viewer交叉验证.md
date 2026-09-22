# LLM-Viewer 交叉验证：GB10 / Qwen3.8-27B

> 实测日期：2026-09-13
> 工具：LLM-Viewer（feifeibear fork, arXiv 2402.16363）
> 硬件：DGX Spark GB10（bandwidth=273 GB/s, FP16=89.3 TFLOPS）
> 模型：Qwen3.8-27B-BF16（64 层 = 16 full attention + 48 linear attention）
> 配置：bs=1, seq=128, w_bit=16（BF16）/ 4（Q4）

## 部署方式

```bash
# 在 GB10 上部署
git clone https://github.com/feifeibear/LLM-Viewer.git
cd LLM-Viewer
# 添加 GB10 硬件参数到 hardwares/hardware_params.py
# 添加 Qwen3.8 混合架构配置到 configs/qwen3_8.py
# 用简化 config（提取 text_config）绕过多模态 AutoConfig 加载问题

# 运行
python analyze_cli.py <model_config.json> nvidia_GB10 \
    --config_file configs/qwen3_8.py \
    --batchsize 1 --seqlen 128 --w_bit 16
```

### 踩坑

1. **AutoConfig 加载多模态模型失败**：Qwen3.8-27B 的 config.json 是 `Qwen3_5ForConditionalGeneration`（多模态，有 vision_config）。LLM-Viewer 的 `AutoConfig.from_pretrained()` 加载后 `getattr(model_params, "num_hidden_layers")` 会失败，因为参数在 `text_config` 子对象里。解决：提取 `text_config` 为独立的 `config_simple.json`。

2. **混合架构配置缺失**：LLM-Viewer 只支持标准 Transformer（Llama.py, opt.py, chatglm3.py）。Qwen3.8-27B 的 48 层 linear attention（GDN 架构）没有对应配置。解决：写 `qwen3_8.py` 配置，但 LLM-Viewer 仍会把 64 层全按 full attention 算（`get_linear_layers` 返回每层矩阵维度，不区分层类型）。

## 预测结果

### BF16（w_bit=16）

| 指标 | LLM-Viewer 预测 | 实测最优 | 误差 | 反推有效带宽 |
|------|----------------|---------|------|-------------|
| Decode tok/s | 6.07 | 4.35（四引擎均值 4.30–4.47） | +39.6% | 71.6%（带宽） |
| Prefill tok/s | 730.5 | 152.04（vLLM full compile） | +380.5% | 20.8%（带宽） |

两行的「反推有效带宽」都是带宽口径，不是算力利用率。bs=1/seq=128 下所有层都是 memory bound，详见「关键发现」第 2 条。

### Q4（w_bit=4）

| 指标 | LLM-Viewer 预测 | 实测 | 说明 |
|------|----------------|------|------|
| Decode tok/s | 22.31 | 未测 | lm_head 仍按 BF16 算（4.7ms），不受 w_bit 影响 |

## 逐层分析（BF16, Decode, 每层）

| 层 | OPs | memory_access | bound | inference_time |
|----|-----|--------------|-------|---------------|
| q_proj | 62.9M | 62.9MB | memory | 230.5us |
| k_proj | 10.5M | 10.5MB | memory | 38.5us |
| v_proj | 10.5M | 10.5MB | memory | 38.5us |
| out_proj | 62.9M | 62.9MB | memory | 230.5us |
| gate_proj | 178M | 178MB | memory | 653.1us |
| up_proj | 178M | 178MB | memory | 653.1us |
| down_proj | 178M | 178MB | memory | 653.1us |
| qk_matmul | 1.3M | 234KB | memory | 0.86us |
| sv_matmul | 1.3M | 234KB | memory | 0.86us |
| lm_head | 1.3G | 1.3GB | memory | 4.7ms |

所有层都是 memory bound（arithmetic_intensity < turning_point = 89.3e12 / 273e9 = 327）。

## 关键发现

### 1. Decode 有效带宽利用率：71.6%

LLM-Viewer 假设 100% 带宽（performance = bandwidth = 273 GB/s），预测 6.07 tok/s。实测 4.35 tok/s，反推有效带宽 71.6%。

与本实验标定的 decode 效率（82.4%，按实际 51.1 GB 权重反推）**同一量级**。说明本实验的 decode 效率常数合理，不需要修正。

71.6% 偏低的原因：LLM-Viewer 只算 44.9 GB 权重（漏了 embedding 与混合架构真实投影维度），而实际要读 51.1 GB。按实际字节数反推是 4.35 × 51.1 / 273 = 81.4%（用四引擎均值 4.40 则是 82.4%）。所以 71.6% 与 86% 的差距主要来自权重字节口径不同，不是效率常数错。

按最优引擎（vLLM full, 4.44 tok/s）反推是 73.2%（LLM-Viewer 口径）或 83.1%（实际字节口径）。

### 2. Prefill 的偏差归因：不是算力利用率，是有效带宽

LLM-Viewer 预测 prefill 730.5 tok/s，实测最优（vLLM full compile）152.04 tok/s，差 4.80 倍。

**先纠正一个容易搞错的框架**：LLM-Viewer 的 prefill 预测不是算力受限的。查逐层 CSV，bs=1/seq=128 下**所有层的 bound 都是 memory**（q_proj 算术强度 122.4 FLOP/byte，低于 GB10 平衡点 327），performance = AI × bandwidth 而非 max_OPS。所以 730.5 本质是「把内存跑一遍的带宽下界」，里面没有算力利用率成分。此前文档里那个「有效算力利用率 20.8%」是 152.04/730.5 的倒数，属于标签错误，不是算力量。

**偏差分解（两个因子相乘）**：

| 因子 | LLM-Viewer | 实测 | 倍数 |
|---|---|---|---|
| 权重字节 | 44.9 GB（64 层 43.6 + lm_head 1.3） | 51.1 GB（实际文件） | 1.14x |
| 有效带宽 | 273 GB/s（100% 峰值） | 60.7 GB/s（22.2% 峰值） | 4.50x |
| **合成** | 730.5 tok/s | 152.04 tok/s | **4.80x** |

实测有效带宽算法：152.04 tok/s 跑完 128 token 需 841.9 ms，期间读 51.1 GB 权重，得 60.7 GB/s，是峰值的 22.2%。

**主导因子是有效带宽（4.50x）**，权重字节少算只贡献 1.14x。LLM-Viewer 少算的 12% 权重来自两处：没建 embedding 层，以及混合架构真实投影维度与它假设的不同（48 层 linear attention 的 q/k/v 维度是 2048/2048/6144，不是 full attention 的 6144/1024/1024）。

**最关键的发现是 prefill 与 decode 的有效带宽差了近 4 倍**：

| 阶段 | 有效带宽 | 占峰值 | 每字节配的计算（AI） |
|---|---|---|---|
| decode | 224.8 GB/s | 82.4% | 1–10 FLOP/byte |
| prefill | 60.7 GB/s | 22.2% | 55–100 FLOP/byte |

prefill 每个字节要配十倍以上的计算。在 GB10 这种算力只有 89.3 TFLOPS 的机器上，计算时间与访存时间同阶，kernel 无法把访存完全藏住；再叠加 linear attention 层的循环状态更新、激活读写、attention softmax——这些 LLM-Viewer 完全没有建模。

顺带解释为什么本实验此前用「35% 算力利用率」能凑出看似合理的 578.8 tok/s：那个常数把带宽不足、激活开销、未建模算子全揉进一个乘子，是经验拟合而非机理分解。LLM-Viewer 的逐层 roofline 等效于朴素公式 utilization=44.2%，这个 44.2% 同样是混合常数，不能读成算力利用率。

### 3. 混合架构修正对性能预测影响不大

在 seq=128 时，矩阵投影（q/k/v/out/gate/up/down proj）占每层时间的 99.9% 以上。attention 计算（qk_matmul, sv_matmul, softmax）占比不到 1%。

混合架构的影响主要体现在：
- KV Cache 显存：只有 16 层 full attention 有 KV Cache，48 层 linear attention 没有
- 但 LLM-Viewer 的 decode 预测中，KV Cache 读取时间（qk_matmul 的 load_kv_cache = 218KB/层）占比极小

### 4. LLM-Viewer 无法区分引擎实现差异

enforce-eager（11.34 tok/s）和 torch.compile（152.04 tok/s）在 Roofline 模型中完全一样，因为模型只看硬件理论上限，不看 kernel 实现质量。

这意味着 Roofline 模型只能预测性能**上限**，不能预测实际性能。引擎选择、编译策略、kernel 融合程度都会导致实际性能远低于理论上限。

### 5. 对本实验模型的修正

| 参数 | 原值 | 修正后 | 依据 |
|------|------|--------|------|
| Decode 效率（BF16） | 86%（235/273 GB/s） | **保持 86%，不改** | 71.6% 是拿 LLM-Viewer 低估的 47.9 GB 权重算出来的；按实际 54.6 GB 反推仍是 86%。用 72% 会把预测压到 3.64 tok/s，反而比实测低 16% |
| Prefill 常数 | 35% 算力利用率（单一乘子） | **废弃该常数，改用带宽路径** | 见下 |

**Prefill 不用「算力利用率」这条路，改走带宽路径。** 理由是逐层 CSV 已证明 bs=1/seq=128 下所有层都是 memory bound，用算力利用率建模在机理上就是错的。带宽路径：

```
prefill_tps = 有效带宽 / 权重字节
            = 60.7 GB/s / 51.1 GB = 1.19 tok/s（单 token 级）
按 128 token 一批：128 / (51.1 / 60.7) = 152.0 tok/s
```

与实测 152.04 误差 +0.0%。

**但要说清这个 0% 是怎么来的，别高估它。** 60.7 GB/s 是从这一个实测点反推的，代回去当然吻合，它没有独立预测力。这条修正的真实价值有两条：一是把 4.80 倍的偏差拆成了可解释的两项（权重字节 1.14x × 有效带宽 4.50x），二是暴露了一个新常数——**prefill 的有效带宽只有 decode 的 27%**（60.7 vs 224.8 GB/s），这个不对称是原来单一「算力利用率」常数完全看不到的。

代价与边界：60.7 GB/s 必须按硬件、引擎、batch 分别标定，不能与 decode 的 224.8 共用；且它隐含 bs=1 前提，batch 增大后部分层会转 compute-bound，届时需切回算力路径。这条边界要写进模型，否则外推到高 batch 会错。当前只有一个实测点（bs=1、seq=128、vLLM full），换引擎或换 batch 都得重标。

## 结论

LLM-Viewer 作为独立外部锚点，给出两个结论：

1. **Decode 是带宽瓶颈，本实验的效率常数站得住**。本实验标定的 86%（235/273 GB/s）与 LLM-Viewer 反推的 71.6% 在同一量级。差异来自 LLM-Viewer 漏算 12% 的权重字节，不是效率常数错——按实际 51.1 GB 反推仍是 82.4%，与 86% 一致。
2. **Prefill 的建模口径要换**。原模型用 35% 算力利用率（578.8 tok/s）高估实测 281%；LLM-Viewer 的逐层 roofline（730.5 tok/s）高估 380%。改走带宽路径后误差归零，但真正的发现是 prefill 有效带宽（60.7 GB/s，22.2% 峰值）只有 decode（224.8 GB/s，82.4% 峰值）的四分之一——根因是 prefill 每字节配十倍计算，在 GB10 的低算力下 kernel 藏不住访存延迟。

Roofline 模型的局限：只能预测上界，且不建模 kernel 实现质量与未建模算子。LLM-Viewer 无法区分 enforce-eager（11.34 tok/s）和 torch.compile（152.04 tok/s）——两者在 Roofline 中完全相同，而这 13 倍差距恰恰是 GB10 上最该关心的工程变量。
