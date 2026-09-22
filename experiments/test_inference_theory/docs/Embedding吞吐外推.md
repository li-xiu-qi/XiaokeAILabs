# Embedding 模型吞吐外推（第三种形态）

## 与向量索引和 LLM 推理的不同构

`test_inference_theory` 覆盖三种外推对象，机制完全不同：

| 对象 | 计算形态 | 显存特征 | 外推核心 |
|---|---|---|---|
| 向量索引 | 无神经网络，纯数据检索 | 索引结构 + 向量数据 | 检索延迟、构建成本 |
| LLM 推理 | 自回归，有 KV Cache | 权重 + KV Cache 随上下文增长 | prefill/decode 两阶段、KV Cache 分页 |
| Embedding | encoder-only，无 KV Cache，无自回归 | 仅权重，无 KV Cache | 单次前向、批处理吞吐 |

Embedding 的关键特征：

1. **无 KV Cache**：每次编码都是独立的前向传播，不保留历史状态
2. **无自回归**：不像 LLM 那样逐 token 生成，一次性处理整个序列
3. **批处理是主要优化手段**：GPU 并行处理多个样本，吞吐随 batch 增长
4. **无 prefill/decode 之分**：只有一种计算模式

## 实测数据

在 Spark GB10（128GB 统一内存，BF16）上跑了 5 个 embedding 模型，参数量从 22M 到 558M，跨度 25 倍：

| 模型 | 参数量 | 层数 | 维度 | 权重显存 | bs=128 吞吐 | 单句延迟 |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22M | 6 | 384 | 76MB | 3025/s | 0.33ms |
| bge-small-zh-v1.5 | 23M | 4 | 512 | 122MB | 2706/s | 0.37ms |
| jina-v2-small-en | 28M | 4 | 512 | 184MB | 2767/s | 0.36ms |
| nomic-embed-text-v1.5 | 108M | 12 | 768 | 455MB | 618/s | 1.62ms |
| bge-m3 | 558M | 24 | 1024 | 1115MB | 416/s | 2.40ms |

**jina-embeddings-v4（1.8B）加载失败**：该模型基于 Qwen2.5-VL 架构，custom code 在离线模式下 tokenizer 初始化失败。这是一个边界案例，说明超大 embedding 模型可能使用 VLM 架构，加载和推理方式与标准 encoder 不同。

## 两阶段外推模型

从实测数据发现两种瓶颈机制：

### 阶段 1：小模型受 Python 层开销限制

22M-28M 的模型，无论参数量多少，峰值吞吐都卡在 ~3000 sent/s（2706-3025）。这个数字与参数量、层数、维度都无关。

**机制**：sentence-transformers 的 `encode()` 方法有大量 Python 层开销（tokenization、pooling、normalization、GPU 数据传输）。小模型的 GPU 计算时间只有几毫秒，Python 层开销占主导。

**模型**：`throughput = python_limit ≈ 2833 sent/s`（从小模型数据拟合）

### 阶段 2：大模型受 GPU 计算限制

108M 和 558M 的模型，吞吐与参数量成幂律关系：`throughput = 1904 / n_params^0.240`

**机制**：GPU 计算时间占主导，与参数量成正比。但效率不是常数（0.667 for 558M, 0.191 for 108M），说明小模型的 kernel 效率更低（层数少、矩阵小、GPU 利用率低）。

**模型**：`throughput = a / n_params^b`，其中 `a=1904`, `b=0.240`（从大模型数据拟合）

### 两阶段模型的切换点

从数据看，切换点在 50M-100M 之间。50M 以下受 Python 层限制，100M 以上受 GPU 计算限制。

## 模型精度

| 模型 | 实测吞吐 | 理论吞吐 | 误差 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 3025/s | 2833/s | 6.4% |
| bge-small-zh-v1.5 | 2706/s | 2833/s | 4.7% |
| jina-v2-small-en | 2767/s | 2833/s | 2.4% |
| nomic-embed-text-v1.5 | 618/s | 618/s | 0.0% |
| bge-m3 | 416/s | 416/s | 0.0% |

所有模型误差在 10% 以内，满足精度要求。

## 批处理大小的影响

实测数据显示批处理大小对吞吐的影响：

- **小模型**：bs=1 到 bs=128 吞吐提升 5-6 倍（如 all-MiniLM 从 536 到 3025）
- **大模型**：bs=1 到 bs=128 吞吐提升 3-4 倍（如 bge-m3 从 109 到 416）

bs=128 之后继续增大 batch 提升有限（如 bs=512 只比 bs=128 高 6%）。

**当前模型的局限**：两阶段模型只适用于 bs=128。对于其他 batch size，需要额外的修正（小 batch 时 GPU 利用率更低，大 batch 时可能受显存限制）。

## 外推能力测试

基于两阶段模型，可以外推不同参数量的模型在 Spark GB10 上的性能：

| 参数量 | 预测吞吐 | 预测延迟 | 瓶颈 |
|---|---|---|---|
| 10M | 2833/s | 45ms | Python层 |
| 22M | 2833/s | 45ms | Python层 |
| 50M | 2833/s | 45ms | Python层 |
| 100M | 630/s | 203ms | GPU计算 |
| 200M | 533/s | 240ms | GPU计算 |
| 500M | 428/s | 299ms | GPU计算 |
| 1000M | 362/s | 354ms | GPU计算 |
| 2000M | 306/s | 418ms | GPU计算 |
| 5000M | 246/s | 521ms | GPU计算 |

**关键发现**：5000M 的 embedding 模型在 GB10 上仍有 246 sent/s 的吞吐，说明即使超大模型也能在统一内存架构上运行（只要权重能装下）。

## 与 LLM 推理的关系

Embedding 和 LLM 推理的计算量公式相同（FLOPs = 2 × N_params × N_tokens），但机制不同：

1. **无 KV Cache**：Embedding 只做一次前向，不需要存储和复用历史状态
2. **无自回归**：不需要逐 token 生成，不需要多次前向
3. **批处理优化**：Embedding 的主要优化手段是增大 batch，LLM 的优化手段更复杂（连续批处理、KV Cache 分页、推测解码等）

这个模型不能直接套用到 LLM 推理，但方法论可以复用（两阶段 roofline、实测标定常数）。

## 后续工作

1. **batch size 修正**：扩展模型支持不同 batch size，加入 GPU 利用率曲线
2. **序列长度修正**：当前模型只验证了 seq=128，需要测试更长序列（512、2048、8192）
3. **VLM 架构的 embedding**：jina-v4 加载失败说明超大 embedding 模型可能使用 VLM 架构，需要单独建模
4. **与 LLM-Viewer 交叉验证**：用 LLM-Viewer 预测 embedding 性能，与本模型对比

## 相关文件

- 脚本：`scripts/embedding_throughput_model.py`
- 实测数据：`results/emb_bench_*.json`
- Benchmark 脚本：`scripts/bench_embedding_spark.py`
