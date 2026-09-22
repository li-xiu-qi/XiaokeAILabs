# LLM 推理性能模型：显存、速度、规模化与外推

这个实验不测工具，测大模型推理本身的数学性质。目标是建立可计算的性能模型（cost model），输入「模型配置 + 硬件规格 + 量化精度」，就能递推出显存占用、推理速度、最大上下文、服务吞吐，并能外推到没实测过的模型和硬件。

和 `test_index_theory`（索引与存储引擎的性能模型）是同一方法论，对象从索引结构换成了 LLM 推理。

改模型前先读 `docs/迭代日志.md`。它记每个系数是怎么来的、哪条结论作废了、哪些数字是 A100 实测不能直接搬到 GB10。docs/ 下其余文档是结论态，那份是过程态，没有它就不敢动已成型的公式。

## 为什么要建这套模型

三个具体问题，靠拍脑袋答不了，靠可计算模型才能答：

- **这台机器能跑哪个模型、什么精度**：27B 全量 54.6GB，121GB 内存装得下，但 320B 的 MoE 连量化到 nvfp4 都装不下。装不下和跑得动是两回事，中间隔着显存公式。
- **生成速度由什么决定**：实测 Qwen3.8-27B 在 GB10 上 bf16 4.3 t/s、量化 12.1 t/s，差 2.8 倍。这个差距能由「带宽 ÷ 权重字节」一条公式预测到 7% 以内，说明 decode 是带宽瓶颈，和算力无关。
- **外推到没测过的组合**：换个硬件（4090）、换个模型（320B MoE）、换个上下文（1M），不用重新实测，代入公式就有量级正确的预测。

## 七个理论模型

每个模型三件套：公式推导 + 自测脚本（`scripts/*_model.py`）+ 文档（`docs/*.md`，含实测对比）。

| 模型 | 回答的问题 | 核心公式 | 文档 |
|---|---|---|---|
| Roofline | 速度上限、瓶颈判断、外推 | tok/s = 带宽 × 效率 / 权重字节 | `docs/Roofline模型.md` |
| 权重显存 | 装不装得下、量化怎么压 | 显存 = 参数量 × 字节/参数 | `docs/权重显存模型.md` |
| KV Cache | 长上下文内存成本、最大上下文 | KV = 2×层数×KV头×head_dim×长度×batch | `docs/KV Cache显存模型.md` |
| 注意力复杂度 | 上下文拉长的瓶颈、解法取舍 | 算力 O(S²)、FlashAttention 显存 O(S) | `docs/注意力复杂度模型.md` |
| 吞吐规模化 | 单流 vs 服务吞吐、batch 拐点 | 吞吐 ≈ batch × 单流，受算力天花板限制 | `docs/吞吐规模化模型.md` |
| Scaling Laws | 规模的预期能力、算力最优配比 | D=20N（Chinchilla）、L=(Nc/N)^α | `docs/Scaling Laws模型.md` |
| 量化算法 | 有哪些量化方案、对速度和智力的影响 | 共享粒度（scale）、weight-only vs 激活量化 | `docs/量化算法与精度模型.md` |
| 外推架构依赖 | 外推和模型结构/参数/架构的关系 | 架构决定公式，参数决定数值 | `docs/外推的模型架构依赖性.md` |
| 内存层级与卸载 | 权重放哪层内存、统一内存 vs 分层 | 有效带宽=权重所在层带宽 | `docs/内存层级与卸载模型.md` |
| 模型加载与IO | 冷启动时间、页缓存、分片并行 | 加载=文件大小/磁盘读带宽 | `docs/模型加载与IO模型.md` |
| KV分页与前缀共享 | 真实服务的KV显存修正 | 分页碎片+前缀共享收益 | `docs/KV Cache分页与前缀共享模型.md` |
| 推测解码 | 投机采样的加速比 | E=(1-α^(k+1))/(1-α) | `docs/推测解码加速模型.md` |

前五个管「跑不跑得动、多快」（工程约束），第六个管「这个规模大致多强」（能力外推）。第七、八篇是横切性的：量化决定代入哪个字节/参数，外推架构依赖决定用哪个公式。后四篇补上真实部署里被前面理想模型忽略的四条假设：权重所在内存层级、模型加载时间、KV cache 的分页与共享、推测解码加速。四者结合才能回答「这台机器能跑的模型里，哪个最值得跑、用哪种量化、按哪个公式算、服务化怎么调」。

## 实测锚点

所有模型都用 DGX Spark 上的真实数据自测，锚点定义在 `scripts/llm_spec.py`：

| 项 | 值 | 来源 |
|---|---|---|
| 硬件 | GB10，121GB 统一内存，273 GB/s 带宽 | 社区公认 + 实测反推 |
| 模型 | Qwen3.8-27B，27.3B 参数，64 层，GQA 4 KV 头 | `config.json` |
| bf16 全量 | 50.9GB，decode 4.3 t/s | llama.cpp cuBLAS 实测 |
| q4_k_xl 量化 | 16.4GB，decode 12.1 t/s | llama.cpp cuBLAS 实测 |

关键验证：decode 公式 `tok/s = 带宽/权重字节` 对两个实测点的预测误差分别是 -7% 和 -6%，反推有效带宽 235/198 GB/s（占峰值 86%/73%），确认了带宽峰值和模型正确性。

## 复现

```bash
cd experiments/test_inference_theory
# 看硬件/模型规格与实测锚点
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/llm_spec.py
# 跑单个理论模型的自测
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/roofline_model.py
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/kv_cache_model.py
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/weight_memory_model.py
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/attention_model.py
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/throughput_model.py
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/scaling_law_model.py
```

## 和推理框架的关系

这套模型是框架无关的。它预测的是「给定硬件和模型配置，理论上限是多少」，不关心用 llama.cpp、vLLM 还是 SGLang 实现。实测框架能跑多接近理论上限，是另一个问题（效率项）。社区数据（2026-08 阿里云测速）显示 GB10 上 SGLang+NVFP4 单流约 23 t/s 接近理论上限，llama.cpp q4 约 12 t/s 是效率损失，vLLM 居中。
