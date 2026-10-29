# 筱可 AI 工程实验仓库 (XiaokeAILabs)

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-DGX%20Spark%20GB10%20(aarch64)-76B900?style=flat-square&logo=nvidia" alt="Hardware">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.14%2Bcu130-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License">
  <a href="https://li-xiu-qi.github.io/XiaokeAILabs-website/"><img src="https://img.shields.io/badge/Website-XiaokeAILabs-purple?style=flat-square" alt="Website"></a>
</p>

> **李秀奇（筱可）的 AI 工程实证仓库。以「能跑、能复现、讲得清」为准则。**  
> 专注沉淀 **大模型推理成本精算、新硬件极限落地（GB10/NPU）、文档解析（OCR）、向量数据库与检索全链路** 的第一手实测代码与基准数据。配套技术拆解同步发布于公众号 **「筱可AI」**。

---

## 📌 实验室原则与定位

- **工程实证展厅**：不放未收敛的半成品；所有训练、调优、试错在私有侧收敛后，仅将成型、可公开验证的实验代码与数据沉淀入仓。
- **结论先行与数据锚点**：每个实验均提供真实测试数据锚点，拒绝无实测支撑的概念推导。
- **环境版本锁定**：关键软件栈、CUDA 算子与依赖版本严格钉死，杜绝版本漂移导致的性能与精度失真。
- **纯粹开源轻量**：大模型权重与原始数据集走标准下载脚本或官方 Release，仓库本体只保留复现必需的脚本、配置与轻量证据。

---

## 🚀 核心实验全景看板

| 领域方向 | 核心项目 / 系列 | 核心验证与实测结论摘要 | 硬件基准 | 源码入口 |
| :--- | :--- | :--- | :--- | :---: |
| **文档智能与 OCR** | **MiniCPM-OCR 1.0 + Jev 全链路** | 绕开 vLLM/Paddle 在 aarch64 跑通原生推理；DSpark 投机加速达 **2.36x**，端到端质检闭环 **2.5s** | GB10 (aarch64) | [直达 ➔](experiments/minicpm-ocr) |
| **GB10 极限推理** | **单卡 299B MoE 与双引擎 A/B 系列** | 299B MoE 2-bit 单卡常驻 94.69G (**20.16 tok/s**)；NInfer 移植跑通百 TPS；Qwen3.6-35B 并发吞吐 **223.7 tok/s**；Strata 提速 **2.5x** | GB10 (121GB 统一内存) | [GB10 总组 ➔](experiments/gb10-inference) |
| **向量存储与检索** | **Vector DB Benchmark (6 库评测)** | Milvus / Qdrant / LanceDB 等 6 库真实数据实测：索引矩阵、存储三段拆解、内存与 BM25 混合检索 | Linux / x86_64 | [直达 ➔](experiments/vector-db-benchmark) |
| **检索增强全链路** | **Retrieval & Rerank** | FAISS 稠密 + BGE-M3 词项稀疏 + ColBERT 晚期交互 + BGE Reranker 逐层递进实操 | Linux / CUDA | [直达 ➔](experiments/retrieval-and-rerank) |
| **理论成本模型** | **Inference & Structure Cost Models** | 12 个可计算推理成本模型（Roofline/KV Cache/吞吐）+ 37 个数据结构成本公式，GB10 锚点实测反推 | 通用 / GB10 | [推理成本](experiments/inference-cost-models) / [数据结构](experiments/data-structure-costs) |
| **端侧算力与驱动** | **Jina CLIP OpenVINO** | Intel CPU/NPU 驱动 A/B 实测：定位 NaN 驱动陷阱，NPU 吞吐达 CPU 的 **12.57x** | Intel Core Ultra NPU | [直达 ➔](experiments/jina-clip-openvino) |
| **基础微调与视觉** | **Image LoRA & Jev 判别式** | 人脸身份特征保真 LoRA 训练；Jev 判别式模型三大开源复刻（Laya/Nimble/metask）实测 | Linux / GB10 | [LoRA 微调](experiments/image-finetune) / [Jev 开源](experiments/jev-open-source) |

---

## ⚠️ 经典工程陷阱与负向结论 (Post-mortems)

工程实测中最有价值的往往不是正向优化，而是**证伪与避坑**。本仓记录了多个在特定场景下被推翻的假设与实测陷阱：

1. **vLLM 0.23 的 MTP 投机解码负收益陷阱（-33%）**：
   - 在 DeepSeek-V4-Flash 2-bit (299B MoE) 实验中，非投机解码达 20.16 tok/s，启用官方 MTP (K=2) 后生成速度反而暴跌至 13.42 tok/s。
   - **根因定位**：vLLM 0.23.0 拒绝了 drafter 的 FULL cudagraph 构建，导致草稿模型反复发生 Host-Device 同步开销，吞噬了所有投机收益。
2. **Intel NPU 的驱动 NaN 隐形变量**：
   - 在 Core Ultra 7 155H NPU 上部署 Jina CLIP FP16 时，旧驱动（32.0.100.4023）文本输出全部为 NaN，视觉分支却完全正常。
   - **避坑启示**：设备侧异常切忌盲目改动模型结构，升级驱动至 32.0.100.5540 后即恢复正常，NPU 吞吐达 CPU 的 12.57 倍。
3. **LoRA 简化的“伪适配”陷阱**：
   - 为规避端侧算子不支持而关闭 `retrieval.query` 对应的 LoRA 权重，模型虽能顺利转换运行，但向量余弦相似度暴跌至 0.9166~0.9384，失去检索判别力。格式能转不等于语义可用。
4. **混合检索评测的双重数据错位**：
   - 在多库 BM25 + 向量检索评测中，排查出传统基准常见的两大缺陷：“查询文本与向量索引错位”及“相关度阈值未校准”，由此确立了双 Ground Truth 验证标准。

---

## 🔬 重点实验深度剖析

### 1. MiniCPM-OCR 双模型落地与 DSpark 投机加速
> 完整报告见 [experiments/minicpm-ocr](experiments/minicpm-ocr)

- **工程绕障**：官方主线强依赖 vLLM 与 Paddle 布局检测，在 aarch64 GB10 架构下存在构建死锁。本实验验证了纯原生 `transformers` + `PP-DocLayoutV3` torch provider 的可行性，成功打通 A4 全页解析流水线。
- **Jev 无参考质检打分**：传统 OCR 质量依赖人工抽查或带真值的 CER。Jev 采用「原图 + 识别文本 + 文本渲染图」三元组机制，无需 Ground Truth 即可输出 0~1 的置信一致性得分。实测与官方公布指标偏差 ≤ 0.0045，单块质检排序闭环耗时仅 **2.5 秒**（显存峰值仅 5.18 GiB）。
- **DSpark 投机加速突破**：挂载 0.3B 5 层双向块草稿模型 + 低秩马尔可夫状态转移头（`block_size = 15`），通过严格的 18 层 GatedDeltaNet 线性注意力状态精确回滚，达成 **100% 逐 Token 贪婪一致**：
  - **解码耗时降低 57.6%**（整体加速 **2.36x**，多行数学公式块最高达 **3.08x**）；
  - **平均吞吐从 73.83 tok/s 跃升至 174.32 tok/s**（最高 236.22 tok/s）；
  - **主模型前向步数缩减 87.2%**，单步平均接受长度 $\tau = 9.18$ tok/step。

---

### 2. GB10 单机大模型极限部署与引擎对比系列
> 完整报告见 [experiments/gb10-inference](experiments/gb10-inference)

在统一内存架构（NVIDIA GB10, aarch64, 121.6 GiB）上，系统化验证超大参数 MoE 单卡可行性、引擎底层移植及投机解码收益边界：
- [**DeepSeek-V4-Flash 2-bit (299B MoE) 单卡落地**](experiments/gb10-inference/ds4-flash-2bit)：
  - 成功跑通 101 GB 权重分片加载与推理，常驻内存 94.69 GiB，非投机解码达 **20.16 tok/s**（49.61 ms/tok）。
- [**NInfer GB10 移植与百 TPS 复现**](experiments/gb10-inference/ninfer-27b)：
  - 突破官方源码限定 `sm_120a` 导致在 GB10 (`sm_121a`) 上构建被拒的问题，完成 ARM64 与统一内存移植（898/898 构建目标全通）；
  - 实测 Qwen3.8-27B (W4A4 + DFlash2) 在 C8 并发下达成 **132.1~223.7 tok/s** 聚合吞吐，并证实统一内存环境下 **NVFP4 W4A4 实际吞吐全面超越 FP8**。
- [**Qwen3.6-35B-A3B 双引擎 A/B**](experiments/gb10-inference/qwen36-35b-ab)：
  - 同机同模型压测，NInfer MTP3 单请求达 llama.cpp 的 **1.37 倍**；并发扩展性实测中，NInfer 在 C8 并发下聚合吞吐达 **223.7 tok/s**（相比 llama.cpp 单槽位限制的 54.8 tok/s 带来跨数量级优势）。
- [**Strata 统一内存推理**](experiments/gb10-inference/strata-dgx)：
  - Qwen3.8-Flash-Next IQ3_S 开启稀疏激活与专家缓存，相比基准 llama.cpp 提速 **2.5 倍**。

---

### 3. 向量数据库真实基准：6 库 × 索引矩阵 × 存储三段拆分
> 完整报告见 [experiments/vector-db-benchmark](experiments/vector-db-benchmark)

- **评测对象与真实数据**：同一硬件下，基于 20 Newsgroups 全量 11,293 篇文本（all-MiniLM-L6-v2 真实 384 维向量），横向对比 Milvus 3.0、Qdrant 1.19、SurrealDB 3.2.4、LanceDB 0.38、ChromaDB 1.5.9、sqlite-vec 0.1.9。
- **存储三段深度拆解**：严密剥离计量「向量自身 / 原始正文 / 索引开销」：
  - **LanceDB 存储开销最小**：列式存储压缩优势显著；
  - **Milvus IVF_SQ8 效率最高**：为兼顾延迟与召回的最优 ANN 索引形态；
  - **SurrealDB 向量成本最高**：通用 RocksDB 文档模型在稠密向量场景产生较大存储膨胀；
  - **Qdrant 预分配机制**：落盘主要受固定页预分配支配，单条正文增量难以测出。

---

### 4. 理论成本模型与实测锚点反推
> 完整报告见 [experiments/inference-cost-models](experiments/inference-cost-models) ｜ [experiments/data-structure-costs](experiments/data-structure-costs)

- **12 个可计算推理成本模型**：输入硬件规格与模型配置，即可递推显存、带宽、吞吐上限（包括 Roofline 瓶颈边界、KV Cache 显存、注意力算力、吞吐规模化、量化影响与投机加速比）。
- **实测锚点与有效带宽反推**：
  - 在 GB10 上测试 Qwen3.8-27B，bf16 全量 (4.3 tok/s) 与 q4_k_xl 量化 (12.1 tok/s) 的速度差异，通过 `tok/s = 带宽 / 权重字节` 理论预测误差在 **7% 以内**；
  - 反推得出统一内存有效显存带宽为 235 / 198 GB/s（达标称峰值的 86% / 73%），印证了 Decode 阶段纯带宽瓶颈本质；
  - 修正了 48 层 linear-attention 导致传统 KV Cache 模型误算显存的缺陷。
- **37 个底层数据结构成本模型**：覆盖 B+ 树、LSM-tree、列存、倒排索引、BM25 打分、Count-Min Sketch、HLL、Swiss Table 等，均包含纯数学公式计算器与自测验证脚本。

---

## ⚡ 场景化复现卡片 (Recipes)

克隆本仓库后，可通过以下指令快速体验三大经典实验：

### 🍰 Recipe 1：运行 MiniCPM-OCR DSpark 投机加速基准
```bash
cd experiments/minicpm-ocr
# 评测 4 组多样样本的非加速 vs DSpark 投机加速解码耗时与一致性
python scripts/benchmark_acceleration.py
```

### 🍰 Recipe 2：验证 GB10 理论推理速度与有效带宽反推
```bash
cd experiments/inference-cost-models
# 查看真实硬件锚点与理论公式预测误差
python scripts/roofline_model.py
python scripts/kv_cache_model.py
```

### 🍰 Recipe 3：生成 6 大向量数据库的存储与延迟全景对比图
```bash
cd experiments/vector-db-benchmark
# 依据已有实测结果 JSON 重新渲染 7 张标准对比报告图表
python figures/make_figures.py
```

---

## 📚 全量实验索引

### 🔍 检索与向量
- [**embedding-research**](experiments/embedding-research) — 评测基准 + 微调数据 + 相似度行为专项（句长 / 跨语言 / 代码表格 / NLI 迁移）+ 多模态（BGE-VL、CLIP）
- [**retrieval-and-rerank**](experiments/retrieval-and-rerank) — 检索链路四环节：FAISS 稠密、BGE-M3 稀疏、ColBERT 晚期交互、BGE Reranker
- [**vector-db-benchmark**](experiments/vector-db-benchmark) — Milvus / Qdrant / SurrealDB / LanceDB / ChromaDB / sqlite-vec 六库吞吐、延迟、召回实测

### ⚙️ 推理成本与模型机制
- [**inference-cost-models**](experiments/inference-cost-models) — 12 个可计算的推理成本模型（Roofline、显存、KV Cache、吞吐、Scaling Laws、量化），GB10 实测锚点
- [**data-structure-costs**](experiments/data-structure-costs) — 45 个数据结构与算法成本模型与实测复现脚本
- [**llm-inference-perf**](experiments/llm-inference-perf) — 推理成本分阶段报告（MTP、DeepSeek 等）
- [**llm-mechanics**](experiments/llm-mechanics) — 从模型加载到解码的全链路机制实操

### ⚡ GB10 单机大模型极限推理
- [**gb10-inference**](experiments/gb10-inference) — **GB10 系列实验总组**：
  - [ds4-flash-2bit](experiments/gb10-inference/ds4-flash-2bit) — DeepSeek-V4-Flash 2-bit（299B MoE）单卡复现：20.16 tok/s、常驻 94.69 GiB
  - [ninfer-27b](experiments/gb10-inference/ninfer-27b) — NInfer 在 GB10 上的 27B W4A4 + DFlash2 架构部署、百 TPS 复现与 SGLang 对照
  - [qwen36-35b-ab](experiments/gb10-inference/qwen36-35b-ab) — Qwen3.6-35B-A3B 双引擎 A/B：llama.cpp 对 NInfer MTP3 / DFlash7
  - [strata-dgx](experiments/gb10-inference/strata-dgx) — Strata 在 GB10 上的 Qwen3.8-Flash-Next IQ3_S 部署与 llama.cpp 对照

### 🖼️ 多模态与微调
- [**jina-clip-openvino**](experiments/jina-clip-openvino) — Jina CLIP 在 Intel CPU/NPU 上的正确性验证、驱动 A/B 与性能基准，NPU 达 CPU 的 12.57 倍
- [**image-finetune**](experiments/image-finetune) — LoRA 微调（图片）：人脸身份保真 LoRA 训练与评估

### 📄 文档理解与判别式模型
- [**minicpm-ocr**](experiments/minicpm-ocr) — MiniCPM-OCR 1.0 解析 + Jev 一致性判别双模型部署，闭环 2.5 秒
- [**jev-open-source**](experiments/jev-open-source) — Jev 判别式模型三个开源复刻（Laya / Nimble / metask-jev）在 GB10 上的延迟与准确率实测

---

## 📂 目录结构约定

每个实验独立自包含，遵循标准化布局：

```text
experiments/<experiment-name>/
├── README.md       # 结论先行，包含实验对象、实测数据指标、复现指南
├── scripts/        # 完整可复现脚本，按执行顺序编号或按功能命名
├── docs/           # evidence/ 放原始数据、证据与结论图表
└── assets/         # 结构图示与测试样张
```

---

## 🖥️ 统一运行环境

- **主力评测机**：**DGX Spark**（NVIDIA GB10, aarch64, 121.6 GiB 统一内存，273 GB/s 带宽）
- **辅助机型**：Intel Core Ultra 7 155H (CPU/NPU) 及标准 x86_64 Linux 节点
- **基线软件栈**：Ubuntu 24.04, CUDA 13.0.88, PyTorch 2.14+, Python 3.12
- **依赖声明**：渲染、量化与推理后端的版本漂移可能影响实测指标，关键实验均已锁定依赖版本（详见各实验目录说明）。

---

## ⚠️ 免责声明

- 本仓实验主要面向工程实证与技术探索，代码及基准数据仅供技术交流与学习参考。
- 实际生产环境应用前，请结合具体业务场景、数据分布与硬件架构进行充分验证。

---

## 📜 许可证

本项目遵循 [Apache License 2.0](LICENSE)。

---

## 🤝 联系与交流

- **作者**：李秀奇（筱可）
- **微信公众号**：**筱可AI**（分享硬核 AI 工程落地实践、架构复盘与评测）
- **邮箱**：lixiuqixiaoke@qq.com
- **实验室官网**：[https://li-xiu-qi.github.io/XiaokeAILabs-website/](https://li-xiu-qi.github.io/XiaokeAILabs-website/)
- **GitHub 仓库**：[https://github.com/li-xiu-qi/XiaokeAILabs](https://github.com/li-xiu-qi/XiaokeAILabs)
