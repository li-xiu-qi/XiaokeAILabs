# XiaokeAILabs：AI 工程实战

<p align="center">
  <a href="https://li-xiu-qi.github.io/XiaokeAILabs/"><img src="https://img.shields.io/badge/官网-在线阅读-356552?style=flat-square&logo=googlechrome&logoColor=white" alt="在线网站"></a>
  <a href="https://github.com/li-xiu-qi/XiaokeAILabs/stargazers"><img src="https://img.shields.io/github/stars/li-xiu-qi/XiaokeAILabs?style=flat-square&logo=github&color=356552" alt="GitHub Stars"></a>
  <a href="https://github.com/li-xiu-qi/XiaokeAILabs/network/members"><img src="https://img.shields.io/github/forks/li-xiu-qi/XiaokeAILabs?style=flat-square&logo=github&color=356552" alt="GitHub Forks"></a>
  <a href="https://github.com/li-xiu-qi/XiaokeAILabs/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square" alt="License"></a>
  <a href="docs/zh/README.md"><img src="https://img.shields.io/badge/文档-中文全景-green?style=flat-square" alt="中文文档"></a>
</p>

<p align="center">
  <a href="https://li-xiu-qi.github.io/XiaokeAILabs/">🌐 在线展示网站</a> · 
  <a href="README.en.md">English README</a> · 
  <a href="docs/zh/README.md">📖 工程原理与实验系列文档</a>
</p>

---

训练一个检索模型到底有没有改善搜索？单台设备能部署多大的模型，处理多人请求时又会怎样？一项推理优化能节省多少时间，其结果是否依赖特定硬件？

XiaokeAILabs 围绕这些工业级问题开展开放实验，覆盖**检索与重排序、大模型训练、推理优化与单机极限、文档解析与决策模型**。这里既有完整的工程实现与实测数据，也有深入底层的原理解析与显存预算推导。希望它能帮助你在复杂方案间看清关键取舍，并在自己的业务数据和硬件上继续验证。

---

## 全项目目录导航

全项目采用双轨设计：**正文篇**聚焦成体系的工业级核心系统，**专题与原理篇**承载前沿机制推导与单点探索验证。

### 【正文篇】核心工程主干

#### 第一卷：检索与重排序（搜索与 RAG 核心）
从单向量召回、精细重排到细粒度多向量交互与工业级向量库选型。
- **第 1 章 · 双塔向量模型（Embedding）与交叉编码器（Reranker）**
  - 向量召回微调：Batch 内负采样与难负例挖掘 · [`embedding-finetuning`](experiments/embedding-finetuning/)
  - 交叉编码器重排序：二次精排与多模态图文排序 · [`reranker-finetuning`](experiments/reranker-finetuning/)
  - 模型蒸馏：将大型重排模型能力迁移至轻量小模型 · [`reranker-distillation`](experiments/reranker-distillation/)
- **第 2 章 · ColBERT 多向量与晚期交互**
  - 晚期交互原理：多向量表征与 MaxSim 细粒度匹配 · [`colbert-finetuning`](experiments/colbert-finetuning/)
  - 模型蒸馏：从单向量教师蒸馏多向量模型 · [`colbert-distillation`](experiments/colbert-distillation/)
  - 视觉多向量检索：页面截图端到端检索（ColPali 范式） · [`visual-multivector-retrieval`](experiments/visual-multivector-retrieval/)
- **第 3 章 · 向量索引与数据库实测**
  - 索引横向对比：Flat 暴力扫描、IVF 倒排聚类、HNSW 图索引与量化压缩
  - 数据库选型与实测：嵌入式与独立服务端方案评测 · [`vector-db-benchmark`](experiments/vector-db-benchmark/)

#### 第二卷：大语言模型训练（从零走通完整生命周期）
剥离黑盒封装，从因果架构搭建、预训练算力规划一路跑到指令微调与偏好对齐。
- **第 4 章 · 现代大模型架构**
  - 因果架构演进：RoPE 旋转位置编码、RMSNorm 与 SwiGLU 激活
  - 注意力机制：从多头 MHA、分组共享 GQA 到低秩潜变量 MLA
  - 混合专家（MoE）：Top-K 专家路由、共享专家与负载均衡
- **第 5 章 · 预训练规律与算力规划**
  - 扩展律（Scaling Laws）：算力、数据量与模型参数的最优配比 · [`llm-lifecycle-fullchain`](experiments/llm-lifecycle-fullchain/)
  - 训练账本换算：Step、Token、GPU 小时与硬件利用率（MFU）
  - Muon 优化器：矩阵正交化更新机理与训练收敛实测
  - 基座模型评测：为什么 Base 模型不直接考对话，如何用连续似然打分
- **第 6 章 · 模型微调与对齐**
  - 指令微调（SFT）：损失掩码（Loss Masking）与高质量数据构造
  - 参数高效微调（LoRA）：低秩适配器机理、秩的选取与防遗忘
  - 偏好对齐（DPO）：免奖励模型的直接偏好优化 · [`dpo-preference-alignment`](experiments/dpo-preference-alignment/)

#### 第三卷：大语言模型推理优化与单机极限（让模型跑得更快更省）
突破内存带宽墙，从显存预算、投机解码、低比特量化到单机极限并发与双卡协作。
- **第 7 章 · 自回归解码与 KV Cache**
  - 生成双阶段：算力受限的 Prefill 与带宽受限的 Decode
  - KV Cache：显存换时间的计算图流转与注意力头压缩
  - 推理显存预算：权重、KV 缓存与长并发显存定量测算 · [`inference-cost-models`](experiments/inference-cost-models/)
- **第 8 章 · 投机解码加速**
  - 协同推测原理：草稿模型先行推测、目标大模型矩阵核验 · [`llm-inference-perf`](experiments/llm-inference-perf/)
  - 树状验证（Tree Attention）：因果掩码设计与单步校验多分支
  - 投机加速实测：为什么理论接受率高，实际端到端反而可能变慢？
- **第 9 章 · 模型量化与剪枝**
  - 量化等级与精度变化：从 FP16 到 INT4/NVFP4，4-bit 的精度拐点
  - 激活感知压缩：Wanda 激活剪枝、NVIDIA 2:4 硬件稀疏与 AWQ 权重量化
- **第 10 章 · 硬件实战与并发调度**
  - 单卡极限压测：299B MoE 极端量化与 35B 双引擎吞吐对比 · [`moe-2bit-fit`](experiments/moe-2bit-fit/) · [`engine-slot-concurrency`](experiments/engine-slot-concurrency/)
  - 统一内存架构实测：GB10 平台上的 NInfer、Strata 与 llama.cpp 并发 · [`ninfer-serving`](experiments/ninfer-serving/) · [`strata-arm64`](experiments/strata-arm64/)
  - 双卡流水线并行：两台设备间张量切分、网络传输与 MoE 通信瓶颈 · [`pipeline-parallel`](experiments/pipeline-parallel/)

#### 第四卷：文档理解与表格还原（多模态真实输入前端）
解决现实复杂文档进入 AI 系统的“第一公里”：从 PDF、扫描件与报表中抽取高质量结构化数据。
- **第 11 章 · 文档版面与文字识别（OCR）**
  - PDF 解析：规则抽取与矢量元素提取的适用边界 · [`pdf-rule-parsers`](experiments/pdf-rule-parsers/)
  - PaddleOCR 落地：文本行检测（DBNet）与识别（SVTR） · [`paddle-ocr-detection-recognition`](experiments/paddle-ocr-detection-recognition/)
- **第 12 章 · 表格结构识别与还原**
  - SLANet 表格识别：从图像到有线/无线表格 HTML 单元格重组 · [`table-structure-recognition`](experiments/table-structure-recognition/)
  - 方案对比：传统模块化拼接流水线 vs 端到端多模态大模型 · [`minicpm-ocr`](experiments/minicpm-ocr/)

#### 第五卷：判别式决策与偏好系统（用模型做精准评估）
超越逐字聊天：将模型概率转化为置信度明确的打分、排序与决策信号。
- **第 13 章 · 决策模型打分机制**
  - 决策模型与生成模型：直接输出标量分数与逐字采样的本质差异
  - 决策模型微调与置信度校准（概率校准） · [`decision-model-finetuning`](experiments/decision-model-finetuning/)
- **第 14 章 · 决策模型选型与实测**
  - 多款判别式决策模型基准对比与工程选型建议 · [`decision-model-benchmark`](experiments/decision-model-benchmark/)

---

### 【专题与原理篇】独立前沿与探索验证
针对前沿表征理论、单点排查与探索性小实验的独立专篇，作为主干外部的延伸读物。

1. **文本切分与分词机制（Tokenizer）**
   - UTF-8 字节编码如何合并成 token，BPE、WordPiece 与 Unigram 算法对比 · [`tokenizer-mechanics`](experiments/tokenizer-mechanics/)
   - 中文 Token 税与跨词表 BPB 衡量标准
2. **迟分词（Late Chunking）长文本嵌入**
   - 传统滑动窗口信息碎裂问题；先整篇全注意力编码、后池化切片的数学与几何机理
3. **对比学习表征几何（SimCSE）**
   - 高维语义向量各向异性（Anisotropy）退化；对齐度与均匀度的几何直觉 · [`contrastive-learning-simcse`](experiments/contrastive-learning-simcse/)
4. **端侧异构推理探测：Intel NPU 跑多模态嵌入**
   - OpenVINO 在 Intel NPU 上的驱动边界与吞吐表现 · [`jina-clip-openvino`](experiments/jina-clip-openvino/)
5. **训练工程排查：填充长度对损失的干扰**
   - 固定输入下，注意力掩码与首层精度检查；填充长度为何导致 Loss 异常波动
6. **图像生成与抠像轻量实验**
   - 人脸身份 LoRA 微调与量化评估 · [`image-finetune`](experiments/image-finetune/)
   - 极速人物抠像推理测速 · [`portrait-matting`](experiments/portrait-matting/)

---

## 如何使用

所有实验均在真实硬件上完成验证并附带复现脚本。每项实验包含：
- **`README.md`**：实验目的、软硬件环境配置、执行步骤与关键结论；
- **`evidence/`**：原始运行日志、指标 JSON 文件与图表资产；
- **`scripts/`**：开箱即用的复现脚本与自动化测试流水线。

深入阅读原理与选型指南，参见 [工程原理与实验系列文档](docs/zh/README.md)。

---

## Star History

<p align="center">
  <a href="https://star-history.com/#li-xiu-qi/XiaokeAILabs&Date">
    <img src="https://api.star-history.com/svg?repos=li-xiu-qi/XiaokeAILabs&type=Date" alt="Star History Chart" width="100%" style="max-width: 800px;">
  </a>
</p>
