# 筱可 AI 工程实验仓库（XiaokeAILabs）

> 李秀奇（筱可）的 AI 工程实验集合。以「能跑、能复现、讲得清」为标准，沉淀 embedding / 检索 / 推理成本 / 多模态 / 文档理解 / 判别式模型等方向的实验代码。

## 这是什么

- **定位**：AI 工程实验仓库，不是产品仓库。每个 `experiments/<name>` 是一个自包含实验。
- **标准**：能跑、能复现、结论经得起复核；配套文章发布在公众号「筱可AI」。
- **实验分场**：教学、单点验证、探索中的实验放在**私有实验区**；本仓只留**成型、可展示**的实验。

一个实验进本仓，需要同时满足四条：有一句话能说清的结论且附实测数据；换一台机器按 README 能从零跑通；仓内只放复现结论必需的脚本与证据，权重和数据集走下下载脚本或 Release；无内网地址、机器名与内部路径。

## 怎么读这个仓

每个实验的 `README.md` 开头就是结论，不铺垫。要判断某个实验值不值得看，读它前三段即可：核心发现、为什么值得做、实测数字。

想直接跑起来，进对应目录看「怎么跑起来」一节，那是按读者第一个动作组织的，从拉权重到出结果。

## 实验索引

### 检索与向量
- **embedding-research** — 评测基准 + 微调数据 + 相似度行为专项（句长 / 跨语言 / 代码表格 / NLI 迁移）+ 多模态（BGE-VL、CLIP） → `experiments/embedding-research`
- **retrieval-and-rerank** — 检索链路四环节：FAISS 稠密、BGE-M3 稀疏、ColBERT 晚期交互、BGE Reranker → `experiments/retrieval-and-rerank`
- **vector-db-benchmark** — Milvus / Qdrant / SurrealDB / LanceDB / ChromaDB / sqlite-vec 六库吞吐、延迟、召回实测 → `experiments/vector-db-benchmark`

### 推理成本与模型机制
- **inference-cost-models** — 12 个可计算的推理成本模型（Roofline、显存、KV Cache、吞吐、Scaling Laws、量化），GB10 实测锚点 → `experiments/inference-cost-models`
- **data-structure-costs** — 45 个数据结构 / 算法成本模型与实测（公式迁知识库，本仓留复现脚本） → `experiments/data-structure-costs`
- **llm-inference-perf** — 推理成本分阶段报告（MTP、DeepSeek 等） → `experiments/llm-inference-perf`
- **llm-mechanics** — 从模型加载到解码的全链路机制实操 → `experiments/llm-mechanics`

### GB10 单机部署
- **strata-dgx-inference** — Strata 在 GB10 上的 Qwen3.8-Flash-Next IQ3_S 部署与 llama.cpp 对照，生成速度 2.5 倍 → `experiments/strata-dgx-inference`
- **ninfer-gb10-inference** — NInfer 在 GB10 上的 27B W4A4 + DFlash2 + Vision 部署、百 TPS 复现与 SGLang 对照 → `experiments/ninfer-gb10-inference`
- **qwen36-35b-engine-ab** — Qwen3.6-35B-A3B 双引擎 A/B：llama.cpp MTP2 对 NInfer MTP3 / DFlash7，单请求 1.37 倍，C8 聚合 223.7 对 54.8 t/s → `experiments/qwen36-35b-engine-ab`
- **ds4-flash-2bit-gb10-inference** — DeepSeek-V4-Flash 2-bit（299B MoE）单卡复现：20.16 tok/s、权重常驻 94.69 GiB，MTP 为净负收益 → `experiments/ds4-flash-2bit-gb10-inference`

### 多模态与微调
- **jina-clip-openvino** — Jina CLIP 在 Intel CPU/NPU 上的正确性验证、驱动 A/B 与同机重复性能基准，NPU 吞吐达 CPU 的 12.57 倍 → `experiments/jina-clip-openvino`
- **image-finetune** — LoRA 微调（图片）：人脸身份 LoRA，让底模画出一个它画不出来的人 → `experiments/image-finetune`

### 文档理解与 OCR
- **minicpm-ocr** — MiniCPM-OCR 1.0 解析 + Jev 一致性判别双模型在 aarch64 GB10 上的部署，绕开 vLLM 与 Paddle，闭环 2.5 秒 → `experiments/minicpm-ocr`

### 判别式模型
- **jev-open-source** — Jev 判别式模型三个开源复刻（Laya / Nimble / metask-jev）在 GB10 上的延迟与准确率实测 → `experiments/jev-open-source`

## 目录约定

```
experiments/<name>/
├── README.md      结论先行，含实验对象、实测数据、怎么跑、边界
├── scripts/       复现必需的脚本，按执行顺序编号或按功能命名
├── docs/          evidence/ 放原始数据、证据与结论图表
└── assets/        图示与小体积样例
```

不放进来：模型权重（走下下载脚本）、数据集（写来源与获取方式）、虚拟环境、运行日志原文（只留结论与关键片段）、内部过程记录（部署实录与排错笔记写在私有实验区）。

## 运行环境

多数实验在 **DGX Spark（GB10，aarch64，121 GiB 统一内存）** 上完成，少数在 Intel CPU/NPU 与 x86_64 上。每个实验的 README 有独立的环境表，注明硬件、CUDA、torch 与关键依赖版本。

跨实验通用的一条：版本要钉死。渲染、量化、推理后端的版本漂移会静默改变数值结果，`minicpm-ocr` 与 `jina-clip-openvino` 两个实验都因此把关键依赖锁到具体版本。

## 免责声明

- 部分实验为开发 / 教学性质，可能存在失败案例，均为学习过程。
- 以公众号文章说明为准，代码仅供学习参考；生产环境使用前请充分测试。

## 许可证

[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## 联系

- 作者：li-xiu-qi（筱可）｜邮箱：lixiuqixiaoke@qq.com｜公众号：**筱可AI**
- 仓库：https://github.com/li-xiu-qi/XiaokeAILabs
- 实验室介绍页：https://li-xiu-qi.github.io/XiaokeAILabs-website/
