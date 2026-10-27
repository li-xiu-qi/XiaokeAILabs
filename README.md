# 筱可 AI 工程实验仓库（XiaokeAILabs）

> 李秀奇（筱可）的 AI 工程实验集合。以「能跑、能复现、讲得清」为标准，沉淀 embedding / 检索 / 推理成本 / 多模态 / 判别式模型等方向的实验代码。

## 这是什么

- **定位**：AI 工程实验仓库，不是产品仓库。每个 `experiments/<name>` 是一个自包含实验。
- **标准**：能跑、能复现、结论经得起复核；配套文章发布在公众号「筱可AI」。
- **实验分场**：教学、单点验证、探索中的实验放在**私有实验区**；本仓只留**成型、可展示**的实验。

## 实验索引

### 检索与向量
- **embedding-research** — 评测基准 + 微调数据 + 相似度行为专项（句长 / 跨语言 / 代码表格 / NLI 迁移）+ 多模态（BGE-VL、CLIP） → `experiments/embedding-research`
- **retrieval-and-rerank** — 检索链路四环节：FAISS 稠密、BGE-M3 稀疏、ColBERT 晚期交互、BGE Reranker → `experiments/retrieval-and-rerank`
- **vector-db-benchmark** — Milvus / Qdrant / SurrealDB / LanceDB / ChromaDB / sqlite-vec 六库吞吐、延迟、召回实测 → `experiments/vector-db-benchmark`
- **text-segmentation** — 语义 + 结构混合分块、迟分等策略对比 → `experiments/text-segmentation`

### 推理成本与模型机制
- **inference-cost-models** — 12 个可计算的推理成本模型（Roofline、显存、KV Cache、吞吐、Scaling Laws、量化），GB10 实测锚点 → `experiments/inference-cost-models`
- **data-structure-costs** — 45 个数据结构 / 算法成本模型与实测（公式迁知识库，本仓留复现脚本） → `experiments/data-structure-costs`
- **llm-inference-perf** — 推理成本分阶段报告（MTP、DeepSeek 等） → `experiments/llm-inference-perf`
- **ninfer-gb10-inference** — NInfer 在 GB10 上的 27B W4A4 + DFlash2 + Vision 部署、百 TPS 复现与 SGLang 对照 → `experiments/ninfer-gb10-inference`
- **strata-dgx-inference** — Strata 在 GB10 上的 Qwen3.8-Flash-Next IQ3_S 部署与 llama.cpp 对照，生成速度 2.5 倍 → `experiments/strata-dgx-inference`
- **llm-mechanics** — 从模型加载到解码的全链路机制实操 → `experiments/llm-mechanics`

### 多模态与微调
- **jina-clip-openvino** — Jina CLIP 在 Intel CPU/NPU 上的正确性验证、驱动 A/B 与同机重复性能基准 → `experiments/jina-clip-openvino`
- **image-finetune** — LoRA 微调（图片）：人脸身份 LoRA，让底模画出一个它画不出来的人 → `experiments/image-finetune`

### 文档理解与 OCR
- **minicpm-ocr** — MiniCPM-OCR 1.0 解析 + Jev 一致性判别双模型在 aarch64 GB10 上的部署，绕开 vLLM 与 Paddle，闭环 2.5 秒 → `experiments/minicpm-ocr`

### 判别式模型
- **jev-open-source** — Jev 判别式模型三个开源复刻（Laya / Nimble / metask-jev）在 GB10 上的延迟与准确率实测 → `experiments/jev-open-source`

### 应用
- **deepsearch-google** — 深度搜索系统设计 + Google 实现 → `experiments/deepsearch-google`

## 免责声明
- 部分实验为开发 / 教学性质，可能存在失败案例，均为学习过程。
- 以公众号文章说明为准，代码仅供学习参考；生产环境使用前请充分测试。

## 许可证
[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## 联系
- 作者：li-xiu-qi（筱可）｜邮箱：lixiuqixiaoke@qq.com｜公众号：**筱可AI**
- 仓库：https://github.com/li-xiu-qi/XiaokeAILabs
- 实验室介绍页：https://li-xiu-qi.github.io/XiaokeAILabs-website/
