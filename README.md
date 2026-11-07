# XiaokeAILabs：AI 工程实验

[English](README.en.md) · [实验网站](https://li-xiu-qi.github.io/XiaokeAILabs/)

从模型训练与检索，到多模态和推理部署，我们用可检查的方法、代码和结果解释工程选择。这里既有教程和原理，也保留实测条件与结果边界，帮助你判断何时适用，以及如何在自己的任务上验证。项目开源并持续更新。

## 实验索引

- [中文 Embedding 微调](experiments/embedding-research/model_finetuning/test_train_embedding/README.md)：用查询与相关文档学习批内对比训练；内置数据仅四组，不证明检索效果提升。
- [重排序模型微调](experiments/reranker-finetune-scifact/README.md)：使用查询、文档和相关性标签训练交叉编码器。
- [Embedding 蒸馏](experiments/embedding-distillation/README.md)：用大模型分数训练小型向量模型；提供候选池准备脚本。
- [重排序蒸馏](experiments/reranker-distillation/README.md)：对照监督训练与教师分数蒸馏，了解固定候选上的排序变化。
- [ColBERT 蒸馏](experiments/colbert-distillation/README.md)：真实教师复验中蒸馏与监督组差距很小，单种子结果不足以确认优势。
- [视觉文档多向量检索](experiments/visual-multivector-retrieval/README.md)：文档页检索基线及合成页面上的微调对照；本次未观察到蒸馏收益。
- [多模态 Embedding 蒸馏](experiments/multimodal-embedding-distillation/README.md)：比较图文配对监督与教师蒸馏；100 条测试查询中，蒸馏组 R@1 为 0.91，监督组为 0.93。
- [多模态重排序](experiments/multimodal-reranking/README.md)：探索图文候选的重排序，并查看匿名化评测结果。
- [GUI 定位微调](experiments/gui-grounding-finetuning/README.md)：48 条分层留出样本中，基座命中 34、LoRA 命中 42；小样本不能保证跨应用泛化。
- [教师视频训练水波纹生成模型](experiments/video-generation-distillation/README.md)：对照学生训练前后，以及教师和学生自产训练素材；[小样本来源对照](experiments/video-generation-distillation/results/source-control/README.md)的抽帧评审未显示教师数据组优于基模。
- [合成数据训练文生图模型](experiments/image-generation-distillation/README.md)：比较教师与学生自产数据；相机开发集教师优势门槛未通过，本轮停止微调与最终测试。
- [人物图像生成微调](experiments/image-finetune/README.md)：用人物照片训练 LoRA；现有基线与微调结果分辨率不同，不能直接比较。
- [人物视频微调与排障](experiments/wan22-video-diagnostics/README.md)：检查视频基模、种子和 LoRA 生成结果；小规模留出评估未证明学会目标角色。
- [向量数据库基准](experiments/vector-db-benchmark/README.md)：历史数据缺少原始结果，不用于产品排名。
- [向量索引成本](experiments/vector-index-costs/README.md)：比较 Flat、IVF、HNSW 与量化索引的召回、耗时和空间。
- [语言模型 JEPA 训练通路](experiments/language-jepa/README.md)：小规模加法对照中，40 道测试题有 21 道与训练题交换顺序后等价；未证明能力提升。
- [扩散语言模型速度对照](experiments/diffusion-language-models/README.md)：在三道问题上比较生成路径；输出长度和内容不同，未评估输出质量。
- [GB10 单槽与八槽推理](experiments/gb10-inference/qwen36-35b-ab/README.md)：同一模型和引擎下比较短请求的吞吐与完成时间；每档仅两轮。
- [LLM 推理性能](experiments/llm-inference-perf/README.md)：模型与推理配置的性能分析。
- [语言模型量化](experiments/llm-mechanics/quantization/README.md)：对照权重体积、困惑度与 CPU 速度；BF16 路径性能异常。
- [LLM 机制](experiments/llm-mechanics/README.md)：通过示例理解生成机制。
- [Tokenizer 原理](experiments/tokenizer-mechanics/README.md)：从字节级 BPE 到真实编码与 token 计费。
- [KV Cache 原理](experiments/llm-mechanics/kv_cache/README.md)：对照缓存与全前缀计算的数值结果。
- [推理成本模型](experiments/inference-cost-models/README.md)：内存算例与理论模型，性能估算尚未经独立验证。
- [数据结构成本](experiments/data-structure-costs/README.md)：用教学模型探索索引与存储；历史格式记录不能作为选型依据。
- [人像抠像](experiments/portrait-matting/README.md)：评估 30 张单帧照片的主体、边界与推理时间；不能代表视频性能。
- [表格结构识别](experiments/table-structure-recognition/README.md)：用一张合成表格观察裁剪与结构预测，不涵盖文字识别。
- [文字检测与识别](experiments/paddle-ocr-detection-recognition/README.md)：将文字检测框交给识别模型；两阶段结果只覆盖一张合成表格。
- [MiniCPM-OCR](experiments/minicpm-ocr/README.md)：文档识别部署与生成加速。
- [Jina CLIP OpenVINO](experiments/jina-clip-openvino/README.md)：比较 CPU 和 NPU 的正确性与特定负载性能。
- [Jev 开源方案评测](experiments/jev-open-source/README.md)：比较社区方案的小样本决策表现。

具体环境、数据集、模型许可和复现限制请查看各实验。部分实验依赖外部数据、模型或尚未公开的准备文件；已记录的测量只适用于相应条件。

## 许可证

仓库代码采用 [Apache 2.0](LICENSE)；第三方模型、数据和代码适用各自许可。
