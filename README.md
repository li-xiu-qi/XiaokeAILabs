# XiaokeAILabs：AI 工程实验

[English](README.en.md) · [实验网站](https://li-xiu-qi.github.io/XiaokeAILabs/)

从模型训练与检索，到多模态和推理部署，我们用可检查的方法、代码和结果解释工程选择。这里既有教程和原理，也保留实测条件与结果边界，帮助你判断何时适用，以及如何在自己的任务上验证。项目开源并持续更新。

## 实验索引

- [Embedding 蒸馏](experiments/embedding-distillation/README.md)：用大模型分数训练小型向量模型；提供候选池准备脚本。
- [重排序蒸馏](experiments/reranker-distillation/README.md)：对照监督训练与教师分数蒸馏，了解固定候选上的排序变化。
- [ColBERT 蒸馏](experiments/colbert-distillation/README.md)：比较教师分数蒸馏与标签监督；本次蒸馏未超过监督基线。
- [多模态 Embedding 蒸馏](experiments/multimodal-embedding-distillation/README.md)：探索图文检索教师与轻量学生之间的知识迁移。
- [多模态重排序](experiments/multimodal-reranking/README.md)：探索图文候选的重排序，并查看匿名化评测结果。
- [GUI 定位微调](experiments/gui-grounding-finetuning/README.md)：训练截图与指令到坐标的映射；30 个留出样本不足以证明泛化提升。
- [合成数据训练文生图模型](experiments/image-generation-distillation/README.md)：用教师合成数据训练学生 LoRA；结果基于单名 AI 评审。
- [向量数据库基准](experiments/vector-db-benchmark/README.md)：历史数据缺少原始结果，不用于产品排名。
- [扩散语言模型速度对照](experiments/diffusion-language-models/README.md)：在三道问题上比较两种生成路径；输出长度与质量不同。
- [GB10 单槽与八槽推理](experiments/gb10-inference/qwen36-35b-ab/README.md)：同一模型和引擎下比较短请求的吞吐与完成时间。
- [LLM 推理性能](experiments/llm-inference-perf/README.md)：模型与推理配置的性能分析。
- [语言模型量化](experiments/llm-mechanics/quantization/README.md)：对照权重体积、困惑度与 CPU 速度；BF16 路径性能异常。
- [LLM 机制](experiments/llm-mechanics/README.md)：通过示例理解生成机制。
- [推理成本模型](experiments/inference-cost-models/README.md)：内存算例与理论模型，性能估算尚未经独立验证。
- [数据结构成本](experiments/data-structure-costs/README.md)：分析数据结构和数据格式的成本。
- [人像抠像](experiments/portrait-matting/README.md)：评估 30 张单帧照片的主体、边界与推理时间；不能代表视频性能。
- [表格结构识别](experiments/table-structure-recognition/README.md)：用一张合成表格观察裁剪与结构预测，不涵盖文字识别。
- [MiniCPM-OCR](experiments/minicpm-ocr/README.md)：文档识别部署与生成加速。
- [Jina CLIP OpenVINO](experiments/jina-clip-openvino/README.md)：比较 CPU 和 NPU 的正确性与特定负载性能。
- [Jev 开源方案评测](experiments/jev-open-source/README.md)：比较社区方案的小样本决策表现。

具体环境、数据集、模型许可和复现限制请查看各实验。部分实验依赖外部数据、模型或尚未公开的准备文件；已记录的测量只适用于相应条件。

## 许可证

仓库代码采用 [Apache 2.0](LICENSE)；第三方模型、数据和代码适用各自许可。
