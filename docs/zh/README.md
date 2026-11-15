# 实验地图：从工程问题到原理推导与可复现验证

> **阅读导航**：[English](../en/README.md) · [项目首页](../../README.md)

本项目按 AI 工程落地中的真实问题组织体系化知识与可复现实验。想理解机制与设计权衡，进入「讲解」；想查看具体配置、训练脚本与可复核证据，进入「代码与复现」。

文档体系分为六大核心方向：
- **检索与排序**：单向量 Embedding、重排序、ColBERT 多向量晚期交互、向量索引与数据库
- **推理与大模型机理**：完整因果预训练、指令微调与 LoRA、自回归与投机解码机制、显存成本与 Roofline 模型、新型架构
- **文档理解与信息提取**：规则与矢量解析、文字识别（OCR）、表格结构组装、MiniCPM 端到端解析
- **决策模型与判别式打分**：选项概率打分机制、专用头与 Logits 约束评测、RLCD 软标签微调、全模态决策
- **视觉与多模态**：界面定位（GUI Grounding）、高精度抠像、生成模型受控微调与量化评估
- **数据结构与存储**：基础数据结构理论成本模型、真实数据格式读写与压缩开销

---

## 一、 检索与排序

从海量文档中召回候选，再精准排定相关顺序。先看 [检索与排序导览](retrieval/README.md)。

| 工程问题 | 主线讲解 | 代码与复现实验 |
|---|---|---|
| 预训练模型句向量为何各向异性塌缩？对比学习两大几何准则是什么？ | [对比学习与表征几何](retrieval/embedding/对比学习与表征几何.md) | [embedding-finetuning](../../experiments/embedding-finetuning/README.md) |
| 无标注时对比学习的正例从哪来？dropout 为什么能当数据增强？ | [SimCSE 对比学习](retrieval/embedding/SimCSE对比学习.md) | [contrastive-learning-simcse](../../experiments/contrastive-learning-simcse/README.md) |
| 单向量 Embedding 如何监督微调？ | [Embedding 监督微调](retrieval/embedding/Embedding监督微调.md) | [embedding-finetuning](../../experiments/embedding-finetuning/README.md) |
| 单向量模型如何从强教师蒸馏？ | [Embedding 模型蒸馏](retrieval/embedding/Embedding蒸馏.md) | [embedding-distillation](../../experiments/embedding-distillation/README.md) |
| 图文如何直接做跨模态向量检索？ | [多模态 Embedding 蒸馏](retrieval/embedding/多模态Embedding蒸馏.md) | [multimodal-embedding](../../experiments/multimodal-embedding/README.md) · [蒸馏](../../experiments/multimodal-embedding-distillation/README.md) |
| 交叉编码器重排序如何微调与评估？ | [重排序监督微调](retrieval/reranking/重排序监督微调.md) | [reranker-finetuning](../../experiments/reranker-finetuning/README.md) |
| 重排序模型如何学习教师排序偏好？ | [重排序蒸馏](retrieval/reranking/重排序蒸馏.md) | [reranker-distillation](../../experiments/reranker-distillation/README.md) |
| 图文多模态候选如何重排？ | [多模态重排序](retrieval/reranking/多模态重排序.md) | [multimodal-reranking](../../experiments/multimodal-reranking/README.md) |
| ColBERT 多向量如何通过 MaxSim 匹配？ | [晚期交互与多向量检索](retrieval/colbert/晚期交互与多向量检索.md) · [监督微调](retrieval/colbert/文本多向量检索模型微调.md) | [colbert-finetuning](../../experiments/colbert-finetuning/README.md) |
| 多向量模型能否从单向量教师蒸馏？ | [文本多向量检索模型蒸馏](retrieval/colbert/文本多向量检索模型蒸馏.md) | [colbert-distillation](../../experiments/colbert-distillation/README.md) |
| 文档页面图像能否直接端到端多向量检索？ | [视觉多向量检索](retrieval/colbert/视觉多向量检索.md) | [visual-multivector-retrieval](../../experiments/visual-multivector-retrieval/README.md) |
| Flat、IVF、HNSW 与量化索引如何权衡？ | [向量索引与近邻搜索](retrieval/vector-search/向量索引与近邻搜索.md) | [工业级向量数据库评测](../../experiments/vector-db-benchmark/README.md) |
| 嵌入式与服务端向量数据库如何选型与评测？ | [向量数据库架构与选型](retrieval/vector-search/向量数据库架构与选型.md) · [评测方法](retrieval/vector-search/向量数据库怎么测才可比.md) | [六种向量检索方案：索引、召回与存储对比](../../experiments/vector-db-benchmark/README.md) |

---

## 二、 推理与大模型机理

从完整训练流程、指令微调、单步解码、显存公式到单卡极限推理与投机解码。先看 [推理与机理导览](inference/README.md)。

| 工程问题 | 主线讲解 | 代码与复现实验 |
|---|---|---|
| 如何从零搭建 BPE 分词器、因果现代架构并跑通预训练到生成？ | [大语言模型从零构建与全链路训练](inference/大语言模型从零构建与训练全链路.md) | [llm-lifecycle-fullchain](../../experiments/llm-lifecycle-fullchain/README.md) |
| 训练了多少 step、token 和 GPU 小时，分别说明什么？ | [模型训练的计量单位](inference/模型训练的计量单位.md) | [llm-lifecycle-fullchain](../../experiments/llm-lifecycle-fullchain/README.md) |
| SFT 为什么必须设置损失掩码？LoRA 怎么选秩与目标层？如何防范遗忘？ | [大语言模型指令微调与 LoRA](inference/大语言模型指令微调与LoRA.md) | [llm-lifecycle-fullchain](../../experiments/llm-lifecycle-fullchain/README.md) |
| 如何用偏好回答对直接优化对话模型？ | [DPO：直接偏好优化](alignment/DPO直接偏好优化.md) | [dpo-preference-alignment](../../experiments/dpo-preference-alignment/README.md) |
| 分词机制与主流词表算法如何影响有效上下文与生成成本？ | [分词机制与生成成本](inference/分词机制与生成成本.md) · [训练与规模权衡](inference/分词器训练与词表规模权衡.md) · [效果评测与 BPB](inference/分词器效果评测与跨词表BPB计算.md) | [tokenizer-mechanics](../../experiments/tokenizer-mechanics/README.md) |
| 预训练阶段如何定配算力与数据规模？Base 模型怎么科学评测？ | [预训练原理与扩展律](inference/大模型预训练原理与扩展律.md) · [预训练阶段评测与能力基准](inference/预训练阶段评测与能力基准.md) | [llm-lifecycle-fullchain](../../experiments/llm-lifecycle-fullchain/README.md) |
| 自回归单步解码怎么算？KV Cache 省了什么？ | [语言模型生成机制与 KV 缓存](inference/语言模型生成机制与KV缓存.md) | [llm-mechanics](../../experiments/llm-mechanics/README.md) |
| 投机解码原理是什么？为什么解码受制于带宽？ | [从自回归到投机解码](inference/从自回归到投机解码.md) | [llm-inference-perf](../../experiments/llm-inference-perf/README.md) |
| 投机解码在什么任务与参数下才能获得真收益？ | [投机解码收益与实测条件](inference/投机解码收益与实测条件.md) | [llm-inference-perf](../../experiments/llm-inference-perf/README.md) |
| 百 GB 级单卡如何部署 299B MoE 与高并发服务？ | [DGX-Spark 大模型推理与并发](inference/DGX-Spark大模型推理与并发.md) | [moe-2bit-fit](../../experiments/moe-2bit-fit/README.md) · [engine-slot-concurrency](../../experiments/engine-slot-concurrency/README.md) · [ninfer-serving](../../experiments/ninfer-serving/README.md) · [strata-arm64](../../experiments/strata-arm64/README.md) |
| 部署前如何定量推算模型显存与并发 KV Cache？ | [推理内存预算与成本估算](inference/推理内存预算与成本估算.md) | [inference-cost-models](../../experiments/inference-cost-models/README.md) |
| Intel NPU 异构图文嵌入推理如何搭建？ | [OpenVINO 与 NPU 多模态推理](inference/OpenVINO与NPU多模态推理.md) | [jina-clip-openvino](../../experiments/jina-clip-openvino/README.md) |

---

## 三、 文档理解与信息提取

规则解析、文字识别、版面结构与端到端质检。先看 [文档理解导览](document-understanding/README.md)。

| 工程问题 | 主线讲解 | 代码与复现实验 |
|---|---|---|
| 原生 PDF 如何 3 毫秒极速提取？为什么扫描件会完全提取不到文字？ | [PDF 规则解析与矢量提取](document-understanding/PDF规则解析与矢量提取.md) | [pdf-rule-parsers](../../experiments/pdf-rule-parsers/README.md) |
| 图像文字检测与文字识别如何执行？ | [PaddleOCR 文字检测与识别](document-understanding/PaddleOCR文字检测与识别.md) | [paddle-ocr-detection-recognition](../../experiments/paddle-ocr-detection-recognition/README.md) |
| 表格的行列与单元格拓扑如何提取？ | [SLANet+ 表格结构识别](document-understanding/SLANet表格结构识别.md) | [table-structure-recognition](../../experiments/table-structure-recognition/README.md) |
| 模块化拼接与端到端大模型如何还原表格？ | [表格还原的两代方案](document-understanding/表格还原：模块化拼接与端到端多模态.md) | [table-structure-recognition](../../experiments/table-structure-recognition/README.md) |
| 视觉语言模型如何端到端解析并打分质检？ | [MiniCPM 文档解析与质检](document-understanding/MiniCPM文档解析与质检.md) | [minicpm-ocr](../../experiments/minicpm-ocr/README.md) |

---

## 四、 决策模型与判别式打分

面向选项直接输出结构化概率，替代长文本生成的轻量分类、路由与状态判断。先看 [决策模型导览](decision-models/README.md)。

| 工程问题 | 主线讲解 | 代码与复现实验 |
|---|---|---|
| 分类与路由任务如何避免生成式 JSON 的长延迟与格式损坏？ | [决策模型原理与打分机制](decision-models/决策模型原理与打分机制.md) | [baseline-llm](../../experiments/decision-model-benchmark/baseline-llm/README.md) |
| 专用编码器、大模型单 Token 约束与 LoRA 适配器延迟与显存差多少？ | [决策模型架构评测与选型](decision-models/决策模型架构评测与选型.md) | [decision-model-benchmark](../../experiments/decision-model-benchmark/README.md) |
| 如何拟合教师软标签并对齐不确定性？ | [决策模型微调与概率校准](decision-models/决策模型微调与概率校准.md) | [rlcd-policy](../../experiments/decision-model-finetuning/rlcd-policy/README.md) |
| 图文音视全模态输入如何直接输出决策概率？如何防范数据泄漏？ | [多模态决策模型](decision-models/多模态决策模型.md) | [omni-multimodal](../../experiments/decision-model-benchmark/omni-multimodal/README.md) |

---

## 五、 视觉与多模态

界面元素定位、高精度抠像与受控生成适配。先看 [视觉与多模态导览](multimodal/README.md)。

| 工程问题 | 主线讲解 | 代码与复现实验 |
|---|---|---|
| 视频与单帧人像如何高效高保真抠像？ | [人物抠像质量与推理速度](multimodal/人物抠像质量与推理速度.md) | [portrait-matting](../../experiments/portrait-matting/README.md) |
| 如何用 20 张照片微调 SDXL，让它学习特定人物的外观？ | [人脸身份 LoRA 微调与量化评估](multimodal/人脸身份LoRA微调与量化评估.md) | [image-finetune](../../experiments/image-finetune/README.md) |
| 如何微调语音识别模型并评估词错误率？ | 暂无独立讲解 | [Whisper ASR 微调](../../experiments/asr-finetuning/README.md)：GB10 上完成小样本 GPU 微调，测试 WER 未见变化 |
| 如何尝试全双工实时多模态交互？ | 暂无独立讲解 | [MiniCPM-o 4.5 全双工实时多模态交互](../../experiments/full-duplex-multimodal/README.md)：录制视频与纯音频的实时双工链路均已验证；空 venv 依赖、CUDA 与解码检查通过，但 Demo 整栈未在空 venv 端到端运行；真人交互未验证 |

---

## 六、 数据结构与存储成本

基础数据结构与文件格式的理论模型与实际测量。先看 [数据结构成本导览](data-structure-costs/README.md)。

| 工程问题 | 主线讲解 | 代码与复现实验 |
|---|---|---|
| 如何用极小常数空间完成海量数据的成员判定、频次统计与 UV 去重？ | [概率数据结构与基数估计](data-structure-costs/概率数据结构与基数估计.md) | [Bloom/Cuckoo/HLL 模型](../../experiments/data-structure-costs/README.md) |
| B+ 树、LSM-Tree、跳表与 ART 在读写放大与内存并发上如何权衡？ | [读写放大与有序索引选型](data-structure-costs/读写放大与有序索引选型.md) | [B+Tree/LSM/SkipList/ART](../../experiments/data-structure-costs/README.md) |
| 现代硬件架构下，哈希表如何利用 SIMD 并行探查解决 Cache 缺失？ | [哈希表演进与内存冲突解决](data-structure-costs/哈希表演进与内存冲突解决.md) | [SwissTable/RobinHood 模型](../../experiments/data-structure-costs/README.md) |
| 倒排索引与大数据列存如何实现 1~3 比特的极限整型压缩与动态剪枝？ | [列式存储与倒排索引压缩编码](data-structure-costs/列式存储与倒排索引压缩编码.md) | [PFor-Delta/Simple8b/WAND](../../experiments/data-structure-costs/README.md) |
| JSON、JSONL、XLSX、SQLite 在读写、追加与索引膨胀上差多少？ | [数据格式体积与速度实测](data-structure-costs/数据格式体积与速度实测.md) | [数据格式基准实测](../../experiments/data-structure-costs/README.md) |
