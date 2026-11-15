# 单向量检索体系（Embedding）

> 阅读导航：[检索与排序](../README.md) · [多向量晚期交互（ColBERT）](../colbert/README.md) · [重排序模型（Reranker）](../reranking/README.md) · [向量数据库选型](../vector-search/README.md)

单向量稠密检索（Dense Embedding Retrieval）是现代语义搜索与检索增强生成（RAG）的第一道初筛环节。它通过深度神经网络将变长自然语言文本映射为固定维度的连续稠密向量，使系统能够在百万级乃至亿级文档库中，借助近似最近邻（ANN）索引在毫秒级内完成候选初筛。

本专题系统涵盖单向量流形机理、长文档切分与迟分治理、对比学习表征几何、监督微调、模型蒸馏以及图文多模态嵌入的完整工程体系。

---

## 一、体系篇目与学习路线

```
retrieval/embedding/
├── 理论基石：
│   ├── 稠密向量表示与最近邻检索.md   # 双塔流形、池化推导、InfoNCE 损失、ANN 边界
│   └── 对比学习与表征几何.md         # 表示塌缩、超球面对齐均匀性、SimCSE 扰动
├── 技术落地：
│   ├── 长文切分与迟分.md             # 全局自注意力后验池化、跨段指代消解实测
│   └── SimCSE对比学习.md             # Dropout 弱增广、NLI 难负例、两阶段目标
└── 优化与进阶：
    ├── Embedding监督微调.md          # 批内对比学习、难负例挖掘与防过拟合门禁
    ├── Embedding蒸馏.md              # 软标签分布对齐、配对 Bootstrap 置信度
    └── 多模态Embedding蒸馏.md        # 图文双塔跨模态对齐、教师优势前置检验
```

| 核心篇目 | 架构定位 | 核心问题与方法论 | 对应代码与凭证 |
|---|---|---|---|
| **[稠密向量表示与最近邻检索](稠密向量表示与最近邻检索.md)** | 高层理论 | 词袋同义词遗漏；双塔解耦前向、掩码均值池化代数推导；L2 归一化点积简化与 ANN 索引协同 | [embedding-finetuning](../../../../experiments/embedding-finetuning/README.md) |
| **[对比学习与表征几何](对比学习与表征几何.md)** | 高层理论 | 原生语言模型各向异性圆锥塌缩；Wang & Isola 对齐性与均匀性双力场；InfoNCE 梯度非对称加权推导 | [contrastive-learning-simcse](../../../../experiments/contrastive-learning-simcse/README.md) |
| **[长文切分与迟分](长文切分与迟分.md)** | 技术专篇 | 传统切块割裂长程指代；“先长文全局编码，再按块局部池化”；实测指代消解提升与均质化代价 | [embedding-finetuning](../../../../experiments/embedding-finetuning/README.md) |
| **[SimCSE对比学习](SimCSE对比学习.md)** | 技术专篇 | 无标注时文本增广两难；双 Dropout 前向构造自一致正例；NLI 矛盾难负例；五大训练失效细节规避 | [contrastive-learning-simcse](../../../../experiments/contrastive-learning-simcse/README.md) |
| **[Embedding监督微调](Embedding监督微调.md)** | 实验专篇 | 垂直领域专有名词漏召回；批内负例与 InfoNCE 损失；BM25 难负例挖掘与交叉编码器假负例过滤 | [embedding-finetuning](../../../../experiments/embedding-finetuning/README.md) |
| **[Embedding蒸馏](Embedding蒸馏.md)** | 实验专篇 | 8B 教师向 22M MiniLM 蒸馏；候选相似度分布 Softmax KL 散度拟合；配对 Bootstrap 显著性检验 | [embedding-distillation](../../../../experiments/embedding-distillation/README.md) |
| **[多模态Embedding蒸馏](多模态Embedding蒸馏.md)** | 实验专篇 | Jina v5 向 BGE-VL 蒸馏；图文双向跨模态相似度拟合；教师优势前置门禁与指标饱和效应反思 | [multimodal-embedding-distillation](../../../../experiments/multimodal-embedding-distillation/README.md) |

---

## 二、主流单向量模型选型

在工业生产实践中，不同基础架构在上下文窗口、向量维度和吞吐上差异明显：

| 模型家族 | 典型代表 | 基础架构 | 上下文窗口 | 向量维度 | 特征与适用场景 |
|---|---|---|---|---|---|
| **BGE 系列** | BGE-large-zh-v1.5 / BGE-M3 | BERT / 多功能混合 | 512 / 8192 | 1024 | 中文语义检索常用基线，M3 支持密集、稀疏与多向量三合一 |
| **E5 系列** | e5-mistral-7b / multilingual-e5 | Decoder-only / XLM-R | 32k / 512 | 4096 / 1024 | 需指定 `query:` 与 `passage:` 非对称指令前缀，跨语言泛化能力强 |
| **Jina 系列** | jina-embeddings-v3 | ModernBERT / RoPE | 8192 | 1024 (支持 Matryoshka 截断) | 原生支持任务特定 LoRA 适配器与 Late Chunking 迟分 |

---

## 三、边界与协同建议

1. **单向量是粗召回，不是终点**：受固定向量长度限制，单向量模型难以保留所有局部细节。在高精度要求场景中，通常将单向量初筛召回的 Top-100 结果送入 [重排序模型（Reranker）](../reranking/README.md) 进行二次精排；
2. **多实体复合查询建议评估多向量**：若业务查询经常出现同时包含多个实体、药物、法条相互制约的长句子，建议评估采用保留 Token 级特征的 [晚期交互多向量检索（ColBERT）](../colbert/晚期交互与多向量检索.md) 方案。
