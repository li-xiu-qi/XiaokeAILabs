# 检索与排序体系

> 阅读导航：[English](../../en/retrieval/README.md) · [项目首页](../../../README.md) · [实验地图](../README.md)

现代信息检索系统普遍采用“初筛召回（Retrieve）+ 细粒度重排序（Rerank）”的分层级联架构。如何在百万至数亿全库候选上实现低延迟高召回初筛，再在数十个高价值候选上实现精细的跨词语义排序，是构建高精度搜索与 RAG 系统的核心挑战。

本方向完整覆盖从统计词频倒排（BM25）、学习型稀疏表示（SPLADE）、单向量密集嵌入（Embedding）、多向量晚期交互（ColBERT）、全自注意力重排序器（Cross-Encoder）到近邻向量索引与数据库系统的全技术谱系。

---

## 一、检索体系路线图

```
docs/zh/retrieval/
├── 全景与方法学：
│   ├── 检索模型与方法全景.md       # 词项稀疏、学习稀疏、单向量、多向量与交叉编码全景
│   ├── 检索系统全景与评估.md       # Recall@k、nDCG@k、MRR 等指标在各环节的物理含义
│   └── 知识蒸馏与软标签监督.md     # 词表分布、成对标量差值 Margin-MSE、候选分布 Softmax KL
├── 稀疏检索（Sparse）：
│   └── sparse/学习型稀疏检索与SPLADE.md   # MLM 词表投影、自适应词项扩展、FLOPS 物理正则化
├── 单向量检索（Embedding）：
│   ├── embedding/稠密向量表示与最近邻检索.md # 双塔流形、掩码均值池化、InfoNCE 损失、ANN 边界
│   ├── embedding/对比学习与表征几何.md     # 各向异性塌缩、超球面两项测度、SimCSE 扰动
│   ├── embedding/长文切分与迟分.md         # 全局自注意力后验池化、跨段指代消解实测
│   ├── embedding/SimCSE对比学习.md         # Dropout 弱增广、NLI 矛盾难负例与两阶段目标
│   ├── embedding/Embedding监督微调.md      # 批内负例与 InfoNCE、BM25 难负例挖掘
│   ├── embedding/Embedding蒸馏.md          # 8B 教师向 22M MiniLM 蒸馏、配对 Bootstrap 检验
│   └── embedding/多模态Embedding蒸馏.md    # Jina v5 图文双向跨模态蒸馏、教师优势门禁
├── 多向量晚期交互（ColBERT）：
│   ├── colbert/晚期交互与多向量检索.md     # 四大匹配范式、MaxSim 算子代数推导、查询扩展
│   ├── colbert/多向量索引与剪枝加速.md     # 残差聚类压缩、PLAID 剪枝算子、14.8x 压缩实测
│   ├── colbert/文本多向量检索模型微调.md   # 批内负例与 Margin-MSE、SciFact 实测与 Bootstrap
│   ├── colbert/文本多向量检索模型蒸馏.md   # 8B 密集教师向 ColBERT 软标签蒸馏、纯监督组对照
│   ├── colbert/视觉多向量检索.md           # ViT Patch 投影、页面级跨模态检索基线
│   ├── colbert/视觉多向量检索模型微调.md   # LoRA 适配器挂载、DocVQA 160 页防作弊实测
│   └── colbert/视觉多向量检索模型蒸馏.md   # Qwen3-VL 跨模态软打分拟合、64 页独立盲测
├── 重排序模型（Reranking）：
│   ├── reranking/重排序监督微调.md         # 全自注意力交叉编码、成对逻辑斯蒂损失实测
│   ├── reranking/重排序蒸馏.md             # Qwen3-8B 向 4.39M TinyBERT 蒸馏、五臂软分数对照
│   └── reranking/多模态重排序.md           # 视觉语言大模型全注意力精排、32.5x 算力代价实测
└── 向量索引与数据库（Vector Search）：
    ├── vector-search/向量索引与近邻搜索.md # Flat、IVF、HNSW、PQ 内存物理推导与选型
    ├── vector-search/向量数据库架构与选型.md # 进程内嵌入式、单机专用服务与分布式存算分离
    ├── vector-search/向量数据库怎么测才可比.md # 体积四口径、同召回率对齐基准、mmap 与容器隔离
```

---

## 二、主线专题导览

| 专题模块 | 核心篇目 | 架构定位与核心解决的问题 |
|---|---|---|
| **全景与基石** | [检索模型与方法全景](检索模型与方法全景.md)<br>[检索系统全景与评估](检索系统全景与评估.md)<br>[知识蒸馏与软标签监督](知识蒸馏与软标签监督.md) | 横向对比四大检索范式；明确度量指标体系在各级流水线中的物理含义；系统归纳跨架构知识蒸馏三大信号形态。 |
| **稀疏检索** | [学习型稀疏检索与SPLADE](sparse/学习型稀疏检索与SPLADE.md) | 复用掩码语言模型词表头实现免同义词表的自适应词项扩展；利用 FLOPS 正则化抑制长倒排链。 |
| **单向量密集检索** | [稠密向量表示与最近邻检索](embedding/稠密向量表示与最近邻检索.md)<br>[对比学习与表征几何](embedding/对比学习与表征几何.md)<br>[长文切分与迟分](embedding/长文切分与迟分.md) | 推导双塔解耦前向与掩码均值池化；推导超球面对齐性与均匀性双力场；先编后切解决长文切块指代割裂。 |
| **多向量晚期交互** | [晚期交互与多向量检索](colbert/晚期交互与多向量检索.md)<br>[多向量索引与剪枝加速](colbert/多向量索引与剪枝加速.md)<br>[视觉多向量检索](colbert/视觉多向量检索.md) | 深入推导 MaxSim 局部点积峰值求和算子；解析 PLAID 质心剪枝与残差量化；页面级多模态跨向量对齐。 |
| **重排序精排** | [重排序监督微调](reranking/重排序监督微调.md)<br>[重排序模型蒸馏](reranking/重排序蒸馏.md)<br>[多模态图文重排序](reranking/多模态重排序.md) | 全自注意力双向跨词交互与成对逻辑斯蒂损失；8B 教师向 4.39M TinyBERT 软分数蒸馏；VLM 视觉精排。 |
| **向量搜索与数据库** | [向量索引与近邻搜索](vector-search/向量索引与近邻搜索.md)<br>[向量数据库架构与选型](vector-search/向量数据库架构与选型.md)<br>[向量数据库评测方法](vector-search/向量数据库怎么测才可比.md) | 推导 Flat、IVF、HNSW、PQ 内存公式；梳理嵌入式、单机与分布式集群选型决策；确立同召回率 Pareto 对齐原则。 |
