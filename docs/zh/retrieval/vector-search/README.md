# 向量索引与检索数据库

> 阅读导航：[检索与排序](../README.md) · [单向量检索（Embedding）](../embedding/README.md) · [多向量晚期交互（ColBERT）](../colbert/README.md) · [代码实践：vector-db-benchmark](../../../../experiments/vector-db-benchmark/)

向量检索是现代语义搜索、检索增强生成（RAG）与多模态推荐系统的底层加速构件。当候选向量规模从数万扩展至数百万乃至数亿时，系统必须在近邻召回率、查询响应延迟、构建吞吐与内存物理开销之间做出严密的工程权衡。

本专题将算法层面的近似近邻（ANN）数据结构与系统层面的向量数据库服务架构贯通起来，提供清晰的数学推导、体系化选型与公平评测基准。

---

## 一、体系篇目与学习路线

```
retrieval/vector-search/
├── 理论与系统架构：
│   ├── 向量索引与近邻搜索.md     # 四大近邻路径、内存物理公式推导、ANN 边界
│   ├── 向量数据库架构与选型.md   # 嵌入式、单机服务与分布式存算分离选型决策
│   └── 向量数据库评测方法.md     # 体积四口径、同召回率对齐、容器/进程度量边界
└── 基准实验与实测对照：
```

| 核心篇目 | 架构定位 | 核心问题与方法论 | 对应代码与凭证 |
|---|---|---|---|
| **[向量索引与近邻搜索](向量索引与近邻搜索.md)** | 高层理论 | Flat、IVF、HNSW 与量化（SQ/PQ）如何交换空间与时间？ | 拆解四大检索路径的内存物理公式推导；候选集放大与二次精排补偿；ANN 精度真值判据 | [vector-db-benchmark](../../../../experiments/vector-db-benchmark/README.md) |
| **[向量数据库架构与选型](向量数据库架构与选型.md)** | 架构指南 | 进程内嵌入式、单机专用服务与分布式集群各适合什么场景？ | 从查询执行路径出发，梳理 sqlite-vec、LanceDB、Qdrant、Milvus 的适用边界与标量过滤代价 | [vector-db-benchmark](../../../../experiments/vector-db-benchmark/README.md) |
| **[向量数据库评测方法](向量数据库怎么测才可比.md)** | 方法规范 | 如何避免脱离召回率比延迟、混淆容器与进程内存的评测误区？ | 确立体积四口径、同召回率 Pareto 对齐、mmap 与容器隔离度量原则，以及混合检索 RRF 评测框架 | [vector-db-benchmark](../../../../experiments/vector-db-benchmark/README.md) |

---

## 二、代码与实验复现

- **[多款向量检索数据库基准评测（vector-db-benchmark）](../../../../experiments/vector-db-benchmark/README.md)**：提供多产品横向对比评测工具，涵盖写入吞吐、Docker 容器与宿主进程内存监控，以及带标量过滤条件的近邻召回统计。
