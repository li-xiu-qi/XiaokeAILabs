# FAISS 预计算表内存限制与 Milvus 的分段解法

## 这个文档是干嘛的

记录 FAISS IVFPQ 预计算表内存限制的准确出处（即「1GB」说法的来源），以及 Milvus 通过数据分段（Segmentation）机制如何绕开单索引规模上限。供 `test_vector_db_bench` 选型时参考。

## FAISS 的限制：IVFPQ 预计算表内存上限

流传的「FAISS 有 1GB 限制」指的是 **IVFPQ 索引预计算表（precomputed table）的内存上限**。

证据在 FAISS 源码 `faiss/IndexIVFPQ.cpp`：参数 `precomputed_table_max_bytes` 控制预计算表的最大字节数，超过阈值自动禁用预计算表以节省内存。该参数默认值为 2GB（`(size_t)1 << 31`），sqlite-vss 的文档中举例将其设为 1GB（`index.precomputed_table_max_bytes = 1 << 30`）。这是内存与性能的权衡。

## Milvus 的解法：数据分段

无论 FAISS 的实际限制是多少，Milvus 都不依赖单索引装下全量数据。它的核心机制是**分段（Segmentation）**，源码证据如下（clone 位置：`_reference/projects/infra-and-backend/milvus`）：

| 证据 | 位置 | 内容 |
|---|---|---|
| 段大小上限 | `configs/milvus.yaml:204` | `maxSize: 256M`，单个 segment 最大 256MB |
| 建索引最小行数 | `configs/milvus.yaml:357` | `minSegmentSizeToEnableIndex: 1024`，少于 1024 行的段不建索引、走暴力搜索 |
| 段内分块 | `configs/milvus.yaml:599` | `chunkRows: 128`，Segcore 把段按 128 行切块 |
| 段类型 | `internal/core/src/segcore/SegmentGrowing.h`、`ChunkedSegmentSealedImpl.h` | Growing 段接收写入，Sealed 段只读并承载索引 |

工作流程是：数据写入先进 Growing 段，达到 `maxSize` 后封存为 Sealed 段，再对每个 Sealed 段独立调用 Knowhere 构建 FAISS 索引。查询时跨多个 Sealed 段并行检索后聚合。单段体积被 `maxSize` 硬性封顶，因此 FAISS 索引永远不会收到超出段容量上限的数据。

Milvus 官方 sizing 工具给出的默认段大小是 1024MB（可选 512MB/2048MB），与源码 `maxSize: 256M` 的差异来自版本演进，两者机制一致。

## 对基准测试的意义

本实验的 11,293 条 × 384 维数据远低于任何段上限，Milvus 只产生了单个 Sealed 段，因此测不到分段带来的规模扩展效应。如果未来要测 Milvus 在亿级向量下的表现，段大小和索引构建的跨段调度会成为主要变量。

## 来源

- Milvus 源码 `configs/milvus.yaml`、`internal/core/src/segcore/`（clone 于 2026-09-05）
- Milvus 官方博客《Introducing the Milvus sizing tool》（2026-09-05 检索）
- FAISS 官方 Wiki 与 GitHub issue #2809（2026-09-05 检索）
- sqlite-vss 文档中关于 FAISS 预计算表 1GB 限制的说明（2026-09-05 检索）
