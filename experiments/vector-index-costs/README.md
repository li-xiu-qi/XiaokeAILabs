# 向量索引：在召回、速度与空间之间做选择

[English](README.en.md)

![四种向量检索路径](assets/search-paths.svg)

向量搜索需要从大量候选中找到与查询最接近的内容。逐个比较可以得到精确结果，但数据变大后计算和内存成本会上升；近似最近邻索引通过缩小搜索范围或压缩向量减少开销，也可能遗漏一部分近邻。理解这笔交换，才能为检索系统选参数，而不是直接把某种索引当成默认答案。

本实验介绍 Flat、IVF、HNSW 和量化的工作方式，推导索引存储的组成，并分别以真实文本向量的历史记录和可重建的合成向量测量召回与开销。它适合需要设计语义搜索、检索增强生成或推荐系统的读者，用来建立基线和制定自己的测试方案。

**从这里开始：**[索引原理与选型](docs/索引原理与选型.md) 解释检索路径、参数与内存公式；[文本向量对照](docs/文本向量索引对照.md) 与[合成向量对照](docs/合成向量索引对照.md) 给出统一距离口径的多索引实测、复现命令和限制。

## 先看一次实测

在一份已有记录标注为 20 Newsgroups 的文本向量上，用 11,293 条、384 维归一化向量和 600 条查询，单线程 Faiss CPU 计算 Top-10。Flat 的精确近邻作为对照。以下耗时是三次批量查询总耗时的中位数除以查询数，并非单次请求的 P50 或 P99。

| 方法及参数 | Recall@10 | 平均耗时（ms/查询） |
|---|---:|---:|
| Flat 精确搜索 | 1.0000 | 0.0715 |
| IVF，nlist=1024，nprobe=8 | 0.8180 | 0.0245 |
| IVF，nlist=1024，nprobe=32 | 0.9555 | 0.0626 |
| HNSW，M=16，efSearch=16 | 0.9342 | 0.0384 |
| HNSW，M=16，efSearch=64 | 0.9923 | 0.1099 |
| SQ8 | 0.9948 | 0.8191 |
| PQ，m=32，8 bit | 0.6200 | 0.1334 |

这组数据里，扩大 IVF 或 HNSW 的搜索范围提高了召回。HNSW 在 efSearch=64 时比 Flat 慢，说明在这个小数据集上，近似索引并不一定更快。计时波动明显，不能据此外推其他规模、硬件或负载，也不能把不同召回率的耗时当作公平排名。模型 revision 和编码脚本没有保留，原始向量文件不随仓库发布；完整配置、训练样本不足警告与证据见[文本向量对照](docs/文本向量索引对照.md)。

上表的 PQ 数据来自统一内积口径的重跑，并非旧结果。旧记录中 PQ 与 IVF-PQ 的度量口径无法从历史证据确认；旧脚本中的压缩索引使用默认 L2 而精确基线是内积，在量化重建后两者不能视为等价。当前脚本显式使用内积并修正了量化索引的构建计时；重跑结果与旧记录并非完全相同的实验。

## 它们怎样工作

- **Flat**：逐一计算查询与候选的距离，作为精确近邻基准。
- **IVF**：先将向量划分到多个分区，查询时只搜索部分分区。`nprobe` 控制搜索多少分区；增加它通常会扩大搜索范围，但实际召回要测量。
- **HNSW**：用多层近邻图组织向量，搜索时沿图探索候选。`efSearch` 控制搜索时保留候选的规模，成本与效果依赖数据和建图参数。
- **量化**：用更紧凑的编码近似表示向量，减少存储与部分计算成本，同时可能引入距离误差。

对于单条查询，以精确搜索得到的前 $k$ 个结果为集合 $G_k$，近似索引返回的结果为 $A_k$，则：

$$\operatorname{Recall@}k = \frac{|A_k \cap G_k|}{k}$$

整体数值是各查询召回率的平均值。这个指标衡量的是近似索引对精确向量近邻的还原程度，不等于用户认为搜索结果相关的比例；检索质量还取决于向量模型与数据。IVF 和 HNSW 的召回率不能仅由索引参数用通用公式算出，应在目标数据上测量。

## 如何动手

1. 准备具有代表性的候选和查询向量，明确维度、归一化方式、距离度量以及 Top-k。
2. 运行 Flat 得到精确近邻，再分别构建索引，改变 `nprobe` 或 `efSearch`，记录召回、耗时和构建成本。
3. 在相近召回率下比较延迟，并重复测量；规模、并发、内存、索引构建和更新成本也需要按实际需求评估。

[合成数据脚本](scripts/benchmark_synthetic.py) 可自行生成向量并从头运行完整对照；依赖及命令见[实验文档](docs/合成向量索引对照.md)。

[真实向量测量脚本](scripts/verify_recall_real_corpus.py) 读取预先准备的 `base_vectors.bin`、`query_vectors.bin` 和 `meta.json`；[转换脚本](scripts/prepare_corpus_from_20news.py) 只能将已有的 NPZ 向量归一化并转为这些文件，不会下载文本或计算 Embedding。两者都不随仓库提供模型与数据。历史记录注明 `all-MiniLM-L6-v2`、Faiss 1.13.2；如自行提供包含 `train_emb`、`test_emb`、`train_labels`、`test_labels` 的 NPZ，可这样运行：

```bash
python scripts/prepare_corpus_from_20news.py --src /path/to/20news_emb.npz --out-dir /tmp/vector-corpus --corpus '20 Newsgroups' --model 'all-MiniLM-L6-v2 (revision 自行填写)'
python scripts/verify_recall_real_corpus.py --base /tmp/vector-corpus/base_vectors.bin --query /tmp/vector-corpus/query_vectors.bin --query-limit 600 --out /tmp/recall.json
```

`--query-limit` 选取查询文件的前 600 条；如果来源、查询顺序或模型版本不同，这不是对历史结果的严格复现。请只使用可信 NPZ 文件，转换脚本为兼容历史数据允许读取 NumPy pickle 对象。路径可按操作系统调整。

[成本模型](scripts/vector_index_model.py) 可用于理解不同索引的空间与搜索工作量估算；它是简化模型，参数和实现细节会改变实际成本，不输出召回率预测。本次表格的原始记录为 [文本向量](evidence/real_corpus_ip_20261107.json) 与[合成向量](evidence/synthetic_n20000_d128_20261107.json)；`evidence/` 中其他 JSON 是历史数据，不能视为上述两次测量的原始结果。

进一步阅读：[Faiss 索引说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) · [Faiss 常见问题](https://github.com/facebookresearch/faiss/wiki/FAQ) · [向量数据库基准](../vector-db-benchmark/)。
