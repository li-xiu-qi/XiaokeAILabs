# 聚类与降维：高维向量的主题发现、查询路由与可视化

这个系列处理 embedding 拿到之后的两个问题：怎么理解这一堆向量（它们分成哪些主题、查询该去哪个知识库），以及怎么把高维向量画出来。包含两个子实验，一个走 K-Means 的工程应用，一个走 UMAP 降维可视化。

## 子实验

### test_k_means：K-Means 聚类的三个工程用法

不是 K-Means 算法本身的演示，是三个具体应用场景，共用 BGE-M3 生成文档向量加 scikit-learn 聚类：

| 脚本 | 解决的问题 |
|---|---|
| `news_topic_clustering.py` | 新闻主题发现：对文档集聚成 5 个簇，用轮廓系数评估簇质量，输出每簇关键词 |
| `knowledge_base_router.py` | 查询路由：`ClusterAwareRouter` 为每个专业知识库预先聚类、保存簇中心，查询按最近簇中心路由到对应知识库 |
| `diverse_document_retrieval.py` | 多样化检索：先取相似度最高的候选，再在候选上聚类，从每簇选最相似文档，避免检索结果全挤在同一语义簇 |

`test_k_means.ipynb` 是同一批逻辑的 notebook 版草稿，`news_clusters_visualization.png` 是聚类可视化输出。

三个用法共享同一条思路：聚类不是终点，是把「对全部向量做昂贵操作」变成「只对最近的簇做」，或者把「同质结果」变成「覆盖多个语义簇」。

### test_umap：UMAP 降维与可视化

`umap.ipynb` 用 UMAP 把 embedding 降到 3 维，用 Plotly 做交互式 3D 可视化，并对比降维前后的余弦相似度与欧氏距离，观察降维损失了多少结构。适合在聚类或检索之前先肉眼看数据分布。

## 阅读顺序

先跑 `test_umap/umap.ipynb` 看自己数据的整体分布，再按场景选 test_k_means 下的脚本：要理解语料用主题聚类，要做知识库分发用路由，要改进检索覆盖面用多样化检索。

## 依赖

```
FlagEmbedding
scikit-learn
umap-learn
plotly
pandas
matplotlib
```

K-Means 聚类对向量数量有要求，`ClusterAwareRouter` 里每个知识库的向量数不能少于簇数，否则初始化直接报错。
