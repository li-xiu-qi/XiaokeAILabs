# SimHash、MinHash 与 Winnowing：文本相似度三兄弟

三个经典的文本相似度/去重算法，全部从零实现，互相对照着讲。共同问题：两份文档像不像，但都不能直接算（全文比对慢、存储大），于是各用一种降维加指纹的思路。

## 三份 notebook 的关系

| Notebook | 算法 | 指纹 | 解决的问题 |
|---|---|---|---|
| `simhash_from_scratch.ipynb` | SimHash | 64 位向量 | 降维：一段文本压成一个指纹，海明距离小于 3 判定相似 |
| `test_simhash.ipynb` | SimHash 词级版 | 词级特征 | 用词而非字做特征，查重场景实测 |
| `simhash_analysis.ipynb` | SimHash 进阶 | 同一指纹 | 系统分析不同程度文本改动下指纹怎么变，给出实用阈值 |
| `minihash.ipynb` | MinHash | 签名矩阵 | Jaccard 相似度估计，带 LSH 分桶加速 |
| `winnowing_from_scratch.ipynb` | Winnowing | k-gram 指纹 | 局部文档指纹，允许内容搬移，适合查重取证 |
| `winnowing_interactive_test.ipynb` | Winnowing 词级 | 词级 k-gram | 交互式验证 k 值与阈值的影响 |

## 三个算法的取舍

**SimHash** 适合「整篇像不像」：全维度随机投影，改动小则指纹变化小，海明距离即相似度。写得快，但内容搬移会误判（段落顺序换了指纹差很远）。

**MinHash** 估计的是 Jaccard 集合相似度，对内容搬移免疫（集合无序），代价是要维护签名矩阵，存储比单指纹大，通常配 LSH 分桶做候选加速。

**Winnowing** 介于两者之间：按 k-gram 窗口取局部指纹，既能容忍搬移和插入，又比全集哈希省存储，是论文查重系统的常用路线。

按上表顺序读，先 SimHash（最好懂，从零实现完就有可用的查重器），再 MinHash（理解 LSH 何用），最后 Winnowing（理解指纹怎么对抗内容搬移）。
