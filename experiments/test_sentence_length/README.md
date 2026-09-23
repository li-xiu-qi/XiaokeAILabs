# 句子长度对相似度计算的影响

两个 notebook 测同一个变量：句子长度变化时，embedding 之间的余弦相似度怎么动。长句和多语言场景下的相似度基准与短句不同，直接用固定阈值判断「像不像」会误判，这组实验给出量化依据。

| Notebook | 内容 |
|---|---|
| `sentence_length_similarity_experiments.ipynb` | 完整实验：不同长度句对的相似度分布 |
| `multilingual_sentence_length_similarity_experiments.ipynb` | 多语言扩展（⚠️ 未完成，结论以主实验为准） |
