# 模型微调训练

两个训练项目，都围绕 BAAI bge 系模型的微调脚手架，训练目标不同：一个训嵌入向量，一个训重排分数。

| 项目 | 训练对象 | 目标与损失 |
|---|---|---|
| `test_train_embedding/` | 文本嵌入模型 | 对比学习，提升语义相似度捕捉，服务向量检索与文本匹配 |
| `test_train_reranker/` | Cross-Encoder 重排模型（`BAAI/bge-reranker-large`） | pointwise 训练（BCEWithLogitsLoss），精排检索结果 |

两者都是可运行的完整训练代码，非实验记录。跑之前按各自 README 与 `requirements.txt` 备环境。

评测基准与训练数据准备看 [../evaluation_and_data/](../evaluation_and_data/)。
