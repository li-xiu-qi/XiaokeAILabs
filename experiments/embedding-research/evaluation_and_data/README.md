# 评测基准与训练数据

了解 MTEB、BEIR 等基准，以及微调前的数据准备。这里保留的基准说明整理自 [FlagEmbedding Tutorials](https://github.com/FlagOpen/FlagEmbedding/tree/master/Tutorials)，不是本项目的实测结论；原始可执行 notebook 请直接阅读上游教程。

- [基准说明](test_embedding/test_evaluation/zh/markdown/4.2.1_MTEB_Intro.md)：MTEB 入门及其余基准文档。
- [对比损失演示](test_embedding/test_embedding_fine_tuing_data_prepare/simcse.py)：观察同一句的 dropout 表示与一批句子的对比损失；它不准备训练数据，也不执行训练或证明检索效果。
- [本项目微调实验](../model_finetuning/)：训练流程、数据和实验结果。

上游项目采用 MIT 许可，版权与许可文本见 [FLAGEMBEDDING-LICENSE.txt](FLAGEMBEDDING-LICENSE.txt)。保留的上游资料中部分相对路径指向上游仓库，需要结合原项目使用。
