# Embedding 评测与数据准备

基准资料包括 [MTEB 总览](test_evaluation/zh/markdown/4.2.1_MTEB_Intro.md)和 [C-MTEB](test_evaluation/zh/markdown/4.2.3_C-MTEB.md)。这些资料源自 [FlagEmbedding Tutorials](https://github.com/FlagOpen/FlagEmbedding/tree/master/Tutorials)，可执行 notebook 和最新版本请查阅上游，许可文本见 [FLAGEMBEDDING-LICENSE.txt](../FLAGEMBEDDING-LICENSE.txt)。

本仓的训练实践见 [Embedding 微调](../../model_finetuning/test_train_embedding/README.md)。[simcse.py](test_embedding_fine_tuing_data_prepare/simcse.py) 只演示 dropout 表示与对比损失的计算，不准备数据、不更新参数，也不能评估微调效果。
