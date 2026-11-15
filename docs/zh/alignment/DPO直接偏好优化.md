# DPO：直接偏好优化

同一个问题可以有两个不同回答。若偏好数据把其中一个标为更好，DPO 使用这对回答训练模型，使偏好回答相对另一回答的概率变化，高于相同变化在冻结参考模型上的比值。参考模型限制优化目标中对原模型的偏移；它不保证输出始终安全或优质。

给定提示词和偏好、非偏好回答，DPO 计算训练模型与参考模型分别对两个回答的条件对数概率。两者的对数概率差之差经过系数缩放后输入 logistic 损失。训练无需单独拟合奖励模型，也无需在更新时从模型采样回答。其推导与适用假设见 [DPO 原论文](https://arxiv.org/abs/2305.18290)。

本仓 [DPO 实验](../../../experiments/dpo-preference-alignment/README.md) 使用公开的模型评分偏好对、小型对话模型和 LoRA 微调，并在官方独立测试划分的固定候选上检查偏好方向。该指标不是生成质量或真实用户偏好的直接测量。训练和测试共用数据来源，近重复与预训练污染仍有可能；样本小、单次训练也不足以推广。

一次 256 对训练、128 对测试的小样本 GPU 运行中，64 次更新前后与数据偏好方向一致的测试回答对均为 71 对，且没有单对预测翻转。回答约半数被截断，评分基于回答 token 对数概率之和并受长度影响；结果不支持声称生成质量或偏好判断能力提升。运行环境、完整截断数量和指标见实验 README。

## 参考资料

- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.
- Cui et al., [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377), 2023.
