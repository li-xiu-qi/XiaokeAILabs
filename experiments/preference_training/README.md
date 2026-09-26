# 偏好训练（Preference Training）

用「 chosen 优于 rejected 」的成对偏好信号直接优化模型策略的实验组。这是对齐（alignment）训练的一条路线：不训奖励模型、不做 RL 采样循环，把偏好对喂进一个闭式损失函数直接更新策略。

## 路线谱系（本组的收录边界）

偏好训练与 RLHF 同属对齐大家族，但机制相反：

| 路线 | 机制 | 本组是否收录 |
|---|---|---|
| RLHF | 先训 reward model，再用 PPO 采样优化 | 否（RL 路线，另组） |
| DPO / IPO / ORPO / KTO / SimPO | 偏好对 + 闭式损失直接优化 | 是 |

与 `embedding_research/model_finetuning` 的边界：那边也用 pair 信号（对比学习、pointwise BCE），但目标是把表示学得好（向量、重排分数）；本组的目标是把策略对齐偏好（生成分布）。信号形态像，优化对象不同，不合并。

## 当前实验

| 实验 | 内容 |
|---|---|
| [test_dpo/](test_dpo/) | DPO 单脚本教学示例：`trl-lib/ultrafeedback_binarized` 偏好对、参考模型 log-ratio 预计算、DPO 损失训练循环、loss 与 preference probability 曲线 |

当前只有一个实验，本组是偏好训练系列的预留入口：后续 RLHF 家族其他变体（ORPO、SimPO 等）与多模型对照实验落这里，rl 路线的 PPO 实验不进来。

`test_dpo` 的教学定位：unsloth/trl 之外的从零件写起的演示（数据解析 `extract_triple`、分词截断、参考模型预计算、梯度累积、bfloat16），换模型与数据集的方法写在该目录 README 的注意事项里。
