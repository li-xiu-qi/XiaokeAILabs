# 判别式决策模型自建微调与训练（decision-model-finetuning）
[English](README.en.md)

本目录包含从零训练与微调专用判别式决策模型的工程实践，覆盖全参数分类头微调、RLCD（基于规则的决策强化）与软标签损失对照，以及多模态视觉决策探索。对应讲解见[决策模型系列导览](../../docs/zh/decision-models/README.md)。

## 训练项目列表

| 子目录 | 训练目标与底座 | 数据集与方法 | 关键结论与状态 |
|---|---|---|---|
| [agent-jev-text/](agent-jev-text/) | 纯文本判别决策模型（Qwen3-0.6B 底座） | `LocalLLaMA/typed-decisions`，全参数微调分类头 | 完成训练、验证集校准、测试集评测与轻量推理服务封装 |
| [rlcd-policy/](rlcd-policy/) | RLCD 策略梯度 vs 软交叉熵损失对照 | 在 DGX Spark 单卡上微调 Laya，对比带规则奖励与纯蒸馏拟合 | 软标签监督与策略梯度的概率校准差异实测 |
| [vision-decision/](vision-decision/) | 视觉判别决策模型（多模态底座 + Scalar Score 头） | 图文多候选决策打分 | **重要边界提示**：历史数据包含参考答案 `solution`，修正后的数据准备尚未完成重新训练和评测 |

## 复现前提

- 各项目分别准备独立的 Python 环境和 CUDA 版 PyTorch；具体依赖、下载与运行步骤见各子目录 README。依赖清单不等同于完整锁定的历史环境。
- `agent-jev-text` 与 `rlcd-policy` 使用合成数据集 `typed-decisions`；`vision-decision` 使用 ScienceQA。下载数据和权重需要网络、磁盘空间，以及遵守各上游许可。
- 历史 `rlcd-policy` 运行使用 DGX Spark（GB10 统一内存）；其他设备的训练内存需求与可复现性需要自行验证。`vision-decision` 依赖 Linux 的内存检查，历史数据存在答案泄漏，历史结果不能用来证明视觉答题效果。
