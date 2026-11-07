# 无交换等价题泄漏的加法数据：基座评估与后续对照

[English](grouped-addition.en.md) | [实验入口](../README.md)

原实验的测试题中有 21/40 与训练题仅交换了加数，且基座已答对全部 40 题。为了给后续训练提供更有区分度的检查，我们生成新的合成数据：每题两个三位数，个位和十位均发生进位；按无序数对去重后分为 320 条训练、80 条测试。随机种子为 20261107。分组防止同一对加数交换顺序后跨训练和测试集，但不能排除相同答案、相似算式或预训练中出现过的题目。

## 基座结果

2026-11-07 在 DGX Spark 上，Qwen2.5-1.5B-Instruct 基座对 80 条测试题的严格完全匹配为 **73/80（91.25%）**。贪心解码、最多生成 12 个 token，去掉生成结果首尾空白后与答案逐字比较。此结果表明这批题不再全对，但基座仍然很强；尚未运行新数据的普通 LoRA 与 JEPA 训练，不能从基座成绩推断两者的效果。旧训练中日志 loss 差异的原因仍未解释。

本次使用 PyTorch 2.14.0+cu130、Transformers 5.19.0、PEFT 0.20.0。模型权重文件 SHA-256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`；未记录模型 revision。生成的训练及测试 JSONL SHA-256 分别为 `ffc78b4df9588ef09709c122756146ef4740d78e944083d57ef7480ec732c518` 和 `ac6c4398f4b2a47df31b953e54bcc51e1c989545d6c45869a819b7c4adc18b9d`。逐条答案保存在 [结果文件](../results/grouped-base-eval.json)，不含提示词。以下命令重新生成预期答案并核对 73/80：

```bash
python scripts/audit_grouped_base.py
```

## 准备和检查数据

输出目录必须不存在；生成后先验证标签、同集重复与跨集交换等价题：

```bash
python scripts/make_grouped_data.py --output /path/to/new-data
python scripts/audit_grouped_data.py --data /path/to/new-data
python scripts/audit_tokenizer.py --model /path/to/model --data /path/to/new-data/train.jsonl --max-length 128
```

Spark 上使用同一 tokenizer 对 320 条训练和 80 条测试均通过前缀与答案检查；CPU 反例检查也确认跨集交换加数和错误答案会被拒绝。数据是合成加法，不能代表自然语言任务。

## 下一轮设计

先保持数据、模板、解码、LoRA 和优化器设置一致，分别测未微调基座、普通 LoRA、带 JEPA 目标的 LoRA；固定测试集，并在多个训练随机种子下保存逐题输出、耗时与内存。320 条、batch 2、梯度累积 4、1 epoch 对应每组 40 次参数更新；JEPA 每步计算量更高，等步数不等于等计算预算。训练前应继续排查历史两条训练路径的 loss 归约与精度差异；在这些差异解释清楚之前，即使得分有变化也不能归因于 JEPA。上述是计划，不是已完成的训练结果。
