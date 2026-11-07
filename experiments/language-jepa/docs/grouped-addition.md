# 无交换等价题泄漏的加法数据：基座评估与后续对照

[English](grouped-addition.en.md) | [实验入口](../README.md)

原实验的测试题中有 21/40 与训练题仅交换了加数，且基座已答对全部 40 题。为了给后续训练提供更有区分度的检查，我们生成新的合成数据：每题两个三位数，个位和十位均发生进位；按无序数对去重后分为 320 条训练、80 条测试。随机种子为 20261107。分组防止同一对加数交换顺序后跨训练和测试集，但不能排除相同答案、相似算式或预训练中出现过的题目。

## 同一测试集上的三组结果

2026-11-07 在 DGX Spark 上，Qwen2.5-1.5B-Instruct 基座对 80 条测试题的严格完全匹配为 **73/80（91.25%）**。贪心解码、最多生成 12 个 token，去掉生成结果首尾空白后与答案逐字比较。同一天，使用该数据分别进行普通 LoRA 和带 JEPA 目标的 LoRA 训练，各 1 epoch、40 次更新，三组均在同一测试集上评估：

| 模型 | 严格完全匹配 |
| --- | ---: |
| 基座 | 73/80 |
| 普通 LoRA | 74/80 |
| JEPA LoRA | 75/80 |

普通 LoRA 与 JEPA 逐题比较，1 题从错变对、0 题从对变错。只有一个训练随机种子、80 道合成题，基座已很强，不能将 1 题差异视为 JEPA 可靠提升。旧训练中日志 loss 差异的原因仍未解释。

本次使用 PyTorch 2.14.0+cu130、Transformers 5.19.0、PEFT 0.20.0。模型权重文件 SHA-256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`；未记录模型 revision。生成的训练及测试 JSONL SHA-256 分别为 `ffc78b4df9588ef09709c122756146ef4740d78e944083d57ef7480ec732c518` 和 `ac6c4398f4b2a47df31b953e54bcc51e1c989545d6c45869a819b7c4adc18b9d`。三组逐条答案保存在 [结果文件](../results/grouped-comparison.json)，不含提示词。以下命令重新生成预期答案并核对三组命中数和配对差异：

```bash
python scripts/audit_grouped_comparison.py
```

## 准备和检查数据

输出目录必须不存在；生成后先验证标签、同集重复与跨集交换等价题：

```bash
python scripts/make_grouped_data.py --output /path/to/new-data
python scripts/audit_grouped_data.py --data /path/to/new-data
python scripts/audit_tokenizer.py --model /path/to/model --data /path/to/new-data/train.jsonl --max-length 128
```

Spark 上使用同一 tokenizer 对 320 条训练和 80 条测试均通过前缀与答案检查；CPU 反例检查也确认跨集交换加数和错误答案会被拒绝。数据是合成加法，不能代表自然语言任务。

## 对照配置与局限

两组使用相同训练集、batch 2、梯度累积 4、学习率 2e-5、LoRA rank 8、seed 42；JEPA 权重为 0.1。普通组与 JEPA 组的 Trainer 记录训练耗时分别为 35.43 秒和 70.24 秒；它们的目标和每步计算量不同，这只是各自训练阶段的一次计时，不是等计算预算比较，也不涵盖模型加载和最终保存。两组训练日志的 loss 分别为 0.4574 和 0.05867，目标不同，数值不能直接比较。

此前发现 GPU BF16 下不同填充长度会影响模型前向，原因仍未完全确认，见[排查记录](../PADDING-AND-PRECISION.md)。因此，这次成绩与耗时只描述本次设置，不能归因于 JEPA；若要检验方法，需要解决数值异常、增加独立测试与训练随机种子，并记录实际计算开销。
