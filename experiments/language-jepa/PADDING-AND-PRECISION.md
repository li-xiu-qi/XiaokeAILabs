# 填充长度、注意力实现与训练损失排查

[English](PADDING-AND-PRECISION.en.md)

这组诊断从两次训练日志的损失差异出发，检查数据、模型前向和注意力计算。它揭示了当前软件和设备组合下的数值敏感性，但还不能解释历史训练中常规组与 JEPA 组的全部差异。

## 固定输入的观察

在 DGX Spark 上，使用 Qwen2.5-1.5B-Instruct、BF16、PyTorch 2.14.0+cu130、Transformers 5.19.0、PEFT 0.20.0。取生成训练集中的两条固定样本，保持输入、标签、模型权重和推理模式相同，比较批处理与填充方式：

| 设备与注意力 | 两条合批，长度 128 | 两条合批，裁剪到长度 32 | 分别推理，长度 128 |
| --- | ---: | ---: | ---: |
| CUDA SDPA | 3.0918 | 0.000201 | 0.000172、0.000312 |
| CPU SDPA | 0.000209 | 0.000209 | 0.000157、0.000277 |
| CUDA eager | 0.3047 | 0.000086 | 0.000137、0.000105 |
| CPU eager | 0.000086 | 0.000086 | 0.000078、0.000096 |

数值为回答 token 的平均交叉熵。对另一对样本，CUDA SDPA 从长度 128 的 4.8876 变为长度 32 的 0.000191。模型在同一进程从 CUDA 移到 CPU 后，同一输入和标签的 SHA-256 一致；两组抽查权重的字节哈希与 dtype 一致。交叉熵由 logits 独立计算得到相同结果，差异已出现在模型前向，不能只用损失归约解释。

所有批次、长度、设备和注意力实现的测量值见[结果文件](results/padding-attention-observations.json)。

固定四个 microbatch，分别执行无梯度前向、逐批反向，以及启用或关闭 gradient checkpointing，CUDA SDPA 每批的 loss 均相同；没有执行优化器更新，抽查的可训练参数未变。这排除了这组控制实验中反向累积和 checkpointing 造成批次差异的解释。

## 掩码和首层精度检查

在 CPU 上，控制样本的答案标签位于 token 28–31；两种注意力实现首层实际使用的允许位置逐元素一致，position IDs 一致。逐个修改这些答案 token，观察其前一个位置的 logits，均未变化。这只覆盖所检查的位置，不能证明模型全部位置都没有泄漏。

取同一组首层 BF16 Q/K/V，转成 FP32 后手算注意力作为局部参照。CPU BF16 SDPA 的平均绝对误差为 0.000166，BF16 手算的误差为 0.01908。这个比较不证明任何实现对完整模型都更准确。

CUDA 上，同一组样本的首层 BF16 SDPA 相对对应 FP32 手算参照，长度 128 和 32 的平均绝对误差分别为 0.000169 和 0.000166；两者完整模型的 loss 却明显不同。首层结果因此不足以确定完整模型差异的原因，也不能将其直接归因于某个内核。

## 复查方法和边界

先按主文档生成训练集并准备修补后的作者训练脚本。使用与实验相同的软件版本和本地模型，分别运行：

```bash
python scripts/inspect_padding_loss.py --trainer /path/to/patched-finetune.py --model /path/to/model --data /path/to/train.jsonl --device cpu
python scripts/inspect_padding_loss.py --trainer /path/to/patched-finetune.py --model /path/to/model --data /path/to/train.jsonl --device cuda
```

公开脚本已在 CPU 和 CUDA 上分别复跑 16 组条件与首层检查；CUDA 结果与原排查记录逐项一致。首层检查只运行长度 128 的输入，不能直接复现上文长度 32 的 CUDA 首层数值。

脚本对模型只读前向，输出输入和标签哈希；追加 `--first-layer-check` 可以比较首层 BF16 注意力输出与 FP32 参照。样本索引依赖原始生成数据的顺序；模型 revision 当时未记录，依赖版本、硬件或权重不同时不保证重现相同数字。GPU 运行前需确认可用内存。模型权重、训练输出和私有机器路径不包含在本仓库中。

本诊断使用单次运行及少量合成样本，没有对训练后的能力重新作出比较。原模型与两组微调在简单测试题上的结果、数据饱和及历史训练日志的限制见[实验主页](README.md)。
