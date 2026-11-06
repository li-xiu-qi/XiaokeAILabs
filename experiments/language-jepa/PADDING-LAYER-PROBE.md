# 填充长度差异从哪里出现

[English](#where-the-padding-difference-first-appears)

上一项实验发现，同一批样本在 GPU 上补齐到不同长度时损失不同。这里逐层检查两个样本中相同的 32 个有效 token：先比较嵌入、每层输出和 logits，再拆开首层的归一化、注意力和前馈网络。两次调用只改变右侧填充长度，不更新权重。

## 观察

在 DGX Spark、BF16、PyTorch 2.14.0+cu130、Transformers 5.19.0 和 PEFT 0.20.0 上，样本 34 与 85 均有 32 个有效 token。下面的最大差与平均差计算对象都是两条样本的有效位置。模型 revision 未记录。

| GPU 注意力路径 | 组件 | 最大绝对差 | 平均绝对差 |
| --- | --- | ---: | ---: |
| SDPA | 首层输入归一化 | 0 | 0 |
| SDPA | 首层注意力模块输出 | 1.3984 | 0.02513 |
| SDPA | 首层前馈网络输出 | 1.7383 | 0.05292 |
| SDPA | 首层完整输出 | 2.3164 | 0.05890 |
| eager | 首层输入归一化 | 0 | 0 |
| eager | 首层注意力模块输出 | 1.1484 | 0.03839 |
| eager | 首层前馈网络输出 | 2.1250 | 0.08585 |
| eager | 首层完整输出 | 3.2500 | 0.09423 |

两次调用在有效位置的旋转位置编码 cos、sin 完全相同。SDPA 下，填充长度 128 与裁剪后长度 32 的损失分别为 4.8876 和 0.000191；在相同条件的 CPU 上，两种注意力路径各自的逐层隐藏状态与 logits 在有效位置完全一致。

注意力模块包括 Q/K/V 投影、注意力计算和输出投影。本观察将第一个可见差异缩小到该模块，**不能单独归因于某一种注意力内核**，也不能证明历史两次训练损失差异的完整原因。此前对首层 Q/K/V 的局部 FP32 对照及其限制见[填充长度与精度排查](PADDING-AND-PRECISION.md)。

进一步检查首层投影：K 和 V 的有效位置输入在两次调用中一致；同一实例的 K 基础线性层输出最大差 4.0、平均差 0.10075，V 基础层最大差 0.03906、平均差 0.000993。此次为新初始化 LoRA，B 分支输出为零，因此这组差异已出现在基础线性层输出。把相同 BF16 输入和权重转成 FP32 后计算局部 K 投影，长、短输入在有效位置的 FP32 参考结果一致；两次 BF16 基础层输出相对参考的平均绝对差分别为 0.10488 和 0.04058。

这说明所测 GPU 环境中，填充长度变化已伴随基础 K/V 投影数值变化；它并未确定具体库或底层内核的原因。历史训练使用的每个运行细节是否与本次诊断完全一致仍需核验，因此不能将这一现象等同于历史损失差异的完整解释。
对同一 K 投影单独调用线性运算，关闭 TF32，切换 PyTorch 的 BF16 reduced-precision reduction 设置：

| 设置 | 长短输入结果平均绝对差 | 最大绝对差 |
| --- | ---: | ---: |
| 允许 BF16 reduced-precision reduction | 0.10085 | 4.0 |
| 禁用 BF16 reduced-precision reduction | 0.001344 | 0.5 |
| FP32 输入与权重 | 0.000000184 | 0.0000305 |

这是局部线性层对照，并非整模型重新训练或推理的对照，不能据此断言关闭该设置能修复完整模型结果，也不能推定历史训练时的底层设置。示例脚本运行结束会恢复原 backend 设置。
## 复查

准备主页所述的模型、生成数据及修补后的作者脚本，再运行：

    python scripts/inspect_padding_components.py --trainer /path/to/patched-finetune.py --model /path/to/model --data /path/to/train.jsonl --device cpu
    python scripts/inspect_padding_components.py --trainer /path/to/patched-finetune.py --model /path/to/model --data /path/to/train.jsonl --device cuda --reduction-check

公开脚本已在同一环境分别运行 CPU 和 CUDA 路径，结果与独立诊断脚本的对应数字一致。设备、软件或权重不同可能得到不同结果；不要把这两个样本的观察视为其他模型或设备的普遍结论。

## Where the padding difference first appears

Two samples have 32 valid tokens each. The comparison keeps their input and model weights fixed and changes only right padding from 128 tokens to 32. On the tested GPU, the input normalization and rotary position embeddings match exactly at valid positions; the first observed difference is at the first attention module output. With SDPA, its maximum and mean absolute differences are 1.3984 and 0.02513. Eager attention also differs. Subsequent normalization and feed-forward outputs differ as well.

The attention module includes Q/K/V projections and the output projection, so this observation does not identify a particular kernel as the cause. On CPU, each attention implementation produced identical valid-token hidden states and logits for both padding lengths. The tested model revision was not recorded. The public inspection script has been rerun on both CPU and CUDA in this environment, matching the corresponding diagnostic results. See the commands above and the [earlier precision investigation](PADDING-AND-PRECISION.md).
Within the first layer, valid-token inputs to the K and V projections match. Their base linear outputs nevertheless differ between the padded and trimmed calls: the K output has maximum absolute difference 4.0 and mean 0.10075; V has maximum 0.03906 and mean 0.000993. The newly initialized LoRA B outputs are zero, so the observed difference is already present in the base linear outputs. A local FP32 K projection from the same BF16 inputs and weights matches across both lengths; mean absolute errors of the BF16 base outputs against that reference are 0.10488 and 0.04058, respectively. These observations do not identify a specific low-level cause, and do not establish the complete cause of the historic training difference.
For a separate local K linear projection on the same inputs and weights, with TF32 disabled, allowing BF16 reduced-precision reduction gave a long/short mean absolute difference of 0.10085 (maximum 4.0); disabling it reduced the mean to 0.001344 (maximum 0.5). FP32 inputs and weights gave a mean difference of 1.84e-7. This is a local operation comparison, not an end-to-end model fix or proof of historical training settings. The optional check restores the original backend settings after it runs.
