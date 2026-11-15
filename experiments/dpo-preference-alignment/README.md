# DPO 偏好对齐

[English](README.en.md)

本实验用已有对话模型和公开的回答偏好对，测试直接偏好优化（DPO）能否改变模型对两种回答的相对概率。原理、实验设计和结果边界见 [DPO：直接偏好优化](../../docs/zh/alignment/DPO直接偏好优化.md)；[语言模型全链路](../llm-lifecycle-fullchain/README.md) 展示了另一种使用奖励和采样的对齐训练。

## 环境准备

需要 Linux、Python 3.12、支持 BF16 的 NVIDIA CUDA GPU，以及足够存放模型、数据和训练产物的磁盘及设备内存。精确最低内存尚未验证。冻结参考模型和待训练模型共用底座，通过禁用 LoRA 计算参考分数。先按设备安装兼容的 CUDA PyTorch，再安装其余依赖：

```bash
cd experiments/dpo-preference-alignment
python3.12 -m venv .venv
source .venv/bin/activate
# GB10 / Linux aarch64：复现本次使用的 PyTorch 版本；其他设备按官方说明选匹配构建
python -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.15.0+cu132"
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
```

约束文件用于防止安装其他依赖时替换已安装的 PyTorch。2026-11-15 在 NVIDIA GB10（Linux aarch64）新建虚拟环境，按上述命令安装后，`pip check`、依赖导入和 CUDA/BF16 检查均通过。随后又从新的空缓存开始下载固定模型和数据，并在同一新环境完成 64 步训练及 128 对评估。可用下方脚本命令重跑；每次都要使用新的外部目录。该验证覆盖 Linux aarch64 / Python 3.12.3 / CUDA 13.2 这组组合，requirements 不是完整锁文件，传递依赖仍由 pip 解析。

```bash
export XL_DATA_DIR=/path/to/dpo-clean-data/run-001
export XL_OUT_DIR=/path/to/dpo-clean-output/run-001
export XL_RUN_ID=run-001
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/clean_env_gpu_smoke.sh
```

脚本会在 `XL_DATA_DIR` 下创建新的虚拟环境、Hugging Face 缓存和数据文件，在 `XL_OUT_DIR` 下保存模型适配器与报告；发现已有目标时拒绝覆盖。

## 模型和数据

- 模型：[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`，Apache-2.0。它是已有后训练的对话模型，不是 Base 模型。
- 数据：[HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)，固定 revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`，数据卡声明 MIT。原始评分由模型产生，不等于人工直接标注；数据卡记录过早期版本的错误标签和评测污染问题。
- 从官方 `train_prefs` 和 `test_prefs` 中按固定种子抽取小样本，检查提示词在选中样本之间不重叠，并保留输入元信息。未排除近重复或预训练数据重叠。

本次从 Hugging Face 镜像获取固定 revision，模型和两份偏好分片的实际 SHA-256 均与对应官方仓库的 LFS 元数据一致：

| 文件 | SHA-256 |
|---|---|
| 模型 `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| 数据 `train_prefs-00000-of-00001.parquet` | `0f951ca4502001d31f3e4c70716ae51d20e4ce4f847d12b6a6695a40d4d353a8` |
| 数据 `test_prefs-00000-of-00001.parquet` | `e9dab2789f419d4204d73ec2c860af6d88d466b906e0109e69b96075467eb389` |

## 运行

以下命令会下载模型和数据，训练 LoRA 适配器，评估固定候选答案的概率差。使用空输出目录；数据与模型缓存不提交版本库。

```bash
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/run-01"
export HF_HOME="$XL_DATA_DIR/huggingface"
python scripts/run_dpo.py --train-size 256 --test-size 128 --steps 64 --seed 42
```

下载后可将文件的输出哈希与上表比较：

```bash
sha256sum "$XL_DATA_DIR/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
sha256sum "$HF_HOME/hub/datasets--HuggingFaceH4--ultrafeedback_binarized/snapshots/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/data/"{train_prefs,test_prefs}-00000-of-00001.parquet
```

执行完成后查看 `$XL_OUT_DIR/report.json`。输入提示词保留末尾最多 192 个 token，回答保留开头最多 191 个 token 并附加结束符；截断可能丢掉影响偏好的信息。截断后两回答完全相同的样本被剔除，报告记录各类截断与有效样本数，但截断也可能改变剩余样本的偏好含义。评估按回答 token 的对数概率之和比较给定回答对，受长度影响，不测实际生成质量、用户满意度或安全性。短训练、单种子和小测试集不能证明一般对齐能力改善。运行前请检查模型和数据的使用许可及资源需求。

`HF_HOME` 将 Hugging Face 的通用缓存定向到数据目录，避免默认用户缓存散落文件。2026-11-15 的全新环境复跑验证了空缓存中的模型和数据下载，以及后续完整训练评估。

## 运行状态

2026-11-15 在 Linux aarch64、NVIDIA GB10 上，使用 Python 3.12、PyTorch 2.15.0+cu132、Transformers 5.16.1、datasets 5.2.0、PEFT 0.21.3 和 Accelerate 1.15.0，完成固定模型与数据的 64 次 CUDA LoRA 更新及 128 对测试。较早的训练使用从已有环境复制的独立环境；本次额外复跑则从新的虚拟环境和空缓存开始，下载数据、模型并完成全流程。初始训练报告中准确率仍为 71/128 → 71/128，准确率变化 bootstrap 95% 区间为 [0, 0]；平均参考相对分差变化为 +0.174。此前一轮同种子运行记录为 +0.136，显示该数值在当前实现和硬件下并未实现逐次完全一致，因此不声称确定性复现。PyTorch 报告的峰值 CUDA 张量分配约 3.60 GiB，不是整机内存或通用硬件下限。干净环境运行的脱敏细节见[证据记录](evidence/clean-env-gpu-smoke-2026-11-15.json)。

测试集按两份固定回答的总对数概率比较，微调前后均有 71/128 对与数据偏好方向一致，逐对预测方向没有变化。配对 bootstrap 的准确率差值 95% 区间为 [0, 0]；这表示本次样本中没有预测翻转，不表示未来数据上的差值必为零。训练后的参考相对平均分差改变 +0.136，但不足以据此认定生成质量改善。训练集提示词截断 69/256，偏好与非偏好回答分别截断 128/256 和 116/256；测试集分别为 21/128、69/128 和 66/128。没有样本因截断后两回答完全相同而被剔除，截断仍可能改变其他样本的偏好含义。输入数据的评分来自模型，小样本、单种子、长度偏差、近重复和预训练重叠限制结论。首步平均 DPO 损失为 0.693147，符合初始参考相对分差接近零的预期；另一次使用同一模型与 LoRA 配置新建初始适配器，对这 128 个测试对的 256 个回答逐一比较启用和禁用适配器时的 logits，最大绝对差为 0；这是训练后的独立初始化检查，并非训练开始时逐项留存的输出。

指标和参数见 [summary.json](evidence/summary.json)，解释见[实验讲解](../../docs/zh/alignment/DPO直接偏好优化.md)。

## 参考资料

- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.
- Cui et al., [UltraFeedback](https://arxiv.org/abs/2310.01377), 2023.
