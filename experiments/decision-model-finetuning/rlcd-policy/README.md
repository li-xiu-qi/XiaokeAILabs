# laya RLCD 微调复现

[English](README.en.md)

## 为什么做这个实验

在分类、打分或选项决策中，模型除了给出答案，还可能需要输出概率，供下游按风险选择行动。本实验比较两种训练方式：RLCD 在蒸馏损失之外加入基于评分规则的策略梯度，soft-ce 只保留蒸馏损失。问题是前者在同一数据集上是否能带来更好的准确率或概率校准。

在 DGX Spark（GB10，128 GB 统一内存，aarch64）上单卡复现 laya 的 RLCD 微调：从公开的
typed-decisions 数据集与 laya English checkpoint 出发，用 RLCD（proper scoring rule 奖励
的 GRPO 风格策略梯度加 soft CE）训练一个决策模型，比较其对参考分布的拟合。

RLCD 是 TypeSafe Jev 官方的训练法（Reinforcement Learning for Calibrated Decisions）。laya
是目前开源生态里把 RLCD 训练代码完整放出来的项目，训练实现在 `laya/train.py`，notebook
在 laya 仓库 `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`（官方路径是
Kaggle 免费 2xT4 DDP）。本实验把它改成单卡 CUDA 路径（无 DDP）。

## 配方（对齐 notebook）

| 项 | 值 | 来源 |
|---|---|---|
| 底座 | `convaiinnovations/laya` English checkpoint（ModernBERT-large，421M） | notebook cell 6 |
| 有效 batch | 64 = micro_batch 8 x grad_accum 8 | train.py TrainConfig 注释（notebook 是 8x2GPUx4accum） |
| loss | rlcd（默认）：GRPO 风格高斯噪声采样 + proper scoring rule 奖励 + soft CE | train.py rlcd_loss |
| rl_samples / sigma | 4 / 0.4 → 0.1 线性退火 | TrainConfig 默认 |
| w_sph / w_rps | 0.75 / 1.0 | TrainConfig 默认 |
| epochs | 4 | notebook |
| max_len / head_max_len | 512 / 192（English checkpoint 自带） | rl_agent_config.json |
| AMP / gradient checkpointing | 开（CUDA 默认） | train_model |
| 校准 | calib_frac 0.1，calib_max 400，每问题类型一个温度 | TrainConfig 默认 |
| 数据 | LocalLLaMA/typed-decisions：train 1200 案例 / test 400 案例（held-out 评估） | HF 数据集 |

## 运行环境

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA DGX Spark（GB10 Grace Blackwell，121 GB 统一内存，aarch64） |
| 系统 | Ubuntu 24.04 LTS |
| Python | 3.12 |
| torch | 2.15.0+cu132（本次新环境实测） |
| datasets / huggingface_hub | 5.2.0 / 1.33.0 |
| laya | 源码提交 `3a38cbe83f6ef556f3125fb0065fdc5a55d05792`，安装后版本 0.4.31 |

## 数据准备与运行

下面的组合已在 DGX Spark（Linux ARM64、Ubuntu 24.04、Python 3.12.3）新建的独立环境中完成安装、依赖检查和完整 GPU A/B 训练。其他设备应先按其架构与 CUDA 版本选择 PyTorch；不能直接照搬 GB10 的 wheel。Laya 固定到公开源码提交，数据集和模型也固定到 Hugging Face commit：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 已实测的 DGX Spark / Linux ARM64 组合；其他设备按 PyTorch 官方说明换用兼容 wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.15.0+cu132"
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, datasets, huggingface_hub; from laya.agent import _fix_tokenizer_config; from laya.train import TrainConfig, finetune; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), datasets.__version__, huggingface_hub.__version__); assert torch.cuda.is_available()"
```

`requirements.txt` 固定了直接依赖和 Laya 源码提交，但没有锁定全部传递依赖；其他安装时间可能得到不同的间接依赖版本。PyTorch CUDA wheel、数据集和模型需要网络下载。

```bash
export DATASET_REVISION=cc1a2e795f5160d417d4d7131a71d1ab60ea33ca
export MODEL_REVISION=85cc74ddae6109077ebe2853a9b3077fb6606a78
export XL_DATA_DIR="$PWD/data/typed-decisions-$DATASET_REVISION"
export XL_OUT_DIR="$PWD/runs/rlcd-ab-$DATASET_REVISION"
export MODEL_DIR="$PWD/models/laya-$MODEL_REVISION"
mkdir -p models
test ! -e "$MODEL_DIR"
MODEL_DIR="$MODEL_DIR" MODEL_REVISION="$MODEL_REVISION" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "convaiinnovations/laya",
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_DIR"],
    allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
)
PY
python scripts/prepare_data.py --revision "$DATASET_REVISION" --out-dir "$XL_DATA_DIR"
export PYTHON="$PWD/.venv/bin/python"
bash scripts/run_ab_compare.sh
```

模型下载到独立目录，因为上游 tokenizer 修复函数可能改写 `tokenizer/tokenizer_config.json`。数据和运行结果也用独立目录；请为每次运行更改 `XL_DATA_DIR` / `XL_OUT_DIR`，脚本不会覆盖已有的 arm 输出。Linux 启动前检查可用内存，默认至少 50 GiB，可通过 `MIN_AVAILABLE_GIB` 调整；这是启动门槛，不是训练期间的持续监控。完整训练每臂约需 26 分钟，两臂串行运行约 52 分钟，实测时间受设备和系统负载影响。

数据准备会先检查并排他创建两份 JSONL；任一目标已存在就停止。如果下载或转换中断，可能留下不完整文件，保留需用内容并换一个新目录重试。`run_ab_compare.sh` 通过 `XL_DATA_DIR` 和 `XL_OUT_DIR` 使用指定数据与结果目录。此次固定修订生成的训练集和测试集 SHA-256 分别为 `6c84f9a369e25a14c5ae47fc567cc86553f56bac128efb48127a94971d77dcc2` 与 `0c6712fc6657dca6371ab96863b3c8f1306a67631141623ea98d066fabbd58c4`；复现后可用 `sha256sum` 核对。

## 文件

```text
rlcd-policy/
├── scripts/
│   ├── prepare_data.py        # typed-decisions → {state, questions, gold} JSONL
│   ├── run_finetune.py        # 单卡训练入口（--loss rlcd / soft-ce 对照臂）
│   └── run_ab_compare.sh      # 一键 A/B 对照（两臂串行 + 并排对比）
├── data/                      # 运行时准备的 JSONL 数据集
├── evidence/                  # 已保存的运行记录及干净环境 GPU A/B 证据
└── runs/                      # 运行时生成的 checkpoint 与 train_report.json
```

## 2026-11-15：新环境完整 GPU A/B 结果

在新建的隔离环境中，RLCD 和 soft-ce 各训练 4 个 epoch，并在同一组 2,000 道 held-out 问题上评估。两臂使用相同 seed、模型、数据和训练超参，唯一预期差异是 loss。每臂完成 352 次优化器更新。

| 指标（微调后） | RLCD | soft-ce | 更低者 |
|---|---:|---:|---|
| 与教师参考标签 top-1 一致率 | 0.7220 | 0.7815 | — |
| ECE | 0.1071 | 0.1464 | RLCD |
| 多分类 Brier | 0.0753 | 0.0508 | soft-ce |
| top-1 Brier | 0.1792 | 0.1685 | soft-ce |
| 平均置信度 | 0.6151 | 0.6351 | — |
| 最终评估 loss | 0.9019 | 0.8581 | soft-ce |

这次单次对照中，soft-ce 的准确率和 Brier 更好，RLCD 的 ECE 更低；每种方法只有一个新增运行，不能据此判断稳定优势。数据为合成案例，软标签由教师模型生成，指标反映对参考分布的拟合，不能证明真实场景的概率校准。上游训练代码还报告 `action`、`urgency` 和 `disposition` 的 schema 冲突，并保留首次遇到的定义；RLCD 代码报告学习率调度器先于优化器更新的警告，可能跳过首个学习率值。两臂还出现 Transformers 梯度检查点 API 弃用警告。证据见[干净环境完整 GPU A/B 记录](evidence/clean-gpu-ab-2026-11-15.json)。

## 既有运行结果（seed 0，held-out 2000 题）

早期记录中两次 RLCD 运行的 top-1 参考标签一致率为 0.7510 和 0.7700；一次 soft-ce 运行是 0.7855。这些早期记录没有锁定本次使用的数据集和模型 revision，故与上面的固定 revision 新环境对照分开保留，不直接合并比较。数据来自合成案例，标签是教师模型生成的软分布，并非真实事件结果；ECE 和 Brier 也针对参考分布，不能证明现实概率校准。现存上游实现按展开后的题目划分训练与校准，同案例可能跨越两者。两次早期 RLCD 运行记录的配置和 seed 相同，但差异原因尚未定位。把早期记录与本次结果合起来看，运行次数仍不足以判断两种损失的普遍优劣或稳定性，也不足以证明它们等价。
与外部结果的比较受评测口径及可追溯证据限制，不能将标注者一致性直接视为能力上限。
详见 [RLCD 微调报告](../../../docs/zh/decision-models/RLCD微调报告.md)；历史对照约 54 分钟，重跑耗时取决于环境。

上游：laya 仓库 `github.com/NandhaKishorM/laya`（Apache-2.0），数据集
`huggingface.co/datasets/LocalLLaMA/typed-decisions`，权重
`huggingface.co/convaiinnovations/laya`。
