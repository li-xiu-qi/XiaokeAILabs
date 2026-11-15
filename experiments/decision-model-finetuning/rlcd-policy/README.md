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
| torch | 2.14.0+cu130（aarch64 CUDA 13 wheel） |
| laya | 0.4.1（历史记录，精确提交未核实） |

## 数据准备与运行

在本目录准备独立的 Linux Python 环境，先按设备安装兼容 CUDA 的 PyTorch，再安装数据依赖和上游 Laya 源码。下面固定的提交是公开源码中包含 `laya.train` 的一个版本，**并非已确认的历史运行版本**；这组安装命令尚未端到端验证，运行前需检查导入与训练接口：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 按 https://pytorch.org/get-started/locally/ 选择并安装 CUDA 版 PyTorch
python -m pip install -r requirements.txt
python -c "import torch, datasets; from laya.agent import _fix_tokenizer_config; from laya.train import TrainConfig, finetune; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()"
```

安装需要 Git 和网络；先确保测试命令显示 CUDA 可用。`laya` 及其传递依赖没有完整锁定，历史运行使用的 Laya 精确提交尚未核实；若接口或依赖不兼容，需要排查后再运行。准备数据后将脚本中的解释器指向该环境：

```bash
python scripts/prepare_data.py --out-dir data
export PYTHON="$(command -v python)"
bash scripts/run_ab_compare.sh
```

数据下载和默认模型下载均未锁定 revision；重跑前应记录上游版本与文件哈希。训练入口会调用上游的 tokenizer 配置修复函数，该函数可能修改模型目录中的 `tokenizer/tokenizer_config.json`；默认下载目录是 Hugging Face 共享缓存。若需保护共享缓存，先准备一份独立模型副本，再运行 `MODEL_DIR=/path/to/model bash scripts/run_ab_compare.sh`；直接运行训练入口时使用 `--model-dir`。脚本默认使用当前环境中的 `python`，也可设置 `PYTHON` 指定解释器；在 Linux 上按 `/proc/meminfo` 检查可用内存，默认至少 50 GiB，可通过 `MIN_AVAILABLE_GIB` 调整。该检查只在启动前执行，不是训练期间的持续内存监控。为避免覆盖，已有 `runs/rlcd`、`runs/soft-ce` 或对应日志时脚本会退出。上游当前源码和历史运行的精确提交未核实，上述命令是复现路径，不保证数值完全一致。

数据准备已在 Linux aarch64、Python 3.12 和 datasets 5.2.0 下运行，生成 1200 条训练案例及 400 条测试案例；此次没有验证 RLCD 训练。数据准备会在下载前检查并排他创建两份 JSONL；任一目标已存在时不会覆盖。若下载或转换中断，输出可能为空或不完整，保留需要的文件后选择新目录重新准备。`run_ab_compare.sh` 固定读取本目录的 `data/typed_decisions_train.jsonl` 和 `data/typed_decisions_test.jsonl`；改用其他输出目录时，须先确认数据完整，再在一个新的实验工作副本中放到这两个路径。

## 文件

```text
rlcd-policy/
├── scripts/
│   ├── prepare_data.py        # typed-decisions → {state, questions, gold} JSONL
│   ├── run_finetune.py        # 单卡训练入口（--loss rlcd / soft-ce 对照臂）
│   └── run_ab_compare.sh      # 一键 A/B 对照（两臂串行 + 并排对比）
├── data/                      # 运行时准备的 JSONL 数据集
├── evidence/                  # 已保存的三组实验记录
└── runs/                      # 运行时生成的 checkpoint 与 train_report.json
```

## 已跑出的结果（seed 0，held-out 2000 题）

RLCD 两次运行的 top-1 参考标签一致率为 0.7510 和 0.7700；soft-ce 单次为 0.7855。数据来自合成案例，标签是教师模型生成的软分布，并非真实事件结果；ECE 和 brier 也针对参考分布，不能证明现实概率校准。现存上游实现按展开后的题目划分训练与校准，同案例可能跨越两者。两次 RLCD 运行
记录的配置和 seed 相同，但差异原因尚未定位。现有重复次数不足以判断两种损失的普遍优劣，
也不足以证明它们等价。RLCD 两次相对于参考标签的 ECE 数值低于 soft-ce，仍需多次重复才能判断稳定性。
与外部结果的比较受评测口径及可追溯证据限制，不能将标注者一致性直接视为能力上限。
详见 [RLCD 微调报告](../../../docs/zh/decision-models/RLCD微调报告.md)；历史对照约 54 分钟，重跑耗时取决于环境。

上游：laya 仓库 `github.com/NandhaKishorM/laya`（Apache-2.0），数据集
`huggingface.co/datasets/LocalLLaMA/typed-decisions`，权重
`huggingface.co/convaiinnovations/laya`。
