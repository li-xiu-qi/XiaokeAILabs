# laya RLCD 微调复现

## 这个实验是什么

在 DGX Spark（GB10，128 GB 统一内存，aarch64）上单卡复现 laya 的 RLCD 微调：从公开的
typed-decisions 数据集与 laya English checkpoint 出发，用 RLCD（proper scoring rule 奖励
的 GRPO 风格策略梯度加 soft CE）训练一个决策模型，对照 Jev 官方数字验证可复现性。

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
| Python | 3.12（venv `~/laya-rlcd-env`） |
| torch | 2.14.0+cu130（aarch64 CUDA 13 wheel） |
| laya | 0.4.1（仓库 main，tarball 落位 `~/laya-main`） |

## 文件

```
laya_rlcd_train/
├── scripts/
│   ├── prepare_data.py        # typed-decisions → {state, questions, gold} JSONL
│   ├── run_finetune.py        # 单卡训练入口（--loss rlcd / soft-ce 对照臂）
│   └── run_ab_compare.sh      # 一键 A/B 对照（两臂串行 + 并排对比）
├── data/                      # JSONL 数据集
├── docs/RLCD微调报告.md       # 主报告（唯一结论来源）
└── runs/                      # 训练产物（checkpoint、train_report.json）
```

## 已跑出的结果（seed 0，held-out 2000 题）

RLCD 臂两次运行准确率 0.751 与 0.770（同配方重跑差 1.9 点，GPU 训练非确定性），超过
Jev 官方 0.727 与教师上限 0.735，官方 2xT4 的 0.766 落在两次实测区间内。A/B 对照
（唯一变量为 loss）中 soft-ce 纯蒸馏 0.7855 数值略高，但差距小于运行间波动，无证据
表明 RLCD 策略梯度项在此数据集带来准确率增益（ECE 上 rlcd 略优，仅方向性观察）。
详见 `docs/RLCD微调报告.md`；重跑对照用 `bash scripts/run_ab_compare.sh`，约 54 分钟。

上游：laya 仓库 `github.com/NandhaKishorM/laya`（Apache-2.0），数据集
`huggingface.co/datasets/LocalLLaMA/typed-decisions`，权重
`huggingface.co/convaiinnovations/laya`。
