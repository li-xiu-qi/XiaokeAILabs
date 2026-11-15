# LLM 全链路训练与微调可复现实验（llm-lifecycle-fullchain）
[English version](README.en.md).


参考 [karpathy/nanochat](https://github.com/karpathy/nanochat) 的全栈流程重构：从零跑通大语言模型生命周期的八个环节：**BPE 分词器（GPT-4 风格正则预分词）→ 现代架构因果语言模型 → Muon 预训练（附 AdamW 基线）→ 对话格式 midtrain → 指令微调（SFT）→ GRPO 强化学习 → KV cache 生成 → 全链路评估**。核心算法与训练脚本零硬编码语料，训练数据通过 `dataset.py` 流式接入，离线冒烟语料由 `examples/` 独立生成，域偏差评测切片固定存放在 `fixtures/`。

同一套代码提供 **CPU、单卡 CUDA、多卡 DDP** 路径（`torchrun --nproc_per_node=2` 单机双卡，或 `nnodes=2` 双机）。Windows 官方 wheel 无 NCCL 时自动选择 gloo，但本次实测版本的 CUDA DDP 存在限制，见下文「GPU 运行」中的已知环境限制。代码还支持 `--amp bf16/fp16` 混合精度、`--grad-accum` 梯度累积与 `--scale` 批大小缩放，默认小规模 CPU 冒烟口径不变。

## 极速上手

如果你是从教程《一台设备带你走完 LLM 全生命周期》跳转而来，本实验支持**开箱即用、一键复现**。整套链路在单机物理机上即可完整闭环：

### 1. 快速运行命令

```bash
# 1. 克隆代码并安装基础依赖
git clone https://github.com/li-xiu-qi/XiaokeAILabs.git
cd XiaokeAILabs/experiments/llm-lifecycle-fullchain
pip install -r requirements.txt

# 2. 一键跑通全链路（按硬件选一个 profile）：
# 选项 A：CPU 极速冒烟（约 2 分钟，无需显卡，快速跑通全链路代码流）
bash run.sh --config configs/smoke.json

# 选项 B：单卡 GPU 实测（单卡 CUDA + BF16，对齐文章中单机实战口径）
bash run.sh --config configs/gpu-smoke.json
```

### 2. 实战环节与脚本对照表

| 阶段 | 对应脚本 | 核心任务与输出 |
|---|---|---|
| **1. 数据准备** | `scripts/prepare_data.py` | 一键下载真实开源语料（Wikitext-2 / Alpaca / GSM8K），或传 `--source toy` 离线快速冒烟 |
| **2. 训练分词器** | `scripts/01_train_tokenizer.py` | 训练 Byte-Level BPE 分词器，导出 `tokenizer.json` 与 `token_bytes.pt` 缓存 |
| **3. 基础预训练** | `scripts/03_pretrain.py` | 预训练模型自监督收敛，对比 Muon 与 AdamW 曲线（可先跑 `02_model.py` 算子等价性自检） |
| **4. 对话中训练** | `scripts/04_midtrain.py` | 低学习率对话续训，平滑过渡概率分布，建立多轮问答语感 |
| **5. 监督微调 SFT** | `scripts/05_sft.py` | 基于 ChatML 模板与 Prompt 掩码（`-100`），让模型学会按角色规矩答题 |
| **6. 强化学习 GRPO** | `scripts/06_rl_grpo.py` | 免 Critic 架构，组内 8 候选采样，规则精确匹配打分，计算相对优势优化策略 |
| **7. 推理生成** | `scripts/07_generate.py` | 贪婪/温度采样验证，实测对比开启与关闭 KV Cache 的生成提速倍率 |
| **8. 全链路评估** | `scripts/08_eval_report.py` | 跨 Pretrain、Midtrain、SFT、GRPO 四阶段拉齐测试集，输出综合客观评估报告 |

## 运行契约

本实验采用「源码配置与运行产物分离」契约（实现见 `scripts/llm_common.py`）：

| 契约项 | 环境变量 | 说明 |
|---|---|---|
| 输出目录 | `XL_OUT_DIR` | 指标、checkpoint、曲线、报告全部落此目录（未设置时默认写 `.work/out/<run_id>`） |
| 数据目录 | `XL_DATA_DIR` | 语料读写目录（未设置时默认使用 `.work/data`） |
| 运行标识 | `XL_RUN_ID` | 写进每个阶段的指标与 manifest（`run.sh` 默认 `local-<时间戳>`，单步脚本默认 `local`） |

直接克隆到本机即可开箱运行；`.work/` 已被 `.gitignore` 忽略，不会污染提交。如需将数据和产物放到仓外独立磁盘或 NAS，显式设置 `XL_OUT_DIR` / `XL_DATA_DIR` / `XL_RUN_ID` 即可。若在 CI 或受控环境中设置了 `XL_REQUIRE_EXTERNAL_ENV=1`，脚本将禁止本地默认回退并强制要求外部注入上述三个变量。

本目录保存实验代码、配置与结果证据。对应理论、数学推导与实验分析见专文：[大语言模型从零构建与训练全链路](../../docs/zh/inference/大语言模型从零构建与训练全链路.md)。如何阅读 step、token 与训练耗时，见[模型训练的计量单位](../../docs/zh/inference/模型训练的计量单位.md)。指标汇总见 `evidence/fullchain-eval-report-toy-synthetic.md`（收敛后的实验报告副本）。

偏好对训练的另一种路径见 [DPO 偏好对齐](../dpo-preference-alignment/README.md)，使用固定偏好对，不在更新时采样。

## LLM 体系认知导航

本项目在 `experiments/` 下构建了三个不同抽象层级的因果语言模型实测体系：

| 层次 | 实验目录 | 对应专文与深度剖析 | 核心关注点与方法论 |
|---|---|---|---|
| **一阶：微观单点机制** | [`llm-mechanics`](../llm-mechanics/README.md) | [语言模型生成机制与 KV 缓存](../../docs/zh/inference/语言模型生成机制与KV缓存.md) | 单步前向原理解剖：BPE/BBPE 字节切分、单步采样概率演化、KV Cache 数值严格等价性验证、低比特量化精度损失。 |
| **二阶：全生命周期闭环** | **本目录（`llm-lifecycle-fullchain`）** | [大语言模型从零构建与训练全链路](../../docs/zh/inference/大语言模型从零构建与训练全链路.md) | 端到端训练动力学与对齐跃迁：流式分词训练、因果架构自检、Muon 优化器预训练、对话格式适配、SFT 指令对齐、GRPO 可验证奖励强化学习。 |
| **三阶：宏观系统性能** | [`llm-inference-perf`](../llm-inference-perf/README.md) | [投机解码收益与实测条件](../../docs/zh/inference/投机解码收益与实测条件.md) | 工业级硬件实测与极端加速：GB10 统一内存平台独占实测、27B/125B 投机解码加速衰减、同源 MTP 与独立 DFlash2 草稿头、长文本与任务类型边界。 |

## 与 nanochat 的对应关系

nanochat 用单节点 8×H100 在约 3 小时内训练出 GPT-2 水平的对话模型；本实验把它的**流程与关键技术**等比例缩放到 CPU 玩具规模（约 2.5M 参数、37KB 合成语料），保留可复现的教学链路：

| nanochat | 本实验 | 说明 |
|---|---|---|
| `nanochat/dataset.py` | `scripts/dataset.py` | 标准数据流式批次读取，支持 Parquet 与文本，支持 max_chars / doc_cap 截断 |
| `scripts/tok_train.py` + tiktoken | `01_train_tokenizer.py` | Byte-Level BPE + GPT-4 风格正则预分词（`tokenizers` 库实现，缓存 token_bytes） |
| `nanochat/gpt.py` | `scripts/llm_common.py` 模型部分 | RoPE、QK norm、relu² MLP、无 bias、untied 权重、嵌入后 norm、logit softcap、GQA 代码路径 |
| `nanochat/optim.py` MuonAdamW | `llm_common.Muon` + `build_optimizer` | 隐藏矩阵走 Muon（Newton-Schulz 正交化），嵌入/lm_head 走 AdamW |
| `scripts/base_train.py` | `03_pretrain.py` | 预训练 + 同 steps 同种子的 AdamW 基线对照 |
| midtraining 阶段 | `04_midtrain.py` | 对话格式适配（低 lr 续训） |
| `scripts/chat_sft.py` | `05_sft.py` | 指令微调，prompt 掩码 |
| `scripts/chat_rl.py` + `tasks/` | `06_rl_grpo.py` | GRPO：组采样、可验证精确匹配奖励、KL 锚定冻结参考模型 |
| `nanochat/engine.py` KV cache | `llm_common.KVCache` | 生成走 KV cache，与 naive 前向逐 token 一致（02 自检） |
| `scripts/base_eval.py` / report | `07_generate.py` + `08_eval_report.py` | 解码对照、温度效应、提速对比、四阶段评估矩阵 |

nanochat 中面向大规模的基础设施（FSDP、FP8、FlashAttention-3、Web chat UI）不在本实验范围内；torchrun DDP 已按玩具规模支持（单机双卡与双机四卡，见 GPU 运行一节）。

## 链路总览

```
$XL_DATA_DIR/corpus.txt ──► 01_train_tokenizer.py ──► $XL_OUT_DIR/tokenizer/tokenizer.json
   (+ sft.jsonl / rl.jsonl)                                │
                                                           ▼
                                                  llm_common.py（模型 + Muon + KV cache）
                                                           │
      ┌──────────────────┬───────────────────────────────┼──────────────────────┐
      ▼                  ▼                               ▼                      ▼
 03_pretrain.py     04_midtrain.py                  05_sft.py            06_rl_grpo.py
 Muon vs AdamW      对话格式适配                    指令微调            GRPO 可验证奖励
      │                  │                               │                      ▼
      ▼                  ▼                               ▼                 rl.pt
 pretrain.pt         midtrain.pt                      sft.pt                   │
 (pretrain_adamw.pt)          └───────────────┬───────────────┘                   │
                                             ▼                                   ▼
                                  07_generate.py（贪心/温度/KV cache 提速） ◄──────┘
                                             │
                                             ▼
                                  08_eval_report.py ──► $XL_OUT_DIR/report.md + summary.json
```

## 环境要求

- Python ≥ 3.10，依赖见 `requirements.txt`（实测：torch 2.15.0+cpu、tokenizers 0.23.3、matplotlib 3.11.2）
- GPU 运行需要带 CUDA 的 torch（实测 2.14.0+cu126，RTX 3080 Laptop）；`--device auto` 时检测到卡即用卡，否则 CPU
- 分布式后端 auto 策略：CUDA 且编译了 NCCL 用 nccl，否则 gloo；Windows 官方 wheel 未编译 NCCL 时选择 gloo；CUDA DDP 的可用性取决于具体环境，见下文已知限制
- 支持公开标准数据集流，离线冒烟测试语料由 `examples/generate_toy_dataset.py` 独立生成
- 运行环境清单写入 `$XL_OUT_DIR/run_manifest.json`（含 run_id / profile / device / amp / nproc 等字段）
- 多核机器上小模型默认 4 线程最快（实测 8 线程因争用更慢），可用 `LLM_THREADS` 环境变量覆盖

在本实验目录准备 Python 环境。以下示例使用 Linux 或 WSL 的 Bash；GPU 运行前先按 [PyTorch 安装指南](https://pytorch.org/get-started/locally/)选择适合平台和 CUDA 环境的 PyTorch 构建，再安装其余依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 如需 GPU，在此先按 PyTorch 安装指南安装带 CUDA 的 torch
python -m pip install -r requirements.txt
# 可选：分阶段读取 Parquet 语料
python -m pip install pyarrow
python -c "import torch, tokenizers; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'tokenizers:', tokenizers.__version__)"
```

依赖清单包含 `matplotlib`，它用于绘图，缺失时会跳过 PNG 输出；两份专题绘图脚本还会使用 NumPy，安装 `matplotlib` 时通常会一起安装。`matplotlib_layout_audit` 是可选的外部版面检查工具，缺失时会跳过审计，不影响出图。多进程运行时还需让 `torchrun` 与所用 Python 环境一致，例如在启动前激活安装了 PyTorch 的虚拟环境。

一键脚本默认使用 `$XL_DATA_DIR`（未设置时为 `.work/data`）中的数据。若该目录没有 `corpus.txt`，脚本会生成玩具数据，同时写入 `corpus.txt`、`sft.jsonl` 和 `rl.jsonl`，因此可能覆盖目录中已有的同名 SFT 或 RL 文件；提供自定义语料时请一并准备三个文件，JSONL 字段可参照 `examples/generate_toy_dataset.py`。`run.sh` 不接收外部数据源参数，使用 Parquet 数据需分阶段调用支持该格式的脚本。

### CPU smoke 运行记录

2026-11-12 的一次运行使用 `smoke` 配置，状态记录的退出码为 0；日志到达 `[8/8]` 全链路评估并写入报告。运行清单记录：Python 3.12.3、PyTorch 2.11.0+cu130、tokenizers 0.22.2、matplotlib 3.11.0；`device=cpu`、`nproc=1`、`amp=none`。运行设备为 GB10，CUDA 检测结果为可用，但此次实际使用 CPU。这次记录只验证该次 CPU smoke 流程，不能据此推断 GPU 训练或 `full` 配置已经跑通。运行时的配置和版本可在输出目录的 `run_manifest.json` 查看，报告见同目录的 `report.md`。

### CUDA GPU smoke 运行记录

2026-11-13 使用 [`configs/gpu-smoke.json`](configs/gpu-smoke.json) 在 NVIDIA GB10 上完成八个阶段，进程退出码为 0。环境为 Ubuntu 24.04 ARM64、Python 3.12.3、PyTorch 2.15.0+cu132 和 tokenizers 0.23.3；实际训练设备为 `cuda:0`，使用 BF16 混合精度和单进程。该配置与上面的 CPU smoke 不同：两组预训练各 60 步，midtrain 40 步、SFT 80 步、RL 20 步。

SFT 指标记录了 80 条训练历史，PyTorch 分配的显存峰值为 73.99 MiB，前后各 4 步的平均训练 loss 从 7.7149 降至 0.0873；49 条训练样本之外的 8 条留出样本命中率为 25%。这些是玩具数据上的流程验证，不是模型质量或性能结论。本次没有安装 matplotlib，因此未生成曲线图；运行清单和报告仍写入输出目录。

### 双机 CUDA 玩具数据流程

2026-11-14 使用两台设备各一张 GPU，在 Ubuntu 24.04 ARM64、Python 3.12.3、PyTorch 2.15.0+cu132、tokenizers 0.23.3 的独立环境中，以 NCCL 和 BF16 分阶段运行玩具数据流程。两台分别生成的语料、分词器和后续复制的检查点均核对了 SHA-256；模型自检中的权重广播、梯度一致性和 checkpoint 加载通过。Muon 与 AdamW 各预训练 60 步、对话续训 40 步、SFT 80 步和 GRPO 20 步均以双机 GPU 完成，两端退出码均为 0；分词器准备、生成和评估使用单机进程。评估中 SFT 和 GRPO 后的留出算术命中率均为 9%（47 题），不能据此声称 RL 提升了泛化能力。环境从已有依赖组合复制后分别隔离，`pip check` 通过，并非从空环境安装验证；这是分阶段启动，不是双机一键脚本验证，结果也不代表真实语料效果。

## 一键复现

```bash
# 直接在本机开箱运行（默认写入被 .gitignore 忽略的 .work/；也可预先导出 XL_OUT_DIR / XL_DATA_DIR / XL_RUN_ID）
#   冒烟档（CPU 最小步数，约 2-3 分钟）
bash run.sh --config configs/smoke.json

# GPU 冒烟档（单卡 CUDA + BF16；使用玩具数据验证训练流程）
bash run.sh --config configs/gpu-smoke.json

# 完整档（tokenizer 目标词表 8192 + 标准步数；数据由实际输入决定）
bash run.sh --config configs/full.json

# 指定解释器 / 单机双进程 CPU DDP / 末尾追加分词器多域评测
PYTHON=/path/to/python bash run.sh --config configs/smoke.json
NPROC=2 bash run.sh --config configs/smoke.json
bash run.sh --config configs/smoke.json --with-tok-eval
```

profile 定义在 `configs/<name>.json`：`device` / `amp` / `scale` / `nproc` / `tokenizer.vocab_size` / `tokenizer.max_chars` / `stages.{pretrain,midtrain,sft,rl}` 步数。也可按阶段单步执行（直接跑默认写入 `.work/out/local` 与 `.work/data`，或按需指定 `XL_*` 环境变量）：

`gpu-smoke` 显式要求 CUDA；环境中没有可用 CUDA 设备时会报错，不会回退到 CPU。可按需显式设置正数环境变量 `XL_MIN_AVAILABLE_GIB`，在训练阶段检查 Linux 的系统可用内存；该检查仅支持单进程，阈值无效、无法读取可用内存或低于阈值时会停止训练。它不是显存预留，默认也不会启用。

### 在外部算力设备或独立存储上运行

如果需要把实验部署到独立 GPU 服务器运行，或把语料、权重放到仓库外的磁盘/NAS，建议显式指定三个契约变量，避免大体积数据留在源码目录中：

```bash
export XL_RUN_ID="exp-20261112-full"
export XL_DATA_DIR="/data/llm-corpus"
export XL_OUT_DIR="/data/runs/$XL_RUN_ID"
# 可选：若希望在变量缺失时直接报错而不是回退到 .work/，开启严格模式
export XL_REQUIRE_EXTERNAL_ENV=1

bash run.sh --config configs/full.json
```

运行完成后，重体积权重在 `$XL_OUT_DIR/checkpoints/` 保留在计算设备即可；需要带回对照时，只取 `$XL_OUT_DIR/report.md`、`summary.json`、`*_metrics.json`、`run_manifest.json` 与图表即可。

```bash
python examples/generate_toy_dataset.py # （可选）无外部语料时生成冒烟数据到 $XL_DATA_DIR
python scripts/01_train_tokenizer.py    # 纯流式语料输入，BPE 训练与导出（纯 CPU 阶段）
python scripts/02_model.py             # 模型自检（QK norm/RoPE/KV cache/softcap/GQA/AMP）
python scripts/03_pretrain.py          # 预训练（Muon vs AdamW 基线）
python scripts/04_midtrain.py          # 对话格式 midtrain
python scripts/05_sft.py               # 指令微调
python scripts/06_rl_grpo.py           # GRPO 强化学习
python scripts/07_generate.py          # 生成 + KV cache 提速对比
python scripts/08_eval_report.py       # 全链路评估与 report.md
```

## GPU 运行（单卡与双卡 DDP）

所有训练脚本（02~06）共享同一组设备参数，由 `llm_common.init_distributed` 统一解析；被 torchrun 拉起时自动建组，直接 `python` 启动则退化为单进程，代码零分支。

| 参数 | 取值 | 说明 |
|---|---|---|
| `--device` | auto / cpu / cuda / cuda:N | auto 时检测到卡即用卡；DDP 下每进程绑定 local_rank 对应的卡 |
| `--amp` | none / bf16 / fp16 | 混合精度；fp16 带 GradScaler 且仅 CUDA，CPU 上会明确报错 |
| `--backend` | auto / nccl / gloo | auto：CUDA 且有 NCCL 用 nccl，否则 gloo（Windows 常态） |
| `--grad-accum` | 整数（默认 1） | 梯度累积微批数，等效放大 batch，显存不足时用 |
| `--scale` | 浮点（默认 1.0） | 按比例放大每卡 batch（03/04/05）或每步 prompt 数（06） |

单卡（本机实测 RTX 3080 Laptop，torch 2.14.0+cu126）：

```bash
# 单卡自检（含 AMP 检查：bf16 与 fp16 前向、CPU 上 fp16 按预期拒绝）
python scripts/02_model.py --device cuda

# 单卡 + bf16 混合精度，batch 翻倍
python scripts/03_pretrain.py --device cuda --amp bf16 --scale 2

# 完整档单卡端到端
bash run.sh --config configs/full.json
```

单机双卡 DDP（`torchrun --nproc_per_node=2`）；下列 CUDA 双卡命令尚未在本次环境中端到端验证：

```bash
# 双卡 DDP：02 自检含权重广播一致 / 同 batch 梯度一致 / checkpoint 无 module. 前缀
torchrun --nproc_per_node=2 scripts/02_model.py --device cuda

# 双卡 DDP 训练（各 rank 抽不同 batch，梯度 all_reduce 平均，rank0 评估与保存）
torchrun --nproc_per_node=2 scripts/03_pretrain.py --steps 60 --device cuda --amp bf16

# 一键链路双卡版（NPROC=2 时 02~06 自动走 torchrun，01/07/08 仍单进程）
NPROC=2 bash run.sh --config configs/gpu-smoke.json
```

Windows 无 NCCL 时把 `--device cuda` 换成 `--device cpu` 即可用 gloo 在 CPU 上验证同样的 2-rank 语义（本机已实测）：

```bash
torchrun --nproc_per_node=2 scripts/02_model.py --device cpu
```

双机双卡（`nnodes=2`，涉及至少两台设备各出 1 卡）在 `torchrun` 语义下与单机双卡一致，适合把计算分摊到两台独立主机。跨节点启动需要两台设备满足以下前提：

1. **代码与环境对齐**：两台设备拉取同一版本代码，安装一致的 PyTorch / CUDA 依赖。
2. **数据路径与内容对齐**：两台设备均可通过 `$XL_DATA_DIR`（或 `--train-source` / `--val-source`）访问内容一致的训练与验证数据（各自拷贝一份到相同路径，或挂载同一份共享存储）。各 rank 内部按 `rank_seed` 抽取不同训练样本，无需手工切分数据文件。
在运行双机预训练前，先在设备 A 准备语料和分词器。以下玩具语料命令会向指定数据目录写文件；请使用新的目录。先将 `/path/to/data` 和 `/path/to/out` 改成实际绝对路径：

```bash
export XL_DATA_DIR=/path/to/data XL_OUT_DIR=/path/to/out
python examples/generate_toy_dataset.py
python scripts/01_train_tokenizer.py --corpus-path "$XL_DATA_DIR/corpus.txt" --vocab-size 2048
```

将生成的 `corpus.txt` 和 `$XL_OUT_DIR/tokenizer/` 完整复制到设备 B 对应位置，或让两台设备通过共享目录访问相同文件；至少要有 `tokenizer.json` 与 `token_bytes.pt`。核对两端文件一致后，再启动以下命令。上面的玩具数据记录采用分阶段启动；后续双机阶段也需要两端读取一致的输入数据和前一阶段检查点。

3. **网络互通与网卡显式指定**：两台设备需互通 `MASTER_ADDR:MASTER_PORT`（如 TCP `29500`）及 NCCL/Gloo 动态通信端口。如果机器有多张网卡（例如管理网口、Docker 虚拟网桥、VPN 与高速内网并存），必须在两侧显式指定同一张可直连互通的网卡，否则分布式初始化会超时挂起：

```bash
# 两侧按实际互通网卡名设置（例如 eth0 / enp1s0 / tailscale0）
export NCCL_SOCKET_IFNAME=<互通网卡名>
export GLOO_SOCKET_IFNAME=<互通网卡名>
```

4. **两端分别拉起与产物收集**：两台设备传入相同的 `--steps`、`--scale`、模型配置与 `XL_RUN_ID` 分别启动（Linux CUDA 下 `--backend auto` 会选用 `nccl`）。**只有 `node-rank=0`（全局 rank 0）会执行验证评估、保存 checkpoint、写指标 JSON 与绘图**，从节点只负责前向/反向与梯度 `all_reduce`，实验结束后直接从主节点的 `$XL_OUT_DIR` 收集产物即可：

```bash
# 设备 A（主节点，node-rank=0，负责落盘指标与 checkpoint）
XL_RUN_ID=dual-run-01 XL_DATA_DIR=/path/to/data XL_OUT_DIR=/path/to/out \
torchrun --nproc_per_node=1 --nnodes=2 --node-rank=0 \
  --master_addr=<NODE0_REACHABLE_IP> --master_port=29500 \
  scripts/03_pretrain.py --steps 250 --device cuda --amp bf16

# 设备 B（从节点，node-rank=1，与主节点配置和步数保持一致）
XL_RUN_ID=dual-run-01 XL_DATA_DIR=/path/to/data XL_OUT_DIR=/path/to/out \
torchrun --nproc_per_node=1 --nnodes=2 --node-rank=1 \
  --master_addr=<NODE0_REACHABLE_IP> --master_port=29500 \
  scripts/03_pretrain.py --steps 250 --device cuda --amp bf16
```

复现上面的玩具数据运行时，将预训练步数改为 `60`。接着依次运行 `04_midtrain.py`（40 步）、`05_sft.py`（80 步）和 `06_rl_grpo.py`（20 步）。每一阶段两台都使用以下命令，分别设置本机的 `NODE_RANK` 为 `0` 或 `1`，并设置两端可访问的 `MASTER_ADDR`；`STAGE` 和 `STEPS` 在两端保持一致。两端应分别设置本机的 `XL_DATA_DIR`、`XL_OUT_DIR` 和相同的 `XL_RUN_ID`：

```bash
STAGE=scripts/04_midtrain.py STEPS=40  # 后续依次改成 05_sft.py / 80、06_rl_grpo.py / 20
torchrun --nproc_per_node=1 --nnodes=2 --node-rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  "$STAGE" --steps "$STEPS" --device cuda --amp bf16
```

进入下一阶段前，将主节点 `$XL_OUT_DIR/checkpoints/` 内的 `pretrain.pt`、`midtrain.pt`、`sft.pt` 中对应的上一步产物复制到另一端相同的相对位置，两端用 `sha256sum` 核对一致；也可使用共享存储。两端的 `sft.jsonl` 和 `rl.jsonl` 同样要一致。GRPO 完成后，在主节点执行 `python scripts/07_generate.py --device cuda` 和 `python scripts/08_eval_report.py --device cuda`；继续使用同一组 `XL_*` 路径。运行期间应由任务管理器设置总时限，避免通信故障导致两端一直等待。

- 若要验证「同全局 batch 下的多机加速比」，双机每卡 `--batch-size`（或 `--scale`）应设为单卡基线的一半（保证总有效 batch 一致）；若维持单卡每卡 batch 不变，则每步全局 batch 翻倍。

DDP 语义要点：

- 各 rank 训练 batch 由 rank 偏移的种子抽取（`batch_generator` / `rank_seed`），数据不重复
- 评估、样例生成、保存 checkpoint、画图只在 rank0 执行；指标（如 06 的 mean_reward）跨 rank 平均
- checkpoint 保存时剥离 DDP `module.` 前缀，`load_checkpoint` 兼容带前缀的旧权重并支持 `map_location`
- 06 的采样阶段对 unwrap 后的策略在 no_grad 下执行，只有策略前向反向走 DDP；逐 group 一次 forward 配一次 backward（与单进程求和后一次 backward 数学等价），各 rank 有效 group 数不同时按全局最大值用零梯度 backward 补齐调用次数，避免 all_reduce 配对死锁
- KV cache、RoPE 缓存、采样 Generator 均与输入张量同设备；Muon 的 Newton-Schulz 在 fp32 累加格上计算，AMP 下无需改动

已知环境限制（本机实测 torch 2.14.0+cu126 Windows wheel）：

- 该 wheel 的 TCPStore 被编译成强制 libuv 但未编入 libuv，`torchrun` 的 rendezvous 直接报
  `use_libuv was requested but PyTorch was built without libuv support`，且 `USE_LIBUV=0`
  无效（上游已知问题）。此时用 `LLM_INIT_METHOD` 指向 file:// 的 FileStore，手动起两个进程
  即可验证同样的 2-rank 语义：

```bash
# 两个进程共用同一 FileStore 路径（Windows 示例）
LLM_INIT_METHOD=file:///C:/temp/xk_store WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 python scripts/02_model.py --device cpu &
LLM_INIT_METHOD=file:///C:/temp/xk_store WORLD_SIZE=2 RANK=1 LOCAL_RANK=1 python scripts/02_model.py --device cpu &
```

- 同一 wheel 上 gloo 后端对 CUDA 张量的 backward 会段错误（最小复现：DDP 包装 Linear 后
  `loss.backward()`），因此本机 2-rank DDP 实测在 CPU gloo 上完成（权重广播、梯度一致、
  checkpoint 无前缀三项自检 + 03/04/05/06 四阶段 2-rank 训练全部通过）；CUDA 2-rank 需要
  gloo/CUDA 传输完好的 wheel 或 Linux NCCL 环境，代码路径相同（backend auto 自动选 nccl）

## 分词器域偏差评测与双机扩展实验

除主链路外，本目录还包含两个配套实验，均遵守同一运行契约：

### 分词器多域压缩率评测（`eval_tokenizer_domains.py`）

对多个分词器在固定域切片（`fixtures/`：Python 代码、LaTeX 数学、中文技术文本、中英混合代码、数字算术）上统一评估 bytes/token、roundtrip 与 token 膨胀率，并审计 `token_bytes.pt` 公式。真实语料域经 `--extra-domain NAME=PATH` 注入，脚本内不写任何本地路径：

```bash
python scripts/eval_tokenizer_domains.py \
    --tokenizer-path "$XL_OUT_DIR/tokenizer/tokenizer.json" --tokenizer-name current \
    --extra-domain climbmix_val=/path/to/shard_06542.parquet \
    --audit-source /path/to/shard_06542.parquet --baseline-domain climbmix_val \
    --out "$XL_OUT_DIR/tokenizer/domain_compression_eval.json"
```

脱敏结果副本见 `evidence/tokenizer-domain-compression-eval.md` / `.json`。

### 数据量级扩展实验（`scripts/scaling/`）

按同一模型架构、同一留出验证集和递增数据量级比较训练结果。现存记录中的有效训练均为单卡；双机预检时第二节点不可达，随后转为单卡运行，因此没有可供比较的双卡 DDP 结果。强依赖特定主机与网络拓扑的多机编排脚本未纳入本仓，本仓保留通用的汇总与基准工具，用法与产物见 [`scripts/scaling/README.md`](scripts/scaling/README.md)。汇总结果副本见 `evidence/scaling-summary.json` / `evidence/scaling-comparison-table.md`。

## 硬件需求档

| 档位 | 入口 | 硬件 | 时长量级 |
|---|---|---|---|
| 冒烟 | `configs/smoke.json` | 纯 CPU（4 线程即可） | 全链路 8 阶段约 2 到 3 分钟 |
| GPU 冒烟 | `configs/gpu-smoke.json` | 单卡 CUDA + BF16 | 用玩具数据验证流程；运行时间随环境变化 |
| 完整 | `configs/full.json` | 单卡 GPU + bf16；所需显存随模型、批次与数据变化，最低要求未验证 | 视步数与数据量，分钟到小时级 |
| 双卡 DDP | `NPROC=2 bash run.sh --config configs/gpu-smoke.json` | 单机双卡（Linux NCCL 或可用的 gloo/CUDA） | 速度取决于通信与工作负载；该命令及双卡对照尚未验证 |

跨节点多机（`nnodes=2`）可通过上节 `torchrun` 手动拉起；`run.sh` 专注单机单卡 / 单机多卡一键复现，绑定特定内网拓扑的跨机编排脚本未纳入本仓。

## 目录结构

```
llm-lifecycle-fullchain/
├── README.md                    本文
├── requirements.txt
├── run.sh                       一键复现入口（运行契约版，profile 见 configs/）
├── configs/
│   ├── smoke.json               冒烟档：CPU 最小步数
│   ├── gpu-smoke.json           GPU 冒烟档：单卡 CUDA + BF16
│   └── full.json                完整档：较大的词表与训练步数
├── examples/
│   └── generate_toy_dataset.py  独立离线冒烟测试语料合成器（写 $XL_DATA_DIR 或 .work/data）
├── fixtures/                    分词器域偏差评测的固定域切片与生成器
├── evidence/                     收敛后的报告与证据副本（原始运行产物留在 $XL_OUT_DIR）
├── scripts/
│   ├── dataset.py               统一流式数据集加载器 (对标 nanochat/dataset.py)
│   ├── llm_common.py            共享模块：模型、Muon、KV cache、数据与生成工具
│   ├── 01_train_tokenizer.py    GPT-4 风格正则预分词的 BPE 训练/导出/校验
│   ├── 02_model.py              模型自检（QK norm / RoPE / KV cache / softcap / GQA）
│   ├── 03_pretrain.py           Muon+AdamW 预训练 + AdamW 基线对照
│   ├── 04_midtrain.py           对话格式 midtrain
│   ├── 05_sft.py                指令微调（prompt 掩码）
│   ├── 06_rl_grpo.py            GRPO（可验证奖励 + KL 锚定）
│   ├── 07_generate.py           贪心/温度采样 + KV cache 提速对比
│   ├── 08_eval_report.py        四阶段评估矩阵 + report.md
│   ├── eval_tokenizer_domains.py  分词器多域压缩率评测
│   ├── generate_md_report.py       markdown 报告渲染
│   ├── plot_pretrain_figures.py    预训练章节配图
│   ├── plot_tokenizer_figures.py   分词器章节配图
│   └── scaling/                 单机/双卡数据量级扩展实验（见其 README）
└── （运行产物写至 $XL_OUT_DIR 或 .work/，不入库）
```

`$XL_OUT_DIR` 下的产物布局：`tokenizer/tokenizer.json` + `tokenizer_info.json` + `token_bytes.pt`、`checkpoints/{pretrain,pretrain_adamw,midtrain,sft,rl}.pt`、`{pretrain,midtrain,sft,rl}_metrics.json`、`pretrain_loss.png` / `midtrain_loss.png` / `sft_loss.png` / `rl_reward.png`、`samples.md` / `samples.json`、`report.md` / `summary.json`、`run_manifest.json`。

## 各环节实现要点

### 01 分词器（Byte-Level BPE + GPT-4 风格预分词）

旧版 `token_bytes.pt` 对 ByteLevel 词表字节数计算有误，且 BPB 分子曾包含特殊 token 的损失。当前代码加载旧缓存时会按词表在内存中校正；旧评估窗口还遗漏每个完整窗口的末位预测；旧 BPB 和 PPL 结果需要使用原模型和原验证集重新评估，不能直接按固定比例换算。

工程原理与流水线拆解见 [分词器训练与词表规模权衡](../../docs/zh/inference/分词器训练与词表规模权衡.md)；不靠模型的四维评测、压缩率实测与基于 `token_bytes.pt` 的 BPB 口径校准见 [分词器效果评测与跨词表 BPB 计算](../../docs/zh/inference/分词器效果评测与跨词表BPB计算.md)。

样本语料由模板确定性生成：首都事实、水果颜色、人物职业、句法组合、英文数词算术、阿拉伯数字算式（108 组）、身份/闲聊与全部事实问答的对话记录，共 37,824 字节、170 个 EOS 分隔记录；SFT 指令对 83 条（68 条事实问答 + 15 条算术，后者给 RL 提供起点能力）；RL 可验证任务 155 条（个位数加减法，108 条算式进过语料 / 47 条完全留出，按算式组划分）。

预分词用 GPT-4 风格正则（`tokenizers` 库 `Split` isolated + ByteLevel），与 nanochat 的 tiktoken 路线一致：先按词/数字/标点切分，再逐块 byte-level BPE。特殊 token 为 `<|endoftext|>`(id=0) 与 `<|pad|>`(id=1)。微型语料下 merge 很快饱和，词表停在 775（目标 2048），属预期现象。

### 02 模型架构（对齐 nanochat gpt.py）

| 组件 | 实现 |
|---|---|
| 归一化 | 无参数 RMSNorm（`F.rms_norm`，无 bias、无均值中心化、无可学习权重） |
| 位置编码 | RoPE（base=100000），先施加旋转、后做 QK norm（nanochat 顺序） |
| 前馈 | relu²：`down(relu(fc(x))²)`，隐层 768；SwiGLU 作为可选对照（`--mlp-type`） |
| 注意力 | 因果自注意力，`F.scaled_dot_product_attention(is_causal=True)`，GQA 代码路径保留 |
| 结构 | 预归一化残差块 ×4，全模型无 bias，词嵌入与 lm_head 默认不共享（untied） |
| 其它 | 嵌入后接一次 norm；logits softcap=15 软截断；KV cache 推理引擎 |

默认配置 vocab=2048 口径 2.56M 参数；实际词表 775 时 **2,067,072（2.07M）**，其中 tok_emb 与 lm_head 各 599,550（untied），blocks 矩阵 868,352。`python scripts/02_model.py` 做形状、QK norm 缩放不变性、RoPE 相对位置性、KV cache 与 naive 前向一致性、softcap、双 MLP 路径、GQA、tie/untie 自检。

### 03 预训练（Muon + AdamW，附基线）

按 EOS 记录切分 95/5 训练/验证集，随机裁剪 128 token 窗口；优化器分组对齐 nanochat：blocks 内矩阵用 **Muon**（动量 0.95 + Newton-Schulz 5 步正交化，matrix_lr 0.02），词嵌入与 lm_head 用 AdamW（lr 0.3 / 0.004），warmup + cosine 衰减，梯度裁剪 1.0，batch 16，250 步。同 steps 同种子再跑一遍纯 AdamW 作基线，两者 val loss 与耗时都写进 `pretrain_metrics.json`。

深入原理、扩展律配比与短语料过拟合诊断见 [大模型预训练原理与扩展律](../../docs/zh/inference/大模型预训练原理与扩展律.md)。

### 04 Midtrain（对话格式适配）

加载预训练权重，只在「User:/Assistant:」对话记录上以 0.2 倍 lr 续训 120 步，全序列 loss。对应 nanochat 在 base_train 与 chat_sft 之间的 midtraining：先让模型适应对话格式，再做指令微调。

### 05 指令微调（SFT）

加载 midtrain 权重，样本拼成 `User: {问题}\nAssistant:` + ` {回答}` + eos，**loss 只在回答 token 上计算**（prompt 位置 `ignore_index=-100`）；lr 缩放 0.2、batch 16、250 步。训练后在留出指令对上贪心生成，按答案前缀命中率评估。

### 06 RL（GRPO，可验证奖励）

以 sft.pt 为策略、同权重冻结副本为参考模型，在个位数加减法上做 GRPO：

- 每个 prompt 采样一组 8 个回答（T=1.0），reward 为可验证的答案匹配
- advantage = 组内 `(r - mean) / std`；组内无差异时跳过（无梯度）
- loss = `-advantage · logprob(回答 token) + β·KL(π ‖ ref)`，β=0.04
- reward 分档：精确匹配 1.0；答案后仅跟标点 0.6；首字符为数字 0.2；其余 0
  （后两档是冷启动整形：全 0 奖励时组内无差异、GRPO 无梯度）
- 训练算式（108 条，进过预训练语料）与留出算式（47 条，完全未见）分别评估，观察 RL 的记忆与泛化边界；同时用聊天问答命中率监测灾难性遗忘

### 07 生成与 KV cache

贪心解码（argmax）与温度采样（`softmax(logits/T)` + top_k/top_p 截断 + multinomial）均已实现，eos 即停；每个 prompt 固定采样种子保证可复现。默认同时用 midtrain / pretrain 模型对照同一批 prompt，并附 T=0.8 / T=2.5 温度效应对照。KV cache 与 naive 全前向解码同参数对比 tokens/s。

### 08 评估

对 pretrain / midtrain / sft / rl 四个 checkpoint 跑同一套评估：聊天问答命中率（SFT 留出集）+ 算术 exact-match（训练/留出），汇总写 `$XL_OUT_DIR/report.md` 与 `summary.json`。

关于预训练 Base 模型与 Post-train 指令期评测边界、连续 `val_bpb` 指标及续写式多选基准的设计原理，见 [预训练阶段评测与能力基准](../../docs/zh/inference/预训练阶段评测与能力基准.md)。

## 实验结果（实测）

> 以下数值来自默认配置的一次完整 CPU 运行（`run_manifest.json` 随运行生成于 `$XL_OUT_DIR`）；收敛后的报告与对比表在 `evidence/`。

| 环节 | 指标 | 数值 |
|---|---|---|
| 01 分词器 | 词表大小 / eos_id / pad_id | 775（目标 2048，微型语料下 merge 饱和）/ 0 / 1 |
| 01 分词器 | 语料压缩率 | 37,824 bytes → 9,684 tokens（3.906 bytes/token） |
| 01 分词器 | encode→decode roundtrip | 0/50 失败 |
| 02 模型 | 参数量（vocab=775） | 2,067,072（2.07M），02 全部自检通过 |
| 03 预训练 | Muon train loss（前 5% → 后 5% 步均值） | 4.4863 → 0.0924（↓ 97.9%） |
| 03 预训练 | Muon best val loss | 1.2501 @ step 49 |
| 03 预训练 | AdamW best val loss（同 steps 同种子） | 1.2775 @ step 149 |
| 03 预训练 | final val loss（Muon / AdamW） | 2.3770 / 1.3380（小语料过拟合，见下） |
| 04 Midtrain | train loss 下降 / final val loss | 0.4366 → 0.0382（↓ 91.3%）/ 0.9505 |
| 05 SFT | sft loss 下降 | 9.7020 → 0.0005（↓ 100.0%） |
| 05 SFT | 答案命中率 | 训练集 100%（71/71），留出集 58%（7/12） |
| 06 RL | 算术 exact-match（训练集，SFT → RL） | 17% → 23% |
| 06 RL | 算术 exact-match（留出集，SFT → RL） | 13% → 9%（未见算式未学会，见边界） |
| 06 RL | 聊天问答命中率（回归检查） | 60% → 60%（无回归） |
| 07 生成 | KV cache vs naive 解码速度 | batch=1：1.4x；batch=8：4.9x（生成长度 96 token） |

**Muon vs AdamW 的读法**：同 steps 同种子下，Muon 的 best val 更低且出现早 3 倍
（step 49 vs 149），优化更快；但 250 步后 Muon 的 final val（2.3770）高于 AdamW
（1.3380）：9K token 的微型语料上 Muon 更快地把训练集背下来（train loss 0.09 vs 0.83），
过拟合出现得更早。这是数据规模的局限，不是优化器缺陷：nanochat 在数十亿 token 上
Muon 的优势才能完整发挥。

生成示例（完整见 `evidence/generation-samples-toy-synthetic.md`）：

| prompt | RL 模型贪心 | SFT 模型贪心（RL 前） |
|---|---|---|
| `User: What is the capital of Japan?` + `Assistant:` | `Tokyo` | `Tokyo` |
| `User: What color is a banana?` + `Assistant:` | `The banana is yellow.` | `The banana is yellow.` |
| `User: What does Carol do?` + `Assistant:` | `Carol is a engineer.` | `Carol is a engineer.` |
| `User: What is 7 plus 5?` + `Assistant:` | `7` | `7` |

预训练 / midtrain 模型对同一批 prompt 只会续写语料片段且不遵循对话格式
（`evidence/generation-samples-toy-synthetic.md` 有完整对照），SFT 后才按问题实体作答：这是「指令微调改变
行为分布」的直接证据。RL 后模型对算术题稳定输出数字形态（如 `7`），但多数还算不对
精确结果，符合 2M 参数玩具模型的预期。温度效应对照显示 T=0.8 输出稳定连贯，
T=2.5 出现多样性同时开始崩坏。

四阶段评估矩阵见 `evidence/fullchain-eval-report-toy-synthetic.md`。

## 复现性和确定性

- 所有随机源均固定种子（语料生成 1234、训练 1234、验证评估 1235、生成采样 1234+序号），同机重跑结果逐位一致
- DDP 下各 rank 种子按 rank 偏移（`rank_seed`），单卡与多卡跑的是同一段代码；多卡因梯度跨 rank 平均，loss 曲线与单卡不完全逐位一致属预期
- `02` 内置七项结构自检（形状 / QK norm 缩放不变性 / RoPE 相对位置性 / KV cache 与 naive 逐 token 一致含 batch=3 / softcap / relu² 与 SwiGLU 双路径 / tie-untie），另加 AMP 前向、Muon 设备更新与（torchrun 下）DDP 权重广播、梯度一致、checkpoint 无 module. 前缀三项分布式自检；`03/04/05` 内置 loss 下降断言，未达标以非零码退出
- 每一步的完整配置写入对应 `*_metrics.json` 的 `config` 字段（含 dist 子块：device / world_size / backend / amp / grad_accum / scale），以及 `run_id`

## 边界与局限

- 词表只有 775：微型语料下 BPE merge 很快饱和，属预期现象；真实场景应增大语料
- 模型仅 2.07M 参数、上下文 128 token，只能回答训练分布内的简单事实问答，不具备泛化语言能力
- 语料为英文模板合成数据，中文及开放域任务不在本实验范围内
- Muon 在 9K token 语料上 250 步即过拟合（best val @ step 49），AdamW 基线衰减更慢；这是数据规模的局限
- GRPO 的奖励含标点宽容与数字形态整形，是玩具规模的简化；实测 RL 主要收益在训练集内算式（17%→23%），47 条留出算式不升反降（13%→9%）：模型没有学会加法运算，只是背下了训练算式，泛化边界如实呈现
- RL 存在对话格式回归风险，本实验用聊天问答命中率量化监测（本次 60%→60% 无回归）
- Midtrain 的对话记录同时是预训练语料的一部分，小数据续训后 val loss 上扬（过拟合），该阶段只为格式适配，硬性门槛只查 train loss 下降
- KV cache 的收益随 batch 与生成长度增长：batch=8 时 4.9x，batch=1 仅 1.4x；模型再小或生成再短，cache 的每步固定开销会吃掉收益
