# SimCSE 对比学习：dropout 正例与 NLI 难负例
[English](README.en.md)

本目录提供 SimCSE 两代训练目标的复现脚本（依赖 PyTorch 和 Transformers）：无监督版用同一句子的两次 dropout 前向构造正例，监督版用 NLI 蕴含对加难负例。配套评测脚本输出 STS 语义相似度（Spearman/Pearson）与检索 Recall@k，可用于横向对比不同基座、温度与池化方式。

主讲解见 [SimCSE 对比学习](../../docs/zh/retrieval/embedding/SimCSE对比学习.md)，其中包含 dropout 正例的构造逻辑、InfoNCE 与温度的作用机制，以及本实验的设计取舍；表示塌缩与对齐均匀性的几何背景见[对比学习与表征几何](../../docs/zh/retrieval/embedding/对比学习与表征几何.md)。

[scripts](scripts/) · [assets](assets/)

## 测什么，不测什么

测：

- 无监督 SimCSE 的 dropout 正例机制能否在通用语料上把各向异性的句向量拉开（以 STS 分数变化衡量）
- 监督版（NLI 三元组）与无监督版的差距，以及 hard negative 的作用
- 温度、批大小、池化方式（CLS 与 mean）对收敛与指标的影响

不测：

- 不测指令式大嵌入模型（E5、BGE-M3、Qwen3-Embedding 等），它们靠指令与大规模对比预训练，与 SimCSE 的训练设定不同源，对比结论不可直接套用
- 不测检索端到端效果（ANN 索引、切块策略），检索指标只作为表征质量的辅助信号
- 不测跨语言与多模态设定

## 对象与版本

| 项 | 取值 | 说明 |
|---|---|---|
| 无监督正例来源 | dropout 采样 | 同一句子两次前向，`train` 模式下 dropout 生效 |
| 监督数据形态 | NLI 蕴含/中性/矛盾 | anchor 与 positive 为蕴含对，negative 为矛盾句 |
| 相似度评测 | STS-B | Spearman 为主指标，Pearson 作辅助 |
| 检索评测 | 自建候选池 | Recall@10，只用于观察表征可分性 |
| 依赖 | PyTorch、Transformers | 不依赖 sentence-transformers，池化在本目录实现 |

复现论文数字请固定基座为 `bert-base-uncased`：SimCSE 原文（Gao et al., EMNLP 2021，[arXiv:2104.08821](https://arxiv.org/abs/2104.08821)）报告 BERT-base 无监督版 STS-B 76.85、七个 STS 任务平均 76.25，监督版 STS-B 84.25、平均 81.57。本仓库脚本未在任何基座上完整重跑论文全部设定，下面的命令是复现路径而非已验证的一键流程。

## 环境

根 `requirements.txt` 不放运行时依赖。以下 Bash 命令以 Linux 为例，在数据准备之前创建独立 Python 环境；需安装支持本机 GPU 的 CUDA PyTorch。完整训练的显存取决于模型和批次，尤其监督版默认批次 512，不保证适用于所有 GPU：

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# 按 https://pytorch.org/get-started/locally/ 安装与系统、架构和 GPU 匹配的 PyTorch
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), '训练需要可用的 CUDA GPU'"
```

本目录将直接依赖固定为 `transformers==5.16.1`、`datasets==5.2.0`、`numpy==2.5.4`；PyTorch 仍须按设备单独选择 CUDA 构建。传递依赖没有完整锁定，因此这不是跨时间、跨平台的完整 lockfile。

2026-11-13 在一台 NVIDIA GB10 的现有环境中，以 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.22.2、Datasets 5.2.0、NumPy 2.5.4、tokenizers 0.23.3 完成最小 GPU 冒烟：使用 tiny-random-bert，分别以 4 条合成句和 4 条合成 NLI 三元组运行无监督、监督训练，各 1 步；再用保存的模型对 4 对合成句执行 STS 评测。进程退出码均为 0。该检查使用已有环境和模型缓存，并非从零安装；合成数据不用于评价模型效果，也不代表真实语料上的完整训练或论文级复现，此依赖组合不代表历史实验环境。该现有环境的 `python -m pip check` 未通过，报告 `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`；小规模 GPU 冒烟成功不代表依赖状态完整。

不需要 sentence-transformers；句向量由 `scripts/simcse_lib.py` 的池化代码产生。

2026-11-15 已在 NVIDIA GB10 的全新隔离环境中按上述顺序安装 PyTorch 与本目录依赖：Python 3.12.3、PyTorch 2.15.0+cu132、CUDA 13.2、Transformers 5.16.1、Datasets 5.2.0、NumPy 2.5.4。`pip check`、依赖导入、CUDA 可用性和 BF16 检查均通过。该次只验证环境安装，没有下载模型或数据，也没有运行训练；细节见[干净环境安装记录](evidence/clean-install-2026-11-15.json)。

2026-11-15 又用全新隔离环境完成了真实数据 GPU 小样本链路：清除本次子进程继承的代理变量后，PyTorch CUDA 与本目录固定依赖安装成功，`pip check` 通过；从 Hugging Face 镜像下载锁定版本的 `bert-base-uncased`、Wikipedia、SNLI 和 GLUE STS-B，生成 32 条无监督句子、32 个 SNLI 三元组及 100 对 STS-B 验证样本；无监督和监督版本各训练 4 步，并在 CUDA 上完成保存后评测。基座 / 无监督 / 监督的 STS-B Spearman 分别为 -0.0155 / 0.0387 / 0.3337。样本与步数很小，这些数字只确认数据、训练、保存和评测流程跑通，不证明模型质量或优于基线。部分数据文件读取有超时重试，但本次最终完整结束。详见[真实数据 GPU 小样本记录](evidence/real-data-smoke-2026-11-15.json)。早先一次 Hub 请求被拒绝的记录仍保留在[失败记录](evidence/clean-env-real-data-blocked-2026-11-15.json)，反映的是当次网络状态。

以上验证固定了直接依赖，但没有固定全部传递依赖。示例数据和模型首次获取需要网络；确认数据集与模型的许可后再使用。`smoke_test.py` 固定使用 CPU，只检查链路，不代表 GPU 已验证。

## 数据准备

训练数据与评测集都不随仓库分发。以下命令从本实验目录执行，在仓库外创建一个全新的临时运行目录（以下示例使用 Linux Bash），避免将数据或权重写入代码仓：

```bash
export RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/simcse.XXXXXXXX")"
export DATA="$RUN_DIR/data"
export OUT="$RUN_DIR/output"
mkdir "$DATA" "$OUT"
printf '运行目录：%s\n' "$RUN_DIR"
```

无监督训练用维基百科等通用语料的单句集合，每行一个 JSON 对象：

```json
{"sentence": "A storm dropped five inches of rain on the valley."}
```

以下以 [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 的 `20231101.en` 为输入示例，与旧版语料或论文预处理不等同；流式读取需要联网，输出句子数量也取决于筛选规则。用 Hugging Face datasets 生成：

```bash
python - <<'PY'
import json
import os
from itertools import islice
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
with open(os.path.join(os.environ["DATA"], "unsup_wiki.jsonl"), "x", encoding="utf-8") as f:
    for row in islice(ds, 100000):
        for sent in row["text"].split(". "):
            sent = sent.strip()
            if 20 < len(sent) < 200:
                f.write(json.dumps({"sentence": sent}, ensure_ascii=False) + "\n")
PY
```

监督训练用 SNLI 或 MNLI 的三元组，每行一个 JSON 对象：

```json
{"anchor": "A man is running.", "positive": "Someone jogs.", "negative": "A man sleeps."}
```

```bash
python - <<'PY'
import json
import os
from collections import defaultdict
from datasets import load_dataset

ds = load_dataset("stanfordnlp/snli", split="train")
by_premise = defaultdict(lambda: {0: set(), 2: set()})
for row in ds:
    if row["label"] in (0, 2) and row["premise"] and row["hypothesis"]:
        by_premise[row["premise"]][row["label"]].add(row["hypothesis"])
with open(os.path.join(os.environ["DATA"], "snli_triplets.jsonl"), "x", encoding="utf-8") as f:
    for premise, groups in by_premise.items():
        negatives = sorted(groups[2] - {premise})
        if not negatives:
            continue
        for positive in sorted(groups[0] - {premise}):
            if positive != negatives[0]:
                f.write(json.dumps({"anchor": premise, "positive": positive,
                                    "negative": negatives[0]}, ensure_ascii=False) + "\n")
PY
```

这个示例按相同 premise 配对蕴含和矛盾假设，并去重；没有两类标注的 premise 会被跳过。它是便于理解的数据准备方式，不等同于论文原始预处理。

相似度评测用 STS-B，每行一个 JSON 对象：

```json
{"sentence1": "A man is playing a guitar.", "sentence2": "A man plays guitar.", "score": 3.8}
```

```bash
python - <<'PY'
import json
import os
from datasets import load_dataset

ds = load_dataset("nyu-mll/glue", "stsb", split="validation")
with open(os.path.join(os.environ["DATA"], "stsb_val.jsonl"), "x", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps({
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "score": row["label"],
        }, ensure_ascii=False) + "\n")
PY
```

## 复现命令

以下命令在 `experiments/contrastive-learning-simcse/` 目录下执行。`OUT` 指向仓库外的新目录，模型与训练指标都写在里面。生成数据采用独占创建；若中断后留下不完整文件，请检查并换新的运行目录重试。训练与评测脚本会覆盖已有的同名模型或结果；重新运行前请按上面的命令创建新的运行目录。

无监督训练（SimCSE 论文的超参：batch 64，学习率 3e-5，温度 0.05，1 epoch）：

```bash
python scripts/train_unsup_simcse.py --model bert-base-uncased --data "$DATA/unsup_wiki.jsonl" --out "$OUT/unsup" --batch-size 64 --lr 3e-5 --temperature 0.05 --epochs 1 --device cuda
```

监督训练（论文超参：batch 512，学习率 3e-5，温度 0.05，3 epoch）：

```bash
python scripts/train_sup_simcse.py --model bert-base-uncased --data "$DATA/snli_triplets.jsonl" --out "$OUT/sup" --batch-size 512 --lr 3e-5 --temperature 0.05 --epochs 3 --device cuda
```

同时评测基座与两个训练产物，口径完全一致才可比：

```bash
python scripts/evaluate_sts.py --model bert-base-uncased --sts-files "$DATA/stsb_val.jsonl" --json-out "$OUT/base-eval.json"
python scripts/evaluate_sts.py --model "$OUT/unsup" --sts-files "$DATA/stsb_val.jsonl" --json-out "$OUT/unsup-eval.json"
python scripts/evaluate_sts.py --model "$OUT/sup" --sts-files "$DATA/stsb_val.jsonl" --json-out "$OUT/sup-eval.json"
```

需要检索辅助指标时加 `--retrieval-file`，格式见 `scripts/evaluate_sts.py` 的 docstring。

### 冒烟测试

用随机模板合成的中文句子检查训练、保存、加载、评测链路；默认模型未缓存时会从模型源下载。它验证脚本可用，不验证效果：

```bash
python scripts/smoke_test.py --model hf-internal-testing/tiny-random-bert
```

换本机已有的小模型也可以：`python scripts/smoke_test.py --model /path/to/local/model`。

### 真实数据小样本运行

2026-11-14 在 Linux ARM64、NVIDIA GB10 上，使用从现有环境复制并调整的独立环境，Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1、Datasets 5.2.0，`pip check` 通过。以 `bert-base-uncased` 的 revision `86b5e0934494bd15c9632b12f734a8a67f723594` 为基座，使用 32 条 Wikipedia 句子和 32 条 SNLI 三元组，分别在 CUDA 上训练 1 轮、batch 8、各 4 步；随后对基座和两个训练产物使用相同的 100 条 STS-B 验证样本进行评测。

这 100 条样本上的 Spearman 分别为基座 0.0013、无监督模型 0.0387、监督模型 0.3337。样本量和训练步数都很小，数据集版本未固定，这只是训练、保存和评测链路的功能检查，不能作为模型质量结论，也不能视为完整复现论文或从零安装验证。

该次运行的模型 revision、环境版本、训练配置与评测指标已整理在[运行摘要](evidence/real-data-smoke-2026-11-14.json)。原始训练指标与评测日志留在内部运行档案；摘要没有补造未记录的数据集 revision 或文件哈希。

### 合成数据运行记录

已有记录使用 `hf-internal-testing/tiny-random-bert` 和 500 条模板句子进行无监督训练，配置为 3 轮、96 步。训练记录的 `elapsed_seconds` 为 1.70 秒；另一份摘要标注硬件为 NVIDIA GB10，记录了 6.059 秒的 `train_duration_sec`，但未说明这两个字段的计时边界，不能直接比较。[训练记录](evidence/synthetic-training-metrics.json) · [指标摘要](evidence/synthetic-eval-summary.json)

摘要记录 500 对句子的 Spearman 为 0.8215、Pearson 为 0.6975。已保存的句对中，262 对文本完全相同且标为 5 分，其余 238 对不同文本均标为小于 2 分。这组模板数据用于检查流程，不能证明模型在真实语义相似度任务上的效果；原始句对与模型权重不随仓库分发，公开摘要不能独立重建这些指标。运行日期和完整环境尚未从现存摘要核实，耗时与内存数值不作为性能基准。

## 输出说明

- `--out` 目录：主干权重、分词器、`pooler.pt`、`simcse_config.json`、`train_metrics.json`（含损失曲线与末段均值）
- `--json-out`：评测指标 JSON，含每个评测文件的 Spearman、Pearson、样本数
- 运行产物目录（`results/`、`models/`）由公开 `.gitignore` 覆盖，不进版本库；要留档的指标请复制成报告

## 已知限制

- 真实数据只完成上述小样本功能检查；干净环境安装已单独验证，但尚未在该环境中训练。完整数据训练及论文结果尚未复现，论文数字引自 SimCSE 原文，仅作参照
- MLM 辅助损失是简化实现：掩码位置直接替换为 `[MASK]`，没有 80/10/10 的替换策略，默认关闭
- 无监督版没有实现动量编码器与队列（MoCo 风格），大批量完全依赖显存；显存不足时减小 `--batch-size` 并相应调低学习率
- 检索评测只在单一候选池上算 Recall@10，池子大小直接影响绝对值，跨实验比较必须固定候选池
- 温度与池化方式对结果影响很大，更换基座时应重跑同一组超参，不要跨配置直接比分
- Windows 上可用 CPU 跑冒烟测试，完整复现需要 Linux CUDA 环境

