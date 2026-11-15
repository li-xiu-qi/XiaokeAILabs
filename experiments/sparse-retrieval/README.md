# 稀疏检索模型微调实验（SPLADE Sparse Fine-tuning）

[English](README.en.md)

> **技术专题**：[学习型稀疏表示与 SPLADE 原理](../../docs/zh/retrieval/sparse/学习型稀疏检索与SPLADE.md)

---

## 一、 实验卡（立项三要素）

| 要素 | 约定规范 |
|---|---|
| **实验目的** | 跑通基于预训练语言模型（BERT/MiniLM/BGE）的全词表稀疏表示微调管线，量化词项扩展能力（Term Expansion）与倒排稀疏度约束（FLOPS Regularization）的动态平衡曲线，验证其在保留倒排索引效率的同时超越传统 BM25 的检索精度。 |
| **达成门槛** | 1. 成功构建包含 MLM Head 投影与序列最大值池化（Max-Pooling）的稀疏编码器；<br>2. 联合训练排序对比损失与 FLOPS 正则化损失，在验证集上相较于标准 BM25 实现 MRR@10 显著提升（+15% 以上）；<br>3. 文档平均激活非零词项数（Active Non-Zero Length）受控在 120 到 180 之间，确保倒排链检索延迟可控。 |
| **终止条件** | 1. 若经过 FLOPS 惩罚后，模型发生表示塌缩（平均非零词项数小于 15，退化为无扩展能力的极度残缺匹配），立即终止并调整 $\lambda_{\text{reg}}$ 权重调度策略；<br>2. 若非零词项数超过 350（退化为伪稠密向量），且连续 500 步未能收敛，终止实验。 |
| **预算上限** | 单卡（DGX Spark 或 16GB 本机 GPU）训练时长 ≤ 6 小时；总批次 ≤ 20,000 steps。 |

---

## 二、 核心网络架构与算法实现

### 1. 模型前向计算（实现见 `src/model.py`）

下方是示意代码；实际实现直接加载带 MLM 预测头的预训练模型，不另行初始化线性词表头。

输入序列经过 Transformer 编码器后，通过 MLM 词表投影头映射为全词表 Logits，经 ReLU 和对数变换后在序列长度维度取最大值：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpladeEncoder(nn.Module):
    def __init__(self, transformer_backbone, vocab_size=30522):
        super().__init__()
        self.transformer = transformer_backbone
        # 词表投影头（通常与预训练 MLM 权重共享）
        self.vocab_head = nn.Linear(transformer_backbone.config.hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # 隐藏状态序列 [B, L, D]
        hidden_states = outputs.last_hidden_state
        # 词表投影 [B, L, |V|]
        logits = self.vocab_head(hidden_states)
        
        # 序列最大值池化与非线性映射
        # w_j = max_i log(1 + ReLU(y_{i, j}))
        relu_log = torch.log(1.0 + F.relu(logits)) * attention_mask.unsqueeze(-1)
        sparse_vec, _ = torch.max(relu_log, dim=1) # [B, |V|]
        return sparse_vec
```

### 2. FLOPS 正则化损失（实现见 `src/loss.py`）

为了约束倒排链的平均查询开销，损失函数联合优化排序目标与 FLOPS 惩罚：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ranking}}(\mathbf{q}, \mathbf{d}^+, \mathbf{d}^-) + \lambda_{\text{reg}} \cdot \sum_{j=1}^{|V|} \left( \frac{1}{B} \sum_{k=1}^B |w_{k, j}| \right)^2$$

```python
class FlopsRegularizer(nn.Module):
    def forward(self, sparse_weights):
        # sparse_weights: [B, |V|]
        # 计算该批次内每个词项的平均激活幅度
        mean_abs = torch.mean(torch.abs(sparse_weights), dim=0)
        # 平方和惩罚长倒排链与高频词堆积
        flops_loss = torch.sum(mean_abs ** 2)
        return flops_loss
```

---

## 三、 目录结构与快速启动

```text
sparse-retrieval/
├── README.md              # 实验卡与规范说明
├── requirements.txt       # 固定版本的直接依赖；PyTorch 单独安装
├── src/
│   ├── model.py           # SPLADE 模型定义
│   ├── loss.py            # Ranking + FLOPS 正则化损失
│   ├── dataset.py         # 三元组样本数据管道
│   └── train.py           # 端到端训练主脚本
```

以下命令从本实验目录执行。训练数据不随仓库提供；自行准备 JSONL 文件，每行包含 `query`、`pos`、`neg` 三个文本字段。输出目录由 `--output_dir` 指定；未指定时使用 Git 忽略的 `.work/artifacts/`。脚本启动时占用新的或空的输出目录并保存模型权重和训练指标；目录非空时会停止，每次运行请指定新目录。即使加载模型或训练失败，保留的 `.training-run` 标记也会阻止后续运行覆盖该目录。

### 环境准备

训练建议使用支持 CUDA 的 NVIDIA GPU，并安装与驱动匹配的 PyTorch；脚本检测不到 CUDA 时会自动退回 CPU，启动日志中的 `[Device]` 可确认实际设备。实验卡中的 16 GB 是预算规划，不是已经验证的最低显存需求；批大小和序列长度会影响显存。模型、训练数据和输出需预留磁盘空间。本目录没有记录经过完整真实数据训练验证的依赖组合。

在独立 Python 环境中操作；Linux 示例：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 按 https://pytorch.org/get-started/locally/ 安装适合设备和驱动的 CUDA PyTorch
# requirements.txt 不含 torch，避免 pip 替换已安装的 CUDA 构建
python -m pip install -r requirements.txt
python -c "import torch, transformers; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'transformers:', transformers.__version__)"
python -m pip check
```

[`requirements.txt`](requirements.txt) 将直接依赖 Transformers 固定为 5.16.1；PyTorch 需单独按设备安装，传递依赖由 pip 解析。训练与评测脚本不直接使用 datasets、scipy 或 NumPy，因此清单不再安装这些无关依赖。`pip check` 和导入只检查环境，不代表训练结果已验证。

### 模型与数据

示例使用 [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)，首次运行会通过 Transformers 下载模型和分词器，也可将 `--model_name_or_path` 替换为已准备好的完整本地模型目录。应记录使用的模型 revision；当前命令未固定版本。仓库不提供训练数据，JSONL 每行示例：

```json
{"query":"What is SPLADE?","pos":"SPLADE learns sparse representations.","neg":"This passage concerns an unrelated topic."}
```

确认训练、验证和测试数据的来源与划分，避免查询或文档跨划分泄漏；当前脚本不会自动执行这些检查。内置合成样本是中文，如用其检查流程，建议将模型参数改为 `bert-base-chinese`。首次下载模型需要网络；使用本地模型时无需下载。

训练脚本记录损失和激活词项数，不会自动运行验证集评测或执行实验卡中的终止门槛。下面提供独立评测入口；仓库没有随附真实评测数据。下方的小样本检查不足以判断实验目标是否达成。

### 启动训练

```bash
python src/train.py \
  --model_name_or_path bert-base-uncased \
  --data_path /path/to/triplets.jsonl \
  --lambda_flops 0.0008 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir /path/to/new-output/splade-v1
```

仅验证训练链路时，可用以下命令下载固定版本的随机小模型，并显式使用四条内置合成样本。小模型和合成数据仅用于检查流程，不能评价真实检索效果；输出目录须不存在或为空：

```bash
MODEL_DIR=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("hf-internal-testing/tiny-random-bert", revision="f171d7baecaf37b5da5a3616d8833b9969753535"))')
python src/train.py --model_name_or_path "$MODEL_DIR" --synthetic_data \
  --epochs 1 --batch_size 2 --max_len_q 16 --max_len_d 32 \
  --output_dir /path/to/new-output/splade-smoke
```

在 NVIDIA GB10/Linux aarch64 上，可用下列入口从空环境安装固定依赖、下载固定 revision 的随机小模型，并在 CUDA 上训练四条内置合成样本。默认 PyTorch 构建适用于该设备；其他设备需按本机情况设置 `TORCH_SPEC` 和 `TORCH_INDEX_URL`。运行前为环境、模型缓存和结果选择仓库外的新目录：

```bash
export XL_RUN_ID="sparse-$(date -u +%Y%m%dT%H%M%SZ)"
export XL_DATA_DIR="/path/outside/repository/sparse-data/$XL_RUN_ID"
export XL_OUT_DIR="/path/outside/repository/sparse-runs/$XL_RUN_ID"
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/verify_clean_gpu.sh
```

2026-11-15 在 NVIDIA GB10/Linux aarch64 上从空虚拟环境运行该脚本：`pip check` 通过，Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1；CUDA 训练完成两次更新并保存权重和指标。模型为固定 revision 的随机 tiny BERT，数据为仓库内四条合成三元组。这只验证干净安装到 GPU 训练保存的路径，不代表检索质量，也未达到实验卡中的指标门槛；摘要见[干净环境 GPU 训练记录](evidence/clean-env-training-2026-11-15.json)。

2026-11-13 在 Linux ARM64、NVIDIA GB10 上，使用 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.22.2 和上述缓存模型完成两步 CUDA 训练，写出权重和指标；随后以另外构造的 1 条查询、2 篇文档跑通 BM25 与 CUDA SPLADE 评测。运行环境的 `pip check` 报告 `nvidia-cusparselt-cu13` 不支持该平台，因此这不是经过依赖一致性检查的可复制锁定环境。随机模型加载时会报告部分其他任务头的权重未使用。

2026-11-14 使用从已有环境复制并调整的独立环境，在同类设备上以固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594` 的 `bert-base-uncased` 和 32 条 SNLI 三元组完成 1 轮、8 步 CUDA 训练。环境为 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1、Datasets 5.2.0，`pip check` 通过。

另从 STS-B 验证集前 100 条中选出 20 对评分至少为 4 的句子，一句作查询、另一句作候选文档；仅排除了与训练文本完全相同的内容，未检查近重复。20 个候选、每条查询只标一个相关项的检查中，BM25、初始模型和训练后模型的 MRR@10 分别为 0.975、0.7375、0.85；训练后文档平均激活词项数为 447.35，高于实验卡的目标区间。这是人为构造的小候选池，标注不完整且数据 revision 未固定，只证明真实文本训练和评测入口能运行，不能据此判断通用检索效果或目标达成，也未验证全新安装。

数据路径不存在时脚本报错，不会自动换用合成数据。训练完成后检查输出目录内的 `splade_weights.pt` 和 `training_metrics.json`，后者记录每步损失及激活词项数；这只能说明训练产物已写出，不能证明检索指标达标。

### 独立检索评测

`src/evaluate.py` 可在同一候选文档集上分别计算 BM25 和模型的 MRR@10。需要另外准备三份 UTF-8 JSONL：

- `corpus.jsonl`：每行 `{"doc_id":"d1","text":"文档内容"}`。
- `queries.jsonl`：每行 `{"query_id":"q1","text":"查询内容"}`。
- `qrels.jsonl`：每行 `{"query_id":"q1","doc_id":"d1","relevance":1}`，大于零表示相关，零表示不相关。每条查询至少需要一个正例；ID 不得重复或指向不存在的记录。

真实评测须自行取得带许可的数据与标注，固定版本和划分，并检查训练集与评测集的查询、文档及近重复内容是否泄漏。四条训练演示样本不能替代独立测试集。下例使用相同的候选集和标注分别运行：

```bash
python src/evaluate.py --method bm25 --corpus /path/to/corpus.jsonl \
  --queries /path/to/queries.jsonl --qrels /path/to/qrels.jsonl --out /path/to/new-run/bm25.json
python src/evaluate.py --method splade --model /path/to/local-base-model \
  --checkpoint /path/to/trained/splade_weights.pt --device cuda \
  --corpus /path/to/corpus.jsonl --queries /path/to/queries.jsonl \
  --qrels /path/to/qrels.jsonl --out /path/to/new-run/splade.json
```

模型目录须与训练时的底模结构和分词器一致；评测会加载本地模型与 tokenizer，`--checkpoint` 是训练生成的完整权重。省略该参数可评估同一底模的初始表示。BM25 使用小写 Unicode 词和单个汉字分词，参数 `k1=1.2`、`b=0.75`；它是明确口径的基础对照，并非所有语言下的最优分词配置。两种方法对同一候选集打分，仅将正分文档列入结果，分数相同时按文档 ID 升序；MRR@10 是前 10 位中首个相关文档排名倒数的查询均值。SPLADE 同时报告权重大于 `1e-4` 的文档平均激活词项数，与训练日志的统计阈值一致。

输出含输入文件、模型目录内全部文件的哈希、参数与逐查询排名；不要覆盖既有输出。BM25 读取全文，SPLADE 默认将查询和文档分别截断为 64 和 256 个 token，因此两者输入的实际文本长度可能不同，比较时应结合截断影响。该实现会保存全部查询与文档之间的分数，适合可放入内存的小型评测集，不测倒排索引延迟。运行前仍须验证模型文件版本、数据来源和标注；相同候选集不等于覆盖全部可能相关文档。

要复跑上面的合成数据评测，可在完成小模型训练后执行：

```bash
EVAL_DIR=$(mktemp -d)
printf '%s\n' '{"doc_id":"d1","text":"SPLADE learns sparse representations"}' '{"doc_id":"d2","text":"The weather forecast predicts rain"}' > "$EVAL_DIR/corpus.jsonl"
printf '%s\n' '{"query_id":"q1","text":"sparse representations"}' > "$EVAL_DIR/queries.jsonl"
printf '%s\n' '{"query_id":"q1","doc_id":"d1","relevance":1}' > "$EVAL_DIR/qrels.jsonl"
python src/evaluate.py --method bm25 --corpus "$EVAL_DIR/corpus.jsonl" \
  --queries "$EVAL_DIR/queries.jsonl" --qrels "$EVAL_DIR/qrels.jsonl" --out "$EVAL_DIR/bm25.json"
python src/evaluate.py --method splade --model "$MODEL_DIR" \
  --checkpoint /path/to/new-output/splade-smoke/splade_weights.pt --device cuda \
  --corpus "$EVAL_DIR/corpus.jsonl" --queries "$EVAL_DIR/queries.jsonl" \
  --qrels "$EVAL_DIR/qrels.jsonl" --max-len-q 16 --max-len-d 32 --out "$EVAL_DIR/splade.json"
```

将 checkpoint 路径改为上一步实际指定的输出路径。该评测只有一条查询，结果不能用来比较检索质量。
