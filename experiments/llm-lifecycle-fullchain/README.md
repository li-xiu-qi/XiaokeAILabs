# LLM 全链路训练与微调可复现实验（llm-lifecycle-fullchain）

参考 [karpathy/nanochat](https://github.com/karpathy/nanochat) 的全栈流程重构：从零跑通大语言模型生命周期的八个环节——**BPE 分词器（GPT-4 风格正则预分词）→ 现代架构因果语言模型 → Muon 预训练（附 AdamW 基线）→ 对话格式 midtrain → 指令微调（SFT）→ GRPO 强化学习 → KV cache 生成 → 全链路评估**。全流程在 CPU 上约 5.5 分钟，无需 GPU、无需下载外部模型或数据，产物与指标落盘 `results/`。

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见本 README 与 `results/` 下的指标文件（`results/report.md` 为评估汇总）。

## 与 nanochat 的对应关系

nanochat 用单节点 8×H100 在约 3 小时内训练出 GPT-2 水平的对话模型；本实验把它的**流程与关键技术**等比例缩放到 CPU 玩具规模（约 2.5M 参数、37KB 合成语料），保留可复现的教学链路：

| nanochat | 本实验 | 说明 |
|---|---|---|
| `scripts/tok_train.py` + tiktoken | `01_train_tokenizer.py` | Byte-Level BPE + GPT-4 风格正则预分词（`tokenizers` 库实现） |
| `nanochat/gpt.py` | `scripts/llm_common.py` 模型部分 | RoPE、QK norm、relu² MLP、无 bias、untied 权重、嵌入后 norm、logit softcap、GQA 代码路径 |
| `nanochat/optim.py` MuonAdamW | `llm_common.Muon` + `build_optimizer` | 隐藏矩阵走 Muon（Newton-Schulz 正交化），嵌入/lm_head 走 AdamW |
| `scripts/base_train.py` | `03_pretrain.py` | 预训练 + 同 steps 同种子的 AdamW 基线对照 |
| midtraining 阶段 | `04_midtrain.py` | 对话格式适配（低 lr 续训） |
| `scripts/chat_sft.py` | `05_sft.py` | 指令微调，prompt 掩码 |
| `scripts/chat_rl.py` + `tasks/` | `06_rl_grpo.py` | GRPO：组采样、可验证精确匹配奖励、KL 锚定冻结参考模型 |
| `nanochat/engine.py` KV cache | `llm_common.KVCache` | 生成走 KV cache，与 naive 前向逐 token 一致（02 自检） |
| `scripts/base_eval.py` / report | `07_generate.py` + `08_eval_report.py` | 解码对照、温度效应、提速对比、四阶段评估矩阵 |

 nanochat 中面向大规模的基础设施（FSDP、FP8、FlashAttention-3、torchrun 分布式、Web chat UI）不在本实验范围内。

## 链路总览

```
data/corpus.txt ──► 01_train_tokenizer.py ──► results/tokenizer/tokenizer.json
   (+ sft.jsonl / rl.jsonl)                          │
                                                     ▼
                                            llm_common.py（模型 + Muon + KV cache）
                                                     │
        ┌──────────────────┬─────────────────────────┼──────────────────────┐
        ▼                  ▼                         ▼                      ▼
  03_pretrain.py     04_midtrain.py            05_sft.py            06_rl_grpo.py
  Muon vs AdamW      对话格式适配              指令微调            GRPO 可验证奖励
        │                  │                         │                      │
        ▼                  ▼                         ▼                      ▼
 pretrain.pt         midtrain.pt                sft.pt                rl.pt
 (pretrain_adamw.pt)                                    │                      │
        └──────────────────────┬───────────────────────┘                      │
                               ▼                                              ▼
                    07_generate.py（贪心/温度/KV cache 提速） ◄────────────────┘
                               │
                               ▼
                    08_eval_report.py ──► results/report.md + summary.json
```

## 环境要求

- Python ≥ 3.10，依赖见 `requirements.txt`（实测：torch 2.15.0+cpu、tokenizers 0.23.3、matplotlib 3.11.2）
- 无需 GPU、无需下载外部模型或数据，样本语料由 `01_train_tokenizer.py` 确定性生成
- 运行环境清单见 `results/run_manifest.json`
- 多核机器上小模型默认 4 线程最快（实测 8 线程因争用更慢），可用 `LLM_THREADS` 环境变量覆盖

## 一键复现

```bash
# 默认配置（CPU 约 5.5 分钟，实测 331s）
bash run_all.sh

# 快速冒烟（约 2-3 分钟，步数缩减）
bash run_all.sh --quick
```

`run_all.sh` 默认使用 PATH 中的 `python`，可用 `PYTHON` 覆盖（如 `PYTHON=./.venv/bin/python bash run_all.sh`）。也可单步执行：

```bash
python scripts/01_train_tokenizer.py   # 语料生成 + BPE 训练与导出
python scripts/02_model.py             # 模型自检（QK norm/RoPE/KV cache/softcap/GQA）
python scripts/03_pretrain.py          # 预训练（Muon vs AdamW 基线）
python scripts/04_midtrain.py          # 对话格式 midtrain
python scripts/05_sft.py               # 指令微调
python scripts/06_rl_grpo.py           # GRPO 强化学习
python scripts/07_generate.py          # 生成 + KV cache 提速对比
python scripts/08_eval_report.py       # 全链路评估与 report.md
```

## 目录结构

```
llm-lifecycle-fullchain/
├── README.md                    本文
├── requirements.txt
├── run_all.sh                   一键复现入口
├── scripts/
│   ├── llm_common.py            共享模块：模型、Muon、KV cache、数据与生成工具
│   ├── 01_train_tokenizer.py    语料生成 + GPT-4 风格正则预分词的 BPE 训练/导出/校验
│   ├── 02_model.py              模型自检（QK norm / RoPE / KV cache / softcap / GQA）
│   ├── 03_pretrain.py           Muon+AdamW 预训练 + AdamW 基线对照
│   ├── 04_midtrain.py           对话格式 midtrain
│   ├── 05_sft.py                指令微调（prompt 掩码）
│   ├── 06_rl_grpo.py            GRPO（可验证奖励 + KL 锚定）
│   ├── 07_generate.py           贪心/温度采样 + KV cache 提速对比
│   └── 08_eval_report.py        四阶段评估矩阵 + report.md
├── data/                        （01 生成）corpus.txt、sft.jsonl、rl.jsonl
└── results/                     （各步骤产出）
    ├── tokenizer/tokenizer.json + tokenizer_info.json
    ├── checkpoints/pretrain.pt、pretrain_adamw.pt、midtrain.pt、sft.pt、rl.pt
    ├── pretrain_metrics.json、midtrain_metrics.json、sft_metrics.json、rl_metrics.json
    ├── pretrain_loss.png、midtrain_loss.png、sft_loss.png、rl_reward.png
    ├── samples.md、samples.json
    ├── report.md、summary.json
    └── run_manifest.json
```

## 各环节实现要点

### 01 分词器（Byte-Level BPE + GPT-4 风格预分词）

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

对 pretrain / midtrain / sft / rl 四个 checkpoint 跑同一套评估：聊天问答命中率（SFT 留出集）+ 算术 exact-match（训练/留出），汇总写 `results/report.md` 与 `summary.json`。

## 实验结果（实测）

> 以下数值来自默认配置的一次完整运行（CPU，见 `results/run_manifest.json`）。

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
（1.3380）——9K token 的微型语料上 Muon 更快地把训练集背下来（train loss 0.09 vs 0.83），
过拟合出现得更早。这是数据规模的局限，不是优化器缺陷：nanochat 在数十亿 token 上
Muon 的优势才能完整发挥。

生成示例（完整见 `results/samples.md`）：

| prompt | RL 模型贪心 | SFT 模型贪心（RL 前） |
|---|---|---|
| `User: What is the capital of Japan?
Assistant:` | `Tokyo` | `Tokyo` |
| `User: What color is a banana?
Assistant:` | `The banana is yellow.` | `The banana is yellow.` |
| `User: What does Carol do?
Assistant:` | `Carol is a engineer.` | `Carol is a engineer.` |
| `User: What is 7 plus 5?
Assistant:` | `7` | `7` |

预训练 / midtrain 模型对同一批 prompt 只会续写语料片段且不遵循对话格式
（`results/samples.md` 有完整对照），SFT 后才按问题实体作答——这是「指令微调改变
行为分布」的直接证据。RL 后模型对算术题稳定输出数字形态（如 `7`），但多数还算不对
精确结果，符合 2M 参数玩具模型的预期。温度效应对照显示 T=0.8 输出稳定连贯，
T=2.5 出现多样性同时开始崩坏。

生成示例见 `results/samples.md`，四阶段评估矩阵见 `results/report.md`。

## 复现性和确定性

- 所有随机源均固定种子（语料生成 1234、训练 1234、验证评估 1235、生成采样 1234+序号），同机重跑结果逐位一致
- `02` 内置七项结构自检（形状 / QK norm 缩放不变性 / RoPE 相对位置性 / KV cache 与 naive 逐 token 一致含 batch=3 / softcap / relu² 与 SwiGLU 双路径 / tie-untie）；`03/04/05` 内置 loss 下降断言，未达标以非零码退出
- 每一步的完整配置写入对应 `*_metrics.json` 的 `config` 字段

## 边界与局限

- 词表只有 775：微型语料下 BPE merge 很快饱和，属预期现象；真实场景应增大语料
- 模型仅 2.07M 参数、上下文 128 token，只能回答训练分布内的简单事实问答，不具备泛化语言能力
- 语料为英文模板合成数据，中文及开放域任务不在本实验范围内
- Muon 在 9K token 语料上 250 步即过拟合（best val @ step 49），AdamW 基线衰减更慢；这是数据规模的局限
- GRPO 的奖励含标点宽容与数字形态整形，是玩具规模的简化；实测 RL 主要收益在训练集内算式（17%→23%），47 条留出算式不升反降（13%→9%）——模型没有学会加法运算，只是背下了训练算式，泛化边界如实呈现
- RL 存在对话格式回归风险，本实验用聊天问答命中率量化监测（本次 60%→60% 无回归）
- Midtrain 的对话记录同时是预训练语料的一部分，小数据续训后 val loss 上扬（过拟合），该阶段只为格式适配，硬性门槛只查 train loss 下降
- KV cache 的收益随 batch 与生成长度增长：batch=8 时 4.9x，batch=1 仅 1.4x；模型再小或生成再短，cache 的每步固定开销会吃掉收益
