# 文本分割算法实验

> 设备：dgx-spark（GB10, 128GB 统一内存）· 模型：BAAI/bge-small-zh-v1.5 · 语料：wikipedia 20231101.zh 采样 300 篇

这个实验不测工具，测**分割算法本身**。一个 RAG 系统的检索质量上限，在建索引时就已被分块策略锁死：切错了地方，再好的向量模型也救不回来。本实验把八类分割算法放在同一把尺子上量，回答三个问题——切得准不准、块好不好用、值不值这个成本。

## 算法不是同一类东西

八种算法分属四类，它们的前提假设互不相容，混在一张表里比总数没有意义。

| 类别 | 算法 | 前提假设 |
|---|---|---|
| 规则 | fixed_128 / 256 / 512 / 256_sb | 文本是均匀的信息流，长度即语义单元 |
| 词法 | texttiling_w40 / w80, c99, bm25_p15 / p25 | 主题转移表现为词汇分布突变，无需理解语义 |
| 语义 | semantic_t050 / t060 / p10 / p25 | 相邻句向量距离能反映主题连续性 |
| 结构 | structural_512 | 内容自带结构标记，结构边界即语义边界 |
| 模型 | llm_pairwise / llm_segment | 只有语言模型能真正理解「这段在讲什么」 |

规则类不产生任何语义判断，它的存在是为了给坐标系定原点。词法类是零模型依赖的基线，在不能跑神经网络的场景仍然可用。语义类是当前主流，但依赖句向量质量这个隐含前提。结构类只对自带标记的内容有效，对纯文本段落几乎不产生断点。模型类质量上限最高，代价是延迟与成本高一个量级，且受内容审查限制。

## 评测轴

五个轴，每个回答一个不同的问题，互相不可替代。

**边界 F1（容忍窗 ±32/±64/±128 token）**：预测断点与 Wikipedia 天然段落边界的重合度。容忍窗必须随块长度缩放，否则会把「语义等价但偏一两句」的正确切法误判为错误。

**Pk / WindowDiff**：主题分割领域的标准指标。它对绝对位置不敏感，只看「相邻两块是否被错分到同一主题」，正好补 F1 的盲区。F1 高不代表聚类对。

**块尺寸分布**：均值、P95、最大值、超长块占比。工程价值的一半在这里——块太大塞不进上下文窗口或触发编码截断，光看召回会奖励切成碎片的算法。

**检索可用性（Recall@1/@5, MRR）**：终极判据。把块编码成向量后做检索，看答案块排第几。切分质量的唯一目的就是让检索认得出块。

**成本**：单篇耗时、LLM token 消耗。语义算法贵，要在质量-成本曲线上给出位置。

## 目录结构

```
scripts/
  corpus_loader.py        语料加载（wikipedia parquet → 带真值边界的 Doc）
  splitter_base.py        分割器统一接口（断点落在 token 空间）
  token_map.py            句子边界 → token 偏移换算
  model_hub.py            分词器与句向量编码器封装
  metrics.py              F1 / Pk / WindowDiff / 尺寸统计
  algo_*.py               八种算法实现
  run_eval.py             全量评测驱动
  verify_retrieval.py     检索可用性评测
corpus/                   wiki_eval_300.jsonl（300 篇采样语料）
results/                  逐算法 JSON + 汇总
docs/                     逐算法文档（公式推导 + 实测数据 + 选型边界）
```

## 复现

```bash
# 在 dgx-spark 上（192.168.1.170）
ssh ke@192.168.1.170
cd ~/text-seg
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/usr/local/cuda/bin:$PATH
export STEPFUN_API_KEY=<见 pkm-hub-configs/coding-cli-model-configs/keys.json>

# 语料准备（本机跑，采样后 scp 到 Spark）
HF_ENDPOINT=https://huggingface.co ../test_index_theory/.venv-embed/Scripts/python.exe -I \
  scripts/corpus_loader.py ../test_index_theory/data/wiki_zh corpus/wiki_eval_300.jsonl 300

# 全量评测
~/xinfer-env/bin/python -I scripts/run_eval.py --n-docs 300

# 只跑部分算法
~/xinfer-env/bin/python -I scripts/run_eval.py --n-docs 300 --only semantic_p10,bm25_p15

# 检索可用性（迟分 vs 传统编码）
~/xinfer-env/bin/python -I scripts/verify_retrieval.py --n-docs 50 --model BAAI/bge-m3
```

## 全量实测数据（300 篇 Wikipedia）

| 算法 | 类别 | F1@32 | F1@64 | F1@128 | Pk | 块数 | 均块 | 最大 | 秒/300篇 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_t060 | 语义 | **0.700** | 0.849 | 0.946 | 0.296 | 5417 | 84 | 559 | 9.5 |
| semantic_t050 | 语义 | 0.697 | 0.832 | 0.927 | 0.250 | 3713 | 123 | 594 | 9.8 |
| semantic_p25 | 语义 | 0.684 | 0.820 | 0.917 | 0.238 | 2882 | 158 | 595 | 9.3 |
| bm25_p25 | 词法 | 0.676 | 0.810 | 0.896 | 0.234 | 2542 | 179 | 1998 | 2.1 |
| bm25_p15 | 词法 | 0.671 | 0.797 | 0.879 | 0.230 | 2312 | 197 | 1998 | 2.0 |
| semantic_p10 | 语义 | 0.604 | 0.746 | 0.876 | 0.246 | 2183 | 208 | 595 | 9.3 |
| c99 | 词法 | 0.478 | 0.587 | 0.696 | 0.277 | 2313 | 195 | 3201 | 2.3 |
| fixed_128 | 规则 | 0.470 | 0.812 | 0.954 | 0.382 | 3701 | 123 | 128 | 1.7 |
| fixed_256 | 规则 | 0.288 | 0.524 | 0.826 | 0.331 | 1930 | 236 | 256 | 1.5 |
| fixed_256_sb | 规则 | 0.288 | 0.508 | 0.806 | 0.313 | 1930 | 236 | 419 | 2.0 |
| late_chunking_256 | 语义 | 0.195 | 0.375 | 0.633 | 0.309 | 1553 | 232 | 406 | 45.0 |
| texttiling_w40 | 词法 | 0.204 | 0.353 | 0.514 | 0.333 | 1875 | 243 | 1376 | 12.9 |
| fixed_512 | 规则 | 0.142 | 0.271 | 0.453 | 0.307 | 1038 | 438 | 512 | 1.4 |
| texttiling_w80 | 词法 | 0.089 | 0.159 | 0.276 | 0.303 | 936 | 486 | 2085 | 1.5 |

框架分块三件套与 DTC（2026-09-09 / 2026-09-16 补入，同口径）：

| 算法 | 类别 | F1@32 | F1@64 | F1@128 | Pk | 块数 | 均块 | 最大 | 秒/300篇 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chonkie_semantic | 语义(框架) | — | — | — | 0.284 | 4254 | 107 | — | — |
| sentence_window_w2 | 规则(框架) | — | — | — | 0.295 | 4210 | 108 | — | — |
| recursive_char_512 | 规则(框架) | — | — | — | 0.294 | 2319 | 196 | — | — |
| recursive_char_256 | 规则(框架) | — | — | — | 0.308 | 4253 | 107 | — | — |
| sentence_window_w1 | 规则(框架) | — | — | — | 0.302 | 6920 | 66 | — | — |
| dtc_50_200 | 词法(动态) | 0.642 | 0.854 | 0.953 | 0.350 | 6088 | 75 | 329 | 2.6 |

框架三件套的 F1 列未单独跑全量（当时只跑了 P_k 检索口径），数字以 pkm-hub 报告为准。DTC 为 2026-09-16 全量跑批实测。DTC 的 max=329 超 k_max=200 是句对齐累积无法切开超长单句的边界行为；密度反比机制在中文维基上把目标块大小压向 k_min（均块 75），块过碎导致 P_k 与检索双输 fixed_256。

Auto-Merge（HiChunk 检索侧复现，P&P 章级召回口径，2026-09-16）：baseline top-8 MRR=0.4670 vs auto_merge MRR=0.4627，平均 merge 0.7 次/查询几乎不触发——章级 QA 语料上「同父多子同时命中」的前置条件极少成立，是忠实复现的负结果（其设计场景是细粒度段落级 QA）。

structural 不在本表：纯文本无结构标记时它退化为段落累积，且逐行正则扫描复杂度爆炸。它的正确定位是 Markdown / 代码仓库。

LLM 分割未纳入本表：water18-new 不服从「只答 YES/NO」格式约束（已改结构化锚点），但 Wikipedia 语料中部分条目触发内容审查返回 451，需另建不含敏感条目的语料子集。

## 与既有实验目录的关系

本目录吸收并取代了四个早期实验，它们都不再维护：

| 旧目录 | 内容 | 去向 |
|---|---|---|
| `test_late_chunking/` | 迟分实现 + 红楼梦测试 | 迟分逻辑重写为 `algo_late_chunking.py`，原实现的硬编码模型路径 `C:\Users\k\Desktop\BaiduSyncdisk\...` 在本机不存在（用户是 `ke`），已失效 |
| `test_sentence_similarity_with_code_or_table/` | 代码/表格与文本的相似度评测 | 其结论（嵌入模型对结构化内容的分辨力）已并入选型边界，正文不重复 |
| `test_semantic_splitter/` | spaCy + 句向量断句 v1/v2/v3 | 核心逻辑并入 `algo_semantic_breakpoint.py`，补上百分位自适应判据与尺寸硬约束 |
| `test_hybrid_chunking/` | HuixiangDou 派生的 Markdown/文档切分 | 规则抽成 `algo_structural.py`，去掉 yaml 依赖 |

`test_semantic_splitter` 的 v1/v2/v3 三个版本与 English 变体命名混乱，吸收时已统一为单一实现，旧目录留作历史参考。

## 实测结论摘要

完整数据见 `results/` 与各 `docs/` 文档。300 篇 Wikipedia，BAAI/bge-small-zh-v1.5（迟分用 bge-m3）。**四个跨算法成立的判断：**

**一、纯规则切分在长文档上必然失配。** fixed_256 的 F1@32 只有 0.288，因为它与段落边界毫无关联。fixed_128 靠更细的粒度把 F1@32 拉到 0.470，代价是块均长仅 123 token。fixed_512 直接掉到 0.142，块太大时单个块跨多个段落，边界信号被平均掉。任何宣称「按 512 token 切就够了」的方案，都是在用容忍窗掩盖错位。

**二、词法方法在这个语料上不输语义方法，BM25 边界是全场第二。** bm25_p25 的 F1@32=0.676，只低于 semantic_t060 的 0.700，且超过其余三个语义配置（0.697 / 0.684 / 0.604）。原因在中文没有空格分词，虚词占比高，原始词频余弦会被高频共现拉高；BM25 的 IDF 加权正好压掉这个效应。这条推翻了「语义算法必然优于词法」的默认假设，至少在 Wikipedia 这种词汇密度高的语料上不成立。

**三、TextTiling 的平滑窗口是一把双刃剑。** w40 的 F1@32=0.204，把伪句长度加到 w80 后崩到 0.089。增大窗口本想压制局部噪声，实际把真实的主题谷底一起抹平了。C99 避开曲线改用矩阵密度，F1@32=0.478，明显更稳。

**四、语义算法的上限由句向量决定，不由算法决定。** semantic_t050 / t060 / p10 / p25 四个配置的 F1@32 落在 0.604-0.700，跨度为 0.096，而阈值与百分位的调参空间远大于此。说明瓶颈已不在断点判据，而在「相邻句相似度能否反映主题转移」这个信号本身。继续调参收益递减，要换模型或换信号。

### 迟分单独说明

迟分的 F1@32=0.195 与 fixed_256 的 0.288 同阶，这是设计如此：迟分不改变切点，切点仍是固定窗口。

**检索维度上迟分显著劣于传统编码，本实验不支持采用迟分。** 跨段检索 50 篇 100 查询，传统 R@1=0.910，迟分 R@1=0.250。更直接的证据来自构造的指代句：段边界切断指代关系时，传统编码下两段相似度 0.325（区分清晰），迟分下 **0.970**（几乎无法区分）。

根因是双向注意力的信息混合无法被窗口切分拦截：`last_hidden_state` 里每个 token 的表示已含全序列信息，段1的内容在段0的每个 token 里都有贡献。迟分用「块间区分度」换「块内上下文完整性」，而检索要求块间保持可区分，两者直接冲突。

迟分可能占优的场景（块作为生成上下文而非检索索引项）本实验未覆盖。成本上迟分无论如何不占优：单篇 0.15s，是 fixed_256 的 30 倍。详见 `docs/迟分模型.md`。

## 文献对照

论文仓（`pkm-hub-papers/论文阅读/papers/`）共登记 13 篇分块/分割论文，全仓扫描 meta.json 关键词确认无遗漏。2026-09-16 已从 arXiv HTML 转出 3 篇 markdown（2603.06976、2509.11552、2410.13070）；L14-1709 为 2014 年 ACL Anthology 论文无 HTML 版，仅有 PDF 文字层。完整登记如下：

| ID | 年份 | 论文 | 内容覆盖 | 与本实验的关系 |
|---|---|---|---|---|
| 2409.04701 | 2024-09 | Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models | 迟分原始论文（Jina AI）：token 级隐状态按窗口 mean pooling | 迟分实现的直接来源 |
| 2504.19754 | 2025-04 | Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG | 迟分 vs Contextual Retrieval 端到端对比，含效率权衡 | 对应 LLM 分块 vs 迟分对照 |
| 2607.01852 | 2026-07 | Evaluating Chunking Strategies for RAG on Academic Texts | 学术文本 chunking 系统评测，nDCG | 同协议第二个检索评测参照 |
| 2312.06648 | 2023-12 | Dense X Retrieval: What Retrieval Granularity Should We Use?（EMNLP 2024） | proposition 检索单元；附录含抽取提示词与质量自检表 | proposition 分块源头，LLM 分块算法模板 |
| 2407.01219 | 2024-07 | Searching for Best Practices in Retrieval-Augmented Generation | chunk size / overlap / 检索器系统超参分析 | 块尺寸结论的标准引用 |
| 2603.06976 | 2026-03 | A Systematic Investigation of Document Chunking | 36 策略 × 6 域 × 5 嵌入模型，nDCG@5 | 域适配权威来源（见下） |
| 2410.13070 | 2024-10 | Is Semantic Chunking Worth the Computational Cost? | 语义 vs fixed-size，文档/证据/生成三轴 | 语义分块成本效益（见下） |
| 2509.11552 | 2025-09 | HiChunk: Evaluating and Enhancing RAG with Hierarchical Chunking | 分层分块 + HiCBench benchmark + Auto-Merge 检索 | 评测协议同构；Auto-Merge 未覆盖方向 |
| cs/0003083 | 2000 | Advances in domain independent linear text segmentation | C99：rank transform + 余弦矩阵 divisive clustering | c99 算法的出处 |
| 2305.11553 | 2023-05 | Unsupervised Scientific Abstract Segmentation with Normalized Mutual Information | 无监督科学摘要分割 | 词法类方法的现代变体参考 |
| L14-1709 | 2014 | Segmentation evaluation metrics, a comparison grounded on prosodic and discourse units | 人工标注 + 三类注入扰动，对比 6 种指标 | F1 容忍窗偏差出处，Pk/WindowDiff 选择依据 |
| 2503.10677 | 2025-03 | A Survey on Knowledge-Oriented Retrieval-Augmented Generation | 知识导向 RAG 综述，5.3 节 chunking 谱系 | 分块策略谱系索引 |
| 2506.18959 | 2025-06 | From Web Search towards Agentic Deep Research | agentic 检索方向 | 背景参考，未直接对照 |

**域适配：我们的单域结论不能外推。** 2603.06976 的核心结论是「不存在普遍最优的分块策略，效果取决于目标域的结构与语义特征」：生物、物理、健康域上 Dynamic Token Size Chunking（按内容密度自适应块长）最优；法律、数学域上 Paragraph Group Chunking（保留多段落逻辑单元）全面占优；农业域异构，段落感知与迟分稳居前三。我们在 300 篇 Wikipedia 上「semantic_t060 第一、BM25 第二」的结论，在文献框架下属于高词汇密度域的特例，迁移到法律、数学这类多段落逻辑单元域之前必须重测。这一判断不是本实验独有，论文已用 6 域 × 36 策略的系统实验给出了同样的方向。

**模型规模：相对排序稳定，绝对水平上升。** 该文显示更大的嵌入模型提升绝对 nDCG@5，但各分块策略的相对排序基本不变；且「即使高容量编码器，次优分块仍给检索效果设上限」——嵌入质量与分块策略是互补而非替代关系。这与结论四「语义算法的上限由句向量决定」方向一致，但补充了一层：换更强的模型只抬绝对水位、不改策略排序，所以选型结论对模型不敏感。

**迟分的文献冲突：待解。** 该文中 Late Chunking Token Spans（LCTS）在全部 6 个域稳定排前三，而本实验的结论是迟分显著劣于传统编码（R@1 0.250 vs 0.910）。两个结果不必然矛盾，差异可能来自三处：其一，模型——该文的迟分搭配支持长上下文的通用编码器，bge-m3 未经迟分训练；其二，评测轴——该文用域内 QA 的 nDCG@5，答案常跨块，迟分的块内上下文完整性占优，而本实验的指代句测法是专门构造的块间区分度压力测试，对迟分是敌对设计；其三，窗口——本实验窗口固定 256 token，该文的 token spans 按语义边界切。要解这个冲突，需在本实验框架下换 Jina 系为迟分训练的模型重跑同一检索测法，区分负结果是模型问题还是轴问题。已列为下一步实验。

**评测协议与文献一致。** L14-1709 指出 P/R 类指标对 near-miss（边界偏移一两句）与增删边界同罚，设容忍阈值会引入偏差，WindowDiff 正是为解决该问题提出，且指标行为随数据结构变化。本实验五轴评测中 Pk/WindowDiff 恰好覆盖了该文指出的 F1 盲区，组合与该文 2014 年的建议同构。HiChunk 的分块点 F1（分层 L1/L2）与我们的边界 F1 同族；它还指出现有 RAG benchmark 的 evidence sparsity 不适合评测分块，这也是本实验用 Wikipedia 天然段落真值而非 QA 数据的理由。

**语义分块的成本争议。** 2410.13070 的结论与直觉相反：语义分块只在拼接数据集（高主题多样性）偶尔占优，真实非合成文档上 fixed-size 常更好，分块策略的影响常被嵌入质量掩盖。但注意轴的区别：该文测端到端 RAG 三轴（文档检索、证据检索、生成），本实验测纯边界 F1，两者不矛盾——边界质量与端到端检索质量不是同一件事，这也印证了「评测轴选错会得出相反结论」。

**未覆盖方向（文献已成熟）。** 2603.06976 的 Dynamic Token Size Chunking（内容密度自适应块长）和 2509.11552 的 HiChunk + Auto-Merge（分层分块配检索时合并）是两个明确的可扩展方向，均未在本实验复现。
