# 相似度分数由什么决定：长度、内容、语言、用词

> 2026-09-26 改名。原为「句子长度对相似度计算的影响」，随补做三（用词因素）落地，实验覆盖的问题从单因素扩展为四因素，原名不再覆盖内容范围。目录名 `test_sentence_length` 保持不变，避免既有引用与日志断链。

## 布局

| 位置 | 内容 |
|---|---|
| `code/` | 重做版实验代码（模块化：数据 / 引擎 / 入口 / 分析），2026-09-24 在 DGX Spark 上跑通 |
| `code/supplement.py` | 补做一（负对照 gap）与补做二（退火机制探针） |
| `code/supplement2.py` | 补做三（用词因素：功能词、同义替换、关键词重叠、纯功能词追加） |
| `results/` | 全量数据与报告（raw_sims.csv、raw_negative.csv、raw_shuffled.csv、raw_wording.csv、REPORT.md、figures/） |
| `sentence_length_similarity_experiments.ipynb` | 旧版小样本实验，每条件 1 对句子，保留供对照 |
| `multilingual_sentence_length_similarity_experiments.ipynb` | 早期占位空文件。多语言扩展已并入 `code/` 的重做版（语言对 zh-zh / en-en / zh-en），不再单独维护 |

## 测什么

核心问题：相似度分数由什么决定。四个可控因素，各自有直接读数。

| 因素 | 条件 | 主结论 |
|---|---|---|
| 长度（加内容） | rel / irr × 单侧 / 双侧 | 方向普适为负，与模型无关；幅度与模型强相关，极差 4.7 到 5.4 倍 |
| 内容相关性 | rel vs irr | 无关内容是最大的单一效应（irr_both 达 -0.150），相关内容也降 |
| 语言匹配 | zh-zh / en-en / zh-en | 跨语言基准由语言覆盖决定；不匹配语言上加长效应退火到噪声 |
| 用词 | mask_func / syn / kw1-3 / func2-8 / noise4 | 关键词重叠推高分数（kw3 达 +0.22），量级超过长度效应一个数量级；实词噪声比纯功能词更能拉低分数 |

控制维度：内容相关性（相关加长 / 无关加长）、加长侧（单侧 / 双侧）、语言对（同语言 / 跨语言）。20 个中英平行主题 × 5 个模型 × 条件 × 3 种语言对。

## 主结论（详见 results/REPORT.md）

方向上 5 个模型全部 4 个条件都是相似度下降，旧版「双边相关内容使相似度上升」不被 20 对中位数支持。幅度与模型强相关，极差达 4.7 倍（irr_both 条件 large-zh 降 0.150，minilm 降 0.028），敏感度排序 bge-large-zh > bge-small-zh ≈ bge-m3 > jina-v3 > minilm-l6。

跨语言基准由模型的语言覆盖决定：多语言模型折扣 0.02 到 0.03，中文专用模型 0.11 到 0.15，英文专用模型跨语言直接归零。模型遇上不匹配的语言时赘述效应退火：中文专用模型在英文场景的双侧无关加长效应从 -0.132 衰减到 -0.007，双侧相关加长翻正（+0.059）；英文模型在中文场景同模式（-0.091 到 -0.028）。机制经乱文本探针修正：退火源于噪声的语义内容不进入模型表示，与表层共性无关（结果七）。负对照实验（190 个跨主题无关句对）推翻「中文专用模型完全不懂英文」：它们对英文仍有 AUC 0.74 到 0.81 的分离能力，gap 腰斩未归零；jina-v3 的 gap 与 AUC 跨语言完全稳定，是唯一的语言无关 discrimination。详见 results/REPORT.md 结果六、结果七与局限节。

用词因素（补做三）给出两个可操作的读数。注入对方句子的实词能把相似度推高 0.13 到 0.22，超过任何长度效应的量级，固定阈值的检索会被词形重叠骗过。等词数的实词噪声比纯功能词更能拉低相似度（差值 0.007 到 0.087），说明主实验测到的长度效应里有一部分实际是内容效应。

## 复现

DGX Spark 上（HF 缓存离线加载）：

```bash
cd ~/sentence_length_lab
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiment.py
~/xinfer-env/bin/python analyze.py ../results/raw_sims.csv
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python supplement.py      # 补做一、二
~/xinfer-env/bin/python supplement.py analyze
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python supplement2.py     # 补做三
~/xinfer-env/bin/python supplement2.py analyze
```

注意 DGX 上脚本与 `results/` 同级（平铺布局），`supplement2.py` 因此覆盖了 `config.RESULTS_DIR`。模型池、条件定义见 `code/config.py`；句对与噪声池见 `code/themes.py`；同义替换表与功能词表见 `code/supplement2.py`。jina-v3 在本机 transformers 5.16.1 下需 `code/lab.py` 的兼容 patch（见该文件 docstring）。
