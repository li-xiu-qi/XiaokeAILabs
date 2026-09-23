# 句子长度对相似度计算的影响

## 布局

| 位置 | 内容 |
|---|---|
| `code/` | 重做版实验代码（模块化：数据 / 引擎 / 入口 / 分析），2026-09-24 在 DGX Spark 上跑通 |
| `results/` | 全量数据与报告（raw_sims.csv、summary_delta.csv、base_bench.csv、delta_long.csv、REPORT.md、figures/） |
| `sentence_length_similarity_experiments.ipynb` | 旧版小样本实验，每条件 1 对句子，保留供对照 |
| `multilingual_sentence_length_similarity_experiments.ipynb` | 早期占位空文件。多语言扩展已并入 `code/` 的重做版（语言对 zh-zh / en-en / zh-en），不再单独维护 |

## 测什么

一个变量：句子长度变化时，同主题句对的余弦相似度怎么动。控制内容相关性（相关加长 / 无关加长）、加长侧（单侧 / 双侧）、语言对（同语言 / 跨语言）三个维度。20 个中英平行主题 × 5 个模型 × 7 个条件 × 3 种语言对。

## 主结论（详见 results/REPORT.md）

方向上 5 个模型全部 4 个条件都是相似度下降，旧版「双边相关内容使相似度上升」不被 20 对中位数支持。幅度与模型强相关，极差达 4.7 倍（irr_both 条件 large-zh 降 0.150，minilm 降 0.028），敏感度排序 bge-large-zh > bge-small-zh ≈ bge-m3 > jina-v3 > minilm-l6。

跨语言基准由模型的语言覆盖决定：多语言模型折扣 0.02 到 0.03，中文专用模型 0.11 到 0.15，英文专用模型跨语言直接归零。模型遇上不匹配的语言时赘述效应退火：中文专用模型在英文场景的双侧无关加长效应从 -0.132 衰减到 -0.007，双侧相关加长翻正（+0.059）；英文模型在中文场景同模式（-0.091 到 -0.028）。「模型只支持某语言」是模型卡定位，不等于训练数据构成，该变量未受控，且结论用的是翻译句对而非未见语料，「不匹配语言上理解更差」的因果表述证据不足，详见 results/REPORT.md 结果四与局限节。

## 复现

DGX Spark 上（HF 缓存离线加载）：

```bash
cd ~/sentence_length_lab
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiment.py
~/xinfer-env/bin/python analyze.py ../results/raw_sims.csv
```

模型池、条件定义见 `code/config.py`；句对与噪声池见 `code/themes.py`。jina-v3 在本机 transformers 5.16.1 下需 `code/lab.py` 的兼容 patch（见该文件 docstring）。
