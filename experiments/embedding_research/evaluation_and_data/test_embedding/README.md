# Embedding 模型：评测基准与微调数据集

嵌入模型的两件事：怎么评（MTEB、BEIR、C-MTEB 等主流基准是什么、怎么跑）和怎么备料（微调前训练数据从哪来、怎么处理）。文档以中英双语整理，原始来源为上游开源项目文档，部分相对链接指向未随附的上游结构。

## 子目录

| 目录 | 内容 |
|---|---|
| `test_evaluation/` | 评测基准教程中英双语：MSMARCO、MTEB（总览/榜单/C-MTEB）、Sentence Transformers 评估、BEIR、MIRACL、MLDR |
| `test_embedding_finetune/` | 嵌入模型微调的 notebook 与说明（中英） |
| `test_embedding_fine_tuing_data_prepare/` | 微调训练数据准备 |

## 从哪读起

要选型，先看 `test_evaluation/zh/markdown/4.2.1_MTEB_Intro.md`（基准总览）和 `4.2.3_C-MTEB.md`（中文基准，选型时最相关的一篇）。要复现评测，直接跑对应编号的 notebook，`utils/` 是公共工具。

要微调，先 `test_embedding_fine_tuing_data_prepare/` 备料，再看 `test_embedding_finetune/` 的完整流程。
