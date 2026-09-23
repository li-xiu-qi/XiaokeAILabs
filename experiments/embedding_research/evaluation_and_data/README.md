# 评测基准与训练数据

嵌入模型的「怎么评」与「怎么备料」：主流评测基准是什么、怎么跑，微调前的训练数据从哪来、怎么处理。内容是中英双语的资料整理与 notebook，不是实测结论。

| 子目录 | 内容 |
|---|---|
| `test_embedding/test_evaluation/` | 评测基准教程（中英）：MSMARCO、MTEB（总览 / 榜单 / C-MTEB）、Sentence Transformers 评估、BEIR、MIRACL、MLDR |
| `test_embedding/test_embedding_finetune/` | 嵌入模型微调 notebook 与说明 |
| `test_embedding/test_embedding_fine_tuing_data_prepare/` | 微调训练数据准备 |

注意：vendored 文档里的部分相对链接（如 `../../examples/...`）指向未随附的上游仓库结构，原文已标注「上游路径」，属预期内的悬空引用，不是断链。

想看某个模型在具体场景下准不准，看 [../behavior_tests/](../behavior_tests/)；想看怎么动手训，看 [../model_finetuning/](../model_finetuning/)。
