# Embedding 研究总组

嵌入模型（Embedding）的全部实验按三个环节分组：怎么评（评测基准与训练数据）、怎么训（微调项目）、多模态怎么办（图文模型）。

三个子组各回答一类问题，互相不重叠：

| 子组 | 回答什么 | 内容 |
|---|---|---|
| [evaluation_and_data/](evaluation_and_data/) | 怎么评一个嵌入模型、微调数据从哪来 | MTEB / BEIR / C-MTEB 等基准教程，微调 notebook 与数据准备 |
| [model_finetuning/](model_finetuning/) | 怎么把模型训成业务要的样子 | 文本嵌入微调（对比学习）与 Reranker 微调（pointwise） |
| [multimodal_embedding/](multimodal_embedding/) | 图文嵌入模型怎么用、怎么训 | BGE-VL 快速验证与 CLIP 冻结层微调 |

不在这组的相邻主题：向量数据库选型看 `vector-db-benchmark`；文本分割算法已移入私有实验区，本仓不再保留。

关于「相似度分数在什么条件下会骗人」的专项测量（句子长度、代码与表格、跨模态），结论已并入文本分割实验的选型边界；那部分内容已移入私有实验区，本仓不再提供入口。
