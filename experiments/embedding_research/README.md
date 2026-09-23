# Embedding 研究总组

嵌入模型（Embedding）的全部实验按四个环节分组：怎么评（评测基准与训练数据）、行为边界在哪（相似度专项）、怎么训（微调项目）、多模态怎么办（图文模型）。

四个子组各回答一类问题，互相不重叠：

| 子组 | 回答什么 | 内容 |
|---|---|---|
| [evaluation_and_data/](evaluation_and_data/) | 怎么评一个嵌入模型、微调数据从哪来 | MTEB / BEIR / C-MTEB 等基准教程，微调 notebook 与数据准备 |
| [behavior_tests/](behavior_tests/) | 嵌入模型的相似度在什么条件下会骗人 | 三个单变量专项：句子长度（含跨语言扩展）、代码与表格、NLI 数据迁移 |
| [model_finetuning/](model_finetuning/) | 怎么把模型训成业务要的样子 | 文本嵌入微调（对比学习）与 Reranker 微调（pointwise） |
| [multimodal_embedding/](multimodal_embedding/) | 图文嵌入模型怎么用、怎么训 | BGE-VL 快速验证与 CLIP 冻结层微调 |

不在这组的相邻主题，都在 `experiments/` 平级：文本分割算法看 `test_text_segmentation`，嵌入的下游应用（聚类、查询路由、降维可视化）看 `clustering_and_dimreduction`，向量数据库选型看 `test_vector_db_bench`，图文模型部署流水线看 `jina_deployment`。
