# ColBERT and multi-vector retrieval

> **Reading guide**: [中文](../../zh/retrieval/colbert/README.md) · [Experiment map](../README.md)

This series examines how multiple query and document vectors support fine-grained matching, and the resulting computation and storage trade-offs.

Start with [ColBERT and multi-vector retrieval](colbert-and-multivector-retrieval.md) for MaxSim, retrieval paths and comparisons with single-vector and cross-encoder models. Then explore the experiments:

| Experiment | Question | Guide | Code and reproduction |
|---|---|---|---|
| Text supervised fine-tuning | Can relevance labels improve candidate ranking? | [Read](../../zh/retrieval/colbert/文本多向量检索模型微调.md) | [Experiment](../../../experiments/colbert-finetuning/README.en.md) |
| Text distillation | What do teacher scores contribute compared with labels? | [Read](colbert-distillation.md) | [Experiment](../../../experiments/colbert-distillation/README.en.md) |
| Visual multi-vector retrieval | Can text questions retrieve page images? | [Principles and baseline](visual-multivector-retrieval.md) · [Fine-tuning](visual-multivector-finetuning.md) · [Distillation](visual-multivector-distillation.md) · [Holdout](../../../experiments/visual-multivector-retrieval/independent-evaluation/README.en.md) | [Experiment](../../../experiments/visual-multivector-retrieval/README.en.md) |

ColSmol shares the multi-vector matching idea with text ColBERT but is a different model. Datasets, training objectives and evaluation scopes differ. The real teacher-scoring path in the distillation experiment has not been rerun; see each experiment for its limits. The text fine-tuning guide is currently in Chinese.
