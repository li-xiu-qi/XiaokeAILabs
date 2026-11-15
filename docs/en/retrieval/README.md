# Retrieval and ranking: from candidates to relevance

> **Reading guide**: [中文](../../zh/retrieval/README.md) · [All series](../README.md)

Search first retrieves candidates and then ranks them against the query. Single-vector, multi-vector and reranking approaches differ in supervision, indexing and evaluation requirements.

| Question | Start here |
|---|---|
| How do multi-vector models match text and images? | [ColBERT and multi-vector retrieval](../colbert/README.md) |
| How are embedding and reranking data, training and evaluation organized? | [Embedding models](../../zh/retrieval/embedding/README.md) · [Reranking models](../../zh/retrieval/reranking/README.md) |
| Can teacher scores improve a smaller model? | [Embedding distillation](../../zh/retrieval/embedding/Embedding蒸馏.md) · [Reranker distillation](../../zh/retrieval/reranking/重排序蒸馏.md) · [ColBERT distillation](../colbert/colbert-distillation.md) |
| How do vector indexes and databases trade off resources? | [Vector indexes](../../zh/retrieval/vector-search/README.md) · [Database benchmarks](../../zh/retrieval/vector-search/README.md) |
| How can multimodal candidates be ranked? | [Multimodal reranking](../../zh/retrieval/reranking/多模态重排序.md) · [Multimodal embedding distillation](embedding/multimodal-embedding-distillation.md) |

Ranking on fixed candidates does not establish full-corpus recall. See each experiment for its data and comparison conditions. Some individual experiment pages are currently in Chinese.
