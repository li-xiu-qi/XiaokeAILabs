# ColBERT and multi-vector retrieval: from token matching to experiments

> **Reading guide**: [Series index](README.md)

When searching papers, the treatment and outcome in a query may appear in different sentences of an abstract. Representing the entire abstract with one vector makes retrieval convenient, but can lose detail. ColBERT retains multiple token vectors for both the query and the document. Each part of the query finds its strongest document match, and the matches are combined into a relevance score. This is called **late interaction**.

This guide explains the common idea behind three experiments. Their data, training objectives and conclusions differ:

| Question | Experiment | Input and scope |
| --- | --- | --- |
| Can domain labels improve ranking? | [Supervised text ColBERT fine-tuning](../../zh/retrieval/colbert/文本多向量检索模型微调.md) | SciFact claims and paper abstracts; ranking on fixed candidates. |
| Do teacher scores add useful ranking information? | [Text ColBERT distillation](colbert-distillation.md) | Scientific-paper candidates; compares teacher scores, label supervision and the initial model. |
| Can we retrieve document pages without first transcribing them? | [ColSmol visual multi-vector retrieval](visual-multivector-retrieval.md) | Text questions and page images; fixed-candidate baseline plus synthetic-data fine-tuning and distillation. |

ColSmol uses ColBERT-style multi-vector matching, but it is not the same model as text ColBERT. Its page inputs and training objectives are described in the visual experiment.

## How matching works

Single-vector retrieval represents each document with one vector, while a cross-encoder jointly processes a query and each candidate. ColBERT pre-encodes documents as multiple vectors and applies MaxSim after encoding. Actual quality and cost depend on the model, workload and implementation; architecture alone does not establish a universal winner.

For text retrieval:

1. **Encode documents** into token vectors. For retrieval, these can be computed and stored ahead of time.
2. **Encode the query** into its own token vectors.
3. **Score a document** by finding the maximum similarity to any document vector for each query vector, then summing these maxima.
4. **Return results** by sorting the candidate documents by score. The score is used to rank candidates for the same query; it is not a relevance probability.

This scoring operation is called MaxSim. Given normalized query vectors $\widehat q_i$ and document vectors $\widehat d_j$, standard ColBERT uses:

$$
s(q,d)=\sum_i\max_j\widehat q_i^{\mathsf T}\widehat d_j.
$$

If two query tokens have maximum similarities 0.8 and 0.6, the resulting score is 1.4. These numbers illustrate the operation; they do not prove that either token found correct evidence. Implementations also handle padding, punctuation and query expansion. Different models and training objectives may apply different normalization.

In the visual experiment the document page is encoded as an image while the question remains text. The page is likewise represented by multiple vectors, allowing different parts of the question to match different page regions; those vectors are not inherently interpretable OCR boxes. The visual experiment's supervised and distillation training use query-length-normalized MaxSim with their own temperatures and losses. Refer to that experiment for the exact formulas and code.

## Where it fits in a search pipeline

Late interaction can rerank an existing shortlist or be combined with a multi-vector index for larger-scale retrieval. These paths need separate evaluation:

```text
Fixed-shortlist reranking: query → BM25 or another candidate retriever → ColBERT scores → ranked results
Multi-vector retrieval:    pre-encode and index documents; encode query → retrieve candidates → score and rank
```

Both text experiments in this series use the first path, ranking fixed BM25 candidates. The visual experiment evaluates fixed pools of 100 or 32 pages. These runs do not measure the cost of building a full multi-vector index, full-corpus recall or online latency. Reranking cannot recover a document omitted by the candidate retriever.

## Quality and cost

A single-vector model typically stores one vector per document; ColBERT keeps multiple vectors. For $N$ documents with an average of $L$ vectors each, $D$ dimensions and $b$ bytes per value, raw vector storage alone is approximately $NLDb$ bytes. This excludes the index, metadata, model parameters and runtime cache; it is not the actual size after compression. Matching also compares multiple query and document vectors, so candidate count, token count, vector dimension, compression and index design affect cost.

More vectors do not automatically ensure better ranking, and they do not prove that a particular deployment is slower. Whether document vectors are precomputed, shortlist size, hardware and batching all affect results. Selection requires measurements on the target data: ranking quality, full-corpus recall, index size, build cost, latency and throughput. The metrics reported by these experiments have narrower scopes and cannot establish unmeasured properties.

## References

- [ColBERT paper](https://arxiv.org/abs/2004.12832): late interaction and MaxSim.
- [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/): teacher supervision and residual compression; its reported results are not measurements from this repository.
- [ColPali paper](https://arxiv.org/abs/2407.01449) and [ColSmol model card](https://huggingface.co/vidore/colSmol-256M): visual document multi-vector retrieval.
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) and [PyLate](https://github.com/lightonai/pylate): open implementations for training, indexing and retrieval.
