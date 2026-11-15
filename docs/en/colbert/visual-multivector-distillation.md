# Visual multi-vector distillation

> **Reading guide**: [Series index](README.md) · [Baseline and principles](visual-multivector-retrieval.md) · [Fine-tuning](visual-multivector-finetuning.md)

[New-page holdout](../../../experiments/visual-multivector-retrieval/independent-evaluation/README.en.md)

Other approaches are possible: [Unveil](https://aclanthology.org/2025.acl-long.1166/) transfers visual-text teacher representations to a visual document retriever. Its models and training objective differ from this experiment, so its reported results are not ours.

## Distilling a multimodal reranker

The single-vector Jina v5 teacher achieved R@1 56/96 on these test queries, below the original ColSmol's 87/96. More parameters alone do not make a better teacher for this task. We therefore used [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B), licensed Apache-2.0, to score question–page pairs. On the first 12 test queries it ranked the known positive first 12 times; the original student did so 11 times. This was a small teacher feasibility probe, not a full test-set score, and exposed that test subset during teacher selection.

We reused the [fine-tuning experiment](visual-multivector-finetuning.md)'s 96/32/32 train/validation/test page split. Each of eight fixed training batches has two different pages and one query for each page; the teacher scored only the resulting 32 training query–page pairs. The teacher emits sigmoid relevance scores, which are converted to a two-candidate soft distribution at temperature 0.1. Student query-length-normalized MaxSim scores use temperature 0.02:

$$p^T_{ij}=\operatorname{softmax}_j(t_{ij}/0.1),\qquad p^S_{ij}=\operatorname{softmax}_j(\bar{s}_{ij}/0.02),$$
$$\mathcal L=0.5\mathcal L_{\mathrm{supervised}}+0.5D_{\mathrm{KL}}(p^T\Vert p^S).$$

Here $t$ is the teacher's sigmoid output in [0, 1], not an unnormalized logit. The student learns both paired labels and the teacher's relative preferences; it does not need the teacher at inference time.

| Condition | Recall@1 | Recall@5 |
|---|---:|---:|
| Original ColSmol | 87/96 | 96/96 |
| Equal-budget supervised | 88/96 | 96/96 |
| Supervised + teacher distillation | 88/96 | 96/96 |

Compared with supervised training, two queries moved into first place and two moved out, for no net change. The original-model per-query ranks matched exactly across both runs. This experiment does not show a distillation gain. Eight steps, a single seed, two candidates per batch, shared synthetic templates and test-subset exposure during teacher selection limit the conclusion. The teacher is a pairwise reranker and the student a multi-vector retriever; these results do not compare their end-to-end speed.

[qwen-train-scores.json](../../../experiments/visual-multivector-retrieval/evidence/qwen-train-scores.json) contains the synthetic training batches and teacher scores. [synthetic-train-results.json](../../../experiments/visual-multivector-retrieval/evidence/synthetic-train-results.json) and [synthetic-distill-results.json](../../../experiments/visual-multivector-retrieval/evidence/synthetic-distill-results.json) contain anonymous per-query ranks. `python scripts/recompute_distillation.py` recalculates metrics and changes and checks that teacher-scored batches contain training records only; ranks alone cannot reproduce inference.

```bash
# Download Qwen 2B from its official model repository into qwen-model first.
PYTHONPATH=../multimodal-reranking/scripts HF_HUB_OFFLINE=1 python scripts/qwen_train_scores.py --root synthetic --model qwen-model --out qwen-train-scores.json
HF_HUB_OFFLINE=1 python scripts/distill_synthetic.py --root synthetic --base base --adapter model --teacher qwen-train-scores.json --out distill-formal --steps 8
python scripts/recompute_distillation.py
```

Teacher scoring uses `multimodal-reranking/scripts/qwen3_vl_reranker.py` and its Qwen image-processing dependencies. Teacher and student require different Transformers versions and were run in separate environments. The teacher used BF16 with maximum image pixels 65536; student distillation peaked at about 7.68 GiB of PyTorch allocation. No teacher or distilled weights are redistributed. The Jina teacher is CC BY-NC 4.0; its noncommercial research result did not supply scores for this distillation run.
