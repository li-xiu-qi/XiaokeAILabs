# Run Record: Five Training Signals

[中文](README.md)

This run fixed SciFact at 390 training, 89 development, and 300 test queries. Each query used up to 30 BM25 candidates; training candidates consisted of one labeled relevant item and one unlabeled item. Unlabeled does not mean irrelevant. See the [experiment entry point](../../README.md) for data preparation and model versions. The test set had been used in earlier experiments and is not an independent blind test.

`plan.json` stores the fixed candidate-pool hash, training conditions, random seeds, and statistics on teacher preferences among candidates. The five signals are `human`, `teacher_hard`, `teacher_soft`, `human_hard`, and `human_soft`. All groups use the same sampling plan and 180 optimization steps; the three seeds vary only training randomness. When hard rankings tie, the first item in the training pair (the labeled relevant item) is selected. Soft scores use the difference between the teacher's `yes` and `no` logits, converted with a two-item softmax at temperature 2; they are not calibrated relevance probabilities.

`*-training.json` stores loss at each training step, `*-eval.json` stores per-query metrics on the development and test sets, and `*-scores.json` stores student scores for each candidate. Filenames include the group and seed. `summary.json` contains per-seed means, sample standard deviations, and exploratory comparisons. From the experiment directory, run `python scripts/summarize-five-arms.py` to recompute the summary from per-query evidence. Query bootstrapping is performed after averaging across the three seeds; it does not include training variation between seeds and does not correct for multiple comparisons. Teacher and student use different input lengths and precision. Equal coefficients in a mixed loss do not imply equal gradient contributions from each term.

Model weights, original papers, and teacher scores are not included in this directory. The teacher is Qwen3-Reranker-8B and the student is ms-marco-TinyBERT-L2-v2; see the [original experiment record](../qwen3-8b/) for the teacher's development-set advantage and version.

The public five-group script was fully rerun on the same DGX Spark with the same teacher scores, candidate pool, and student checkpoint. Test nDCG@10 matched the original run for all 15 runs to nine decimal places. The pre-rerun teacher gate and run configuration are recorded in `public-replay-teacher-gate.json` and `public-replay-plan.json`. This does not guarantee identical values on other devices or software environments.
