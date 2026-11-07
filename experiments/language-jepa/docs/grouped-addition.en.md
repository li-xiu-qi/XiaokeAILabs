# Addition data without swapped-pair leakage: base evaluation and next comparison

[简体中文](grouped-addition.md) | [Experiment](../README.en.md)

In the original experiment, 21/40 test questions had the same operands as a training question in reverse order, and the base model answered all 40 correctly. The new synthetic set uses two three-digit operands with carries in both the ones and tens places. Unique unordered operand pairs are split into 320 training and 80 test examples using seed 20261107. Grouping prevents swapped pairs from crossing the split; it cannot rule out shared answers, similar arithmetic or pretraining exposure.

## Three evaluations on the same test set

On 2026-11-07, Qwen2.5-1.5B-Instruct scored **73/80 (91.25%)** exact matches on DGX Spark. Decoding was greedy with at most 12 new tokens; output was compared literally with the expected answer after trimming whitespace. On the same day we trained ordinary LoRA and LoRA with a JEPA objective for one epoch and 40 updates each, then evaluated all three on the same test set:

| Model | Exact matches |
| --- | ---: |
| Base | 73/80 |
| Ordinary LoRA | 74/80 |
| JEPA LoRA | 75/80 |

Relative to ordinary LoRA, JEPA changed one answer from incorrect to correct and none from correct to incorrect. One training seed and 80 synthetic questions, on which the base is already strong, cannot establish a reliable JEPA improvement. The cause of the earlier training-log loss discrepancy remains unresolved.

The run used PyTorch 2.14.0+cu130, Transformers 5.19.0 and PEFT 0.20.0. Model weight SHA-256 was `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`; model revision was not recorded. Train/test JSONL SHA-256 values were `ffc78b4df9588ef09709c122756146ef4740d78e944083d57ef7480ec732c518` and `ac6c4398f4b2a47df31b953e54bcc51e1c989545d6c45869a819b7c4adc18b9d`. The [per-answer result](../results/grouped-comparison.json) contains no prompts. Reconstruct expected answers and recalculate all three scores and paired differences with:

```bash
python scripts/audit_grouped_comparison.py
```

## Prepare and validate

The output directory must not exist. Validate labels, within-split duplicates and swapped-pair leakage before training:

```bash
python scripts/make_grouped_data.py --output /path/to/new-data
python scripts/audit_grouped_data.py --data /path/to/new-data
python scripts/audit_tokenizer.py --model /path/to/model --data /path/to/new-data/train.jsonl --max-length 128
```

On Spark the tokenizer prefix and target checks passed on all 320 training and 80 test records. CPU negative checks rejected swapped pairs across the split and incorrect answers. Synthetic arithmetic is not representative of natural-language tasks.

## Comparison settings and limitations

Both arms used the same training data, batch 2, accumulation 4, learning rate 2e-5, LoRA rank 8 and seed 42; the JEPA weight was 0.1. Trainer-reported training times were 35.43 and 70.24 seconds respectively. Their objectives and per-step compute differ: these are single-run training-stage timings, exclude model loading and final saving, and are not an equal-compute comparison. Reported training losses were 0.4574 and 0.05867; different objectives make the values unsuitable for direct comparison.

The earlier investigation found padding-dependent forward results under GPU BF16; the full cause remains unresolved ([investigation](../PADDING-AND-PRECISION.en.md)). The scores and timings describe this run and cannot be attributed to JEPA. A stronger test requires resolving the numerical anomaly, independent test data, multiple training seeds and measured compute.
