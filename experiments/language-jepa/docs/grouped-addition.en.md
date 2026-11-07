# Addition data without swapped-pair leakage: base evaluation and next comparison

[简体中文](grouped-addition.md) | [Experiment](../README.en.md)

In the original experiment, 21/40 test questions had the same operands as a training question in reverse order, and the base model answered all 40 correctly. The new synthetic set uses two three-digit operands with carries in both the ones and tens places. Unique unordered operand pairs are split into 320 training and 80 test examples using seed 20261107. Grouping prevents swapped pairs from crossing the split; it cannot rule out shared answers, similar arithmetic or pretraining exposure.

## Base result

On 2026-11-07, Qwen2.5-1.5B-Instruct scored **73/80 (91.25%)** exact matches on DGX Spark. Decoding was greedy with at most 12 new tokens; output was compared literally with the expected answer after trimming whitespace. The set is not saturated by this base model, but the model remains strong. Ordinary LoRA and JEPA training on the new data have not been run, so this result cannot establish their effects. The cause of the earlier training-log loss discrepancy remains unresolved.

The run used PyTorch 2.14.0+cu130, Transformers 5.19.0 and PEFT 0.20.0. Model weight SHA-256 was `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`; model revision was not recorded. Train/test JSONL SHA-256 values were `ffc78b4df9588ef09709c122756146ef4740d78e944083d57ef7480ec732c518` and `ac6c4398f4b2a47df31b953e54bcc51e1c989545d6c45869a819b7c4adc18b9d`. The [per-answer result](../results/grouped-base-eval.json) contains no prompts. Reconstruct the expected answers and recalculate 73/80 with:

```bash
python scripts/audit_grouped_base.py
```

## Prepare and validate

The output directory must not exist. Validate labels, within-split duplicates and swapped-pair leakage before training:

```bash
python scripts/make_grouped_data.py --output /path/to/new-data
python scripts/audit_grouped_data.py --data /path/to/new-data
python scripts/audit_tokenizer.py --model /path/to/model --data /path/to/new-data/train.jsonl --max-length 128
```

On Spark the tokenizer prefix and target checks passed on all 320 training and 80 test records. CPU negative checks rejected swapped pairs across the split and incorrect answers. Synthetic arithmetic is not representative of natural-language tasks.

## Planned comparison

Use the same data, chat template, decoding, LoRA and optimizer settings to compare the base model, ordinary LoRA and LoRA with a JEPA objective. Keep the test set fixed and save per-example outputs, time and memory over multiple training seeds. With 320 examples, batch 2, accumulation 4 and one epoch, each training arm performs 40 updates; JEPA does more work per step, so equal steps are not equal compute. Investigate the historical loss-reduction and precision discrepancies before attributing any score change to JEPA. This is a plan, not a completed training result.
