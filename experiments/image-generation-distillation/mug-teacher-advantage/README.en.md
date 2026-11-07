# Ceramic mugs: development-set teacher check

[简体中文](README.md) | [Experiment](../README.en.md)

On 32 paired development prompts, an SDXL teacher generates at 1024 px and is resized to 512 px, while an SD 1.5 baseline generates at 512 px. Both use 30 steps, guidance 7.5, and their native schedulers. Two AI reviewers independently rated shuffled images.

The designated first reviewer found 32 valid pairs: 23 teacher and 7 baseline composite successes, with 18 teacher-only and 2 baseline-only. The second found 29 valid pairs and 3 unknown pairs: 22 teacher and 7 baseline successes. Both pass the fixed gate: at least 28 valid pairs, 20 teacher successes, and an advantage of 4 or more.

Per-image agreement, excluding unknowns from either reviewer, was 61/62 for one mug, 63/64 for mug color, 61/64 for background, and 61/62 for severe artifacts. Composite success requires all three positive criteria and no severe artifact; an unknown excludes that pair.

This supports proceeding to a training control in this scene; it does not establish distillation success or isolate model, resolution, and scheduler effects. Both reviewers are AI, the sample is small, and the local freeze lacks an independent trusted timestamp. This compact package omits original and review images, so it cannot verify image content or image hashes. SHA-256 values are retained in the keys for comparison with separately provided images. Reproduce the score mapping and gate with Python 3.10 or later:

```bash
python reveal_compact.py --review review-1.csv --key mug-key-1.json --out replay-1.csv
python reveal_compact.py --review review-2.csv --key mug-key-2.json --out replay-2.csv
python check_mug_gate.py --scores replay-1.csv --development development.jsonl
python check_mug_gate.py --scores replay-2.csv --development development.jsonl
```
