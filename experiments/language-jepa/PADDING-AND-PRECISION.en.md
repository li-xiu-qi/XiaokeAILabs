# Padding, attention and training-loss diagnostics

[中文](PADDING-AND-PRECISION.md)

Two historical runs had very different training losses. These tests examine data, model forward passes and attention calculations; they show numerical sensitivity in this particular environment, but do not establish the complete cause of the historical difference.

## Fixed-input observations

On DGX Spark, we used Qwen2.5-1.5B-Instruct in BF16, PyTorch 2.14.0+cu130, Transformers 5.19.0 and PEFT 0.20.0. Two fixed generated training examples were evaluated with the same weights, labels and inference mode, changing batching and padding:

| Device and attention | Batch of two, length 128 | Batch of two, trimmed to 32 | Separate examples, length 128 |
| --- | ---: | ---: | ---: |
| CUDA SDPA | 3.0918 | 0.000201 | 0.000172, 0.000312 |
| CPU SDPA | 0.000209 | 0.000209 | 0.000157, 0.000277 |
| CUDA eager | 0.3047 | 0.000086 | 0.000137, 0.000105 |
| CPU eager | 0.000086 | 0.000086 | 0.000078, 0.000096 |

Values are mean cross-entropy over answer tokens. For another pair, CUDA SDPA changed from 4.8876 at length 128 to 0.000191 at length 32. Input and label hashes matched when the same process moved the model from CUDA to CPU; two checked weight samples also matched in byte hash and dtype. Independently computed cross-entropy agreed with the model loss, placing the difference in the forward pass rather than solely in loss reduction. All measured cases are in [the results file](results/padding-attention-observations.json).

On four fixed microbatches, no-gradient forward passes and individual backward passes with checkpointing on or off produced the same CUDA SDPA loss per batch; no optimizer update occurred and sampled trainable parameters remained unchanged. This excludes those mechanisms as explanations of batch differences in this controlled check, not in every possible run.

## Mask and first-layer checks

On CPU, the checked answer labels occupy token positions 28–31. The first-layer allowed attention positions and position IDs matched between the two attention implementations. Changing each answer token did not change the preceding position's logits. This covers only the positions checked, and does not prove the absence of leakage elsewhere.

For identical first-layer BF16 Q/K/V tensors, FP32 manual attention served as a local reference. CPU BF16 SDPA had mean absolute error 0.000166 and BF16 manual attention 0.01908. These local comparisons do not prove either implementation is more accurate for the complete model.

On CUDA, first-layer BF16 SDPA mean absolute error relative to FP32 manual attention was 0.000169 at length 128 and 0.000166 at length 32, although full-model losses differed substantially. The first layer alone therefore does not identify the cause or establish that a particular kernel is responsible.

## Rechecking and limitations

Prepare the generated data and patched author script as described in the [experiment README](README.md), then run with the recorded software and local model:

```bash
python scripts/inspect_padding_loss.py --trainer /path/to/patched-finetune.py --model /path/to/model --data /path/to/train.jsonl --device cpu
python scripts/inspect_padding_loss.py --trainer /path/to/patched-finetune.py --model /path/to/model --data /path/to/train.jsonl --device cuda
```

The public script was rerun on CPU for 16 conditions and the first-layer check. The CUDA numbers came from the original diagnostic scripts; the public CUDA path has not been separately rerun. Its optional `--first-layer-check` covers length 128 only and cannot directly recreate the CUDA length-32 first-layer figure. The script performs read-only forward passes and prints input and label hashes.

Sample indices depend on the generated data order. The model revision was not recorded, and differences in dependencies, hardware or weights may change results; check available memory before GPU use. Model weights and training outputs are not included. This is a single small diagnostic run, not a new comparison of trained-model capabilities. The historical loss difference remains unresolved.
