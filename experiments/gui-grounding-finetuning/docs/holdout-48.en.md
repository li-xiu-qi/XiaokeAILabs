# GUI grounding: paired evaluation on a 48-example stratified holdout

[简体中文](holdout-48.md) | [Experiment](../README.en.md)

On 2026-11-07 we evaluated the existing LoRA adapter on ScreenSpot images excluded from training and from the earlier 30-example evaluation. A hit means the predicted point falls within the target box; it does not mean an action was executed or a task succeeded.

| Paired outcome | Count |
| --- | ---: |
| Base model hits | 34/48 |
| LoRA hits | 42/48 |
| Incorrect → correct | 8 |
| Correct → incorrect | 0 |

The difference is +16.7 percentage points. A 10,000-resample paired bootstrap with seed 20261107 gives a percentile interval of +6.25 to +27.08 points. This describes resampling variability for these examples only, not a guarantee of population accuracy or cross-application generalization. Six examples per source are insufficient for comparing applications.

## Sampling and limits

The [ScreenSpot dataset card](https://huggingface.co/datasets/bevaya/ScreenSpot) states Apache-2.0; screenshot contents may have additional rights or privacy restrictions. We do not publish images, instructions, coordinates, source row IDs or raw predictions. The [de-identified paired records](../evidence/holdout-48-paired.json) contain only a local sequence number, source, element type and the two hit flags; they cannot reconstruct screenshots or predictions.

We selected three examples from each of the 16 strata formed by eight sources (Android, forum, GitLab, iOS, macOS, shop, tool and Windows) and icon/text types, using seed 20261107. Each candidate was checked against training images, the earlier 30 images and already selected holdout images. We excluded exact SHA-256 matches and images with 64-bit difference-hash Hamming distance at most 5: four exact and two near duplicates. Perceptual hashes can reject unrelated screens or miss semantic overlap; they cannot exclude pretraining contamination. This small, filtered stratified set is not a representative random sample.

The dataset service returns current rows without pinning a revision; a seed alone cannot reproduce identical data. The evaluation manifest SHA-256 was `60ed4f4d021dcf95f73ffa190cc0abe5be997d7aecf93693650d1139320e4471`. Both runs recorded that digest, the prompt digest and the model path. Matching paths do not prove the weights are unchanged; the adapter file SHA-256 was `5b6a43819db604ea8b9f41fdefb551d5ab6177814e49ad14629c2e817184e2d4`. See the experiment entry for the model, training configuration and software versions. A freshly generated manifest is a new evaluation if upstream rows change.

## Recalculate and reproduce

The published metrics require neither images nor a model:

```bash
python scripts/audit_holdout_results.py
```

To prepare another holdout, first obtain the training and earlier evaluation manifests and images, the base model and adapter as described in the experiment. Run from this experiment directory. `TRAIN_DIR` and `PREVIOUS_DIR` contain their respective manifests and images; `HOLDOUT_DIR` must not already exist. The default 50 GiB available-memory stop is a guard used on this GB10, not a universal requirement.

```bash
python scripts/prepare_stratified_holdout.py --training "$TRAIN_DIR" \
  --previous "$PREVIOUS_DIR" --output "$HOLDOUT_DIR" --per-stratum 3
export GUI_EVAL_DIR="$HOLDOUT_DIR"
export GUI_ROOT="$HOME/gui-holdout-results"
export GUI_MODEL="/path/to/Qwen3-VL-2B-Instruct"
mkdir -p "$GUI_ROOT"
unset GUI_ADAPTER
python scripts/gui_eval_holdout.py
export GUI_ADAPTER="/path/to/adapter"
python scripts/gui_eval_holdout.py
python scripts/gui_summarize_holdout.py
```

Run the evaluations sequentially with the same manifest, prompt, image resize settings and greedy decoding. The summarizer checks manifest and prompt hashes, base path, adapter role and per-example labels, requires both outputs to cover the full manifest, and checks each hit against the saved point and box. Keep `GUI_EVAL_DIR` set to the original manifest directory while summarizing. It does not detect replaced model or adapter weights; keep them fixed and record their hashes. Evaluation outputs contain annotations and predictions and should remain private.

The summary is created exclusively; use a new `GUI_ROOT` for another run to avoid overwriting saved results.

The GPU evaluation used the prior normal-mode scripts. The later explicit checks and overwrite protection were CPU-tested and have not been rerun on GPU.
