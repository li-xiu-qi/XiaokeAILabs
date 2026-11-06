# Fine-tune an SDXL identity with LoRA

[中文](README.md) · [Image fine-tuning](../README.en.md)

Can a small set of photographs and a trigger phrase help a general image generator render a particular person? This single-identity experiment covers data preparation, LoRA training, image generation and face-embedding evaluation. Such a workflow can support authorized identity-specific content, but usefulness depends on similarity, visual quality and the target setting.

## Results and limits

The historical run used 20 training photographs and 10 held-out reference photographs of one CelebA identity. ArcFace encodes detected faces; the metric is cosine similarity between each generated face embedding and a target reference embedding. Images without a detected face are excluded from the mean.

| Condition | Resolution | Detected | Mean target similarity |
|---|---:|---:|---:|
| Base model | 768×768 | 12/20 | −0.0172 |
| LoRA, 500 steps | 512×512 | 6/10 | 0.3110 |
| LoRA, 1,000 steps | 512×512 | 7/10 | 0.3232 |
| LoRA, 2,000 steps | 512×512 | 9/10 | 0.2461 |

Sources: [baseline](docs/evidence/baseline.json) and [checkpoint scan](docs/evidence/checkpoint_scan.json). These summaries were transcribed from a prior script. The original generated images, model revision, complete dependency versions and training logs are not all available. **The baseline and LoRA results use different resolutions; their difference does not isolate the effect of fine-tuning.** The scan contains only 6–9 detected images per checkpoint for one identity, so it does not establish an optimal step or the cause of the later decline.

At 1,000 steps the same detected outputs averaged 0.3232 against the target and 0.0546 against one other identity. That observation does not establish reliable identity recognition or generalization to other people. The mean similarity between training photos and the target test reference was 0.7159; it is not a theoretical upper bound for generation. Source images are 178×218, but source resolution was not controlled, so it cannot be identified as the main bottleneck from this experiment.

The split separates photographs, not identities; shared capture conditions or near-duplicates may inflate apparent performance. There is no multi-identity, cross-scene or human identity evaluation. Failed face detections must also be reported.

## Mechanism and inputs

SDXL generates an image by progressively denoising under text conditioning. LoRA keeps the base weights frozen and learns a low-rank weight update:

$$W_{\mathrm{adapted}} = W_0 + \frac{\alpha}{r}BA$$

Here $W_0$ is the original weight matrix, $A$ and $B$ are learned low-rank matrices, and $r$ is the rank. Scaling details depend on the implementation. Training consumes images and captions and optimizes a denoising objective on noisy images with text conditioning; the exact prediction target and loss depend on the toolkit version. This configuration trains the UNet, not the text encoder.

Inference consumes a prompt, seed and LoRA weights and produces an image. The trigger is part of the caption, not an identity database. Identity retention requires held-out photos, varied prompts and a matched base-model comparison.

## Data and configuration

- Data: one CelebA identity, 20 training photos and 10 test references, split by photo. The [dataset agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) limits use to non-commercial research and prohibits further publication or distribution; check consent as well.
- Base: SDXL base; the original model revision was not recorded. LoRA rank 32, alpha 32, dropout 0.05.
- Training: 512×512, batch size 1, learning rate 1e-4, 2,000 steps, saved every 500 steps. See the [configuration](config/face_identity_lora.yaml).
- Evaluation: RetinaFace detection and ArcFace embeddings. Model and evaluation weights carry their own terms.

## Run the workflow

Run from this directory after installing dependencies, obtaining the data and weights, and checking the dataset path in the YAML. The individual original training images and model weights are not distributed here and the historical record is insufficient for exact reproduction. The evaluation script refuses to reuse existing output directories; set a new `FACE_OUT_DIR` for another run.

```bash
python scripts/build_dataset.py --parquet /path/to/celeba_shard0.parquet --also-negative
python scripts/check_dataset.py
AI_TOOLKIT=/path/to/ai-toolkit/run.py bash scripts/train_lora.sh

# Global options precede the subcommand.
python scripts/score_generations.py --size 512 baseline
python scripts/score_generations.py --size 512 lora --lora /path/to/adapter.safetensors
python scripts/score_generations.py --size 512 scan --run-dir /path/to/checkpoints
```

The evaluation script now explicitly configures DPM-Solver++; historical results predate the refactor and were not regenerated with this correction. To measure the effect of fine-tuning, regenerate baseline and LoRA outputs with matched resolution, scheduler, seeds and prompt template, and report all outputs including detection failures.

`build_dataset.py` exports images and captions; `check_dataset.py` checks the samples; `train_lora.sh` launches training; and `score_generations.py` generates and scores images. Check the individual licenses of the training toolkit, models and evaluation weights.
