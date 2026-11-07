# Adapting a smaller video model with teacher-generated water ripples

English | [简体中文](README.md)

Can a smaller video model be adapted to a specific visual scene? We generated water-ripple clips with Wan2.2 TI2V 5B, selected six of them after inspection, and used them for multi-frame LoRA fine-tuning of Wan2.1 T2V 1.3B. Four held-out prompt and seed pairs show the student before and after training.

**Result:** The 200-step run and four paired comparisons completed. Loading the LoRA changed the frames, but the four pairs do not establish an overall quality improvement: one pair lacked clear ripples both before and after training. The six training videos were selected post hoc; this exploratory result is not a successful test of a preregistered hypothesis and does not generalize to other scenes.

## Why use teacher-generated clips?

A video model must handle content, style and change over time. For a narrow application, teacher-generated examples may help adapt a smaller model to its target scene. Smaller size could reduce deployment costs, but this experiment does not measure that claim; errors in the teacher clips can also become training data, so inspect both training clips and held-out outputs.

Here **distillation means supervised fine-tuning on teacher-generated videos and prompts**. The student does not match teacher logits, intermediate features or denoising trajectories. The experiment does not establish that the student reproduced the teacher's internal capabilities, and parameter count alone does not determine inference speed.

## Inputs, outputs and training objective

Each training item is a `(prompt, video)` pair: the prompt describes water ripples from a fixed camera and the teacher clip contains 49 frames. The student's VAE encodes the video into latents, its text encoder processes the prompt, and LoRA updates a small subset of the generation network's weights. At inference, a prompt and seed produce a sequence of frames.

LoRA represents an update with low-rank matrices:

$$W' = W + \frac{\alpha}{r}BA,$$

where $r$ is rank and $\alpha$ is a scaling parameter; both were 8 here. Video diffusion training optimizes a prediction loss on noisy latents. The exact target and noise parameterization depend on the training framework; loss alone is not a measure of image quality or motion accuracy.

## Setup

| Item | Configuration |
| --- | --- |
| Teacher | Wan2.2 TI2V 5B, ComfyUI-adapted weights; 512 × 512, 49 frames, 24 fps, 30 steps |
| Student | Wan2.1 T2V 1.3B, LoRA rank 8; 480 × 480, 49 frames, 200 training steps |
| Student comparison | Before and after: 480 × 480, 49 frames, 30 inference steps, CFG 5; identical prompt and seed within each pair |
| Training data | Eight planned teacher clips; two were excluded after inspection, leaving six for this exploratory run |
| Held out | Four prompts and fixed seeds not used in training; all outputs retained, including weak results |

Teacher and student differ in architecture and resolution, so this is not a fair speed or frame-quality ranking between them. The student training latent shape was `[1, 16, 13, 60, 60]`: 13 latent time positions show that this run used multi-frame information.

## Reproducing the approach

1. Prepare compatible model weights and a video training environment. Record component versions, weight sources and hashes, and check licenses. See [Wan2.2](https://github.com/Wan-Video/Wan2.2), the [ComfyUI teacher weights](https://huggingface.co/Comfy-Org/Wan-2.2), and [Wan2.1](https://github.com/Wan-Video/Wan2.1).
2. Fix training and held-out prompts and seeds before generation. Retain each teacher output, prompt, seed, workflow and log; inspect frame count, visual legibility and corruption. Do not replace held-out samples because their results are poor.
3. Put a same-named `.txt` caption beside each training video. Set 49 frames, 24 fps, 480-pixel resolution, rank 8, batch size 1, learning rate $10^{-4}$ and 200 steps, and save checkpoints. Verify that the training latent has more than one time position before committing to the run.
4. Generate a baseline for every held-out prompt. Load the adapter and generate again with exactly the same student model, prompts, seeds and sampling parameters. Verify that the adapter was injected, then inspect content, motion continuity and artifacts pair by pair.

Frameworks differ in data loading, schedulers and adapter formats; these settings describe this run and do not guarantee the same output in another environment.

### Scripts

This directory includes scripts for student-side data preparation, configuration, inference and frame comparison. **The original teacher training clips and this run's LoRA weights are not fully distributed here.** The published outputs can be reviewed, and a new experiment can use independently generated teacher clips; the historical training cannot be recreated byte-for-byte from this repository. Install [AI Toolkit](https://github.com/ostris/ai-toolkit) separately.

```bash
# From this experiment directory; supply teacher MP4s named with train-00 etc.
python scripts/prepare_data.py --manifest config/prompts.json --source /path/to/teacher-clips \
  --output ./work/train --ids train-00 train-02 train-03 train-04 train-06 train-07

python scripts/make_config.py --template config/train.yaml.template --data ./work/train \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers --output ./work/checkpoints \
  --config ./work/train.yaml
python /path/to/ai-toolkit/run.py ./work/train.yaml

# Identical held-out prompts, seeds and student inference parameters.
python scripts/generate.py --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --manifest config/prompts.json --output ./work/baseline
python scripts/generate.py --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --manifest config/prompts.json --output ./work/lora --lora /path/to/adapter.safetensors

# Print hashes and frame differences for the four published pairs.
python scripts/compare.py --baseline results/baseline --lora results/lora
```

The scripts require Python, NumPy and PyAV. Generation additionally requires PyTorch, Diffusers, Transformers, PEFT and a working CUDA environment. The recorded versions were PyTorch `2.14.0+cu130`, Diffusers `0.39.0.dev0`, Transformers `5.5.3`, PEFT `0.21.2` and PyAV `19.0.1`. Diffusers was a development version, and the AI Toolkit source commit was not recorded, so the environment is not completely pinned. Script parameterization and result checks were tested separately; a full rerun of the 200-step training has not been validated end to end. Set suitable memory monitoring and interruption limits for your device.

## Published video pairs and licensing

All four teacher, student baseline and student LoRA clips are available below. The teacher clips used ComfyUI at 512 × 512; the student clips used Diffusers at 480 × 480. Different models and sampling paths do not share the same noise trajectory even with the same seed. The teacher clips illustrate the samples, while only the student before/after pairs share model, dimensions and sampling settings.

| Held-out case | Teacher | Student baseline | Student + LoRA |
| --- | --- | --- | --- |
| 00 | [Video](results/teacher/eval-00.mp4) | [Video](results/baseline/eval-00.mp4) | [Video](results/lora/eval-00.mp4) |
| 01 | [Video](results/teacher/eval-01.mp4) | [Video](results/baseline/eval-01.mp4) | [Video](results/lora/eval-01.mp4) |
| 02 | [Video](results/teacher/eval-02.mp4) | [Video](results/baseline/eval-02.mp4) | [Video](results/lora/eval-02.mp4) |
| 03 | [Video](results/teacher/eval-03.mp4) | [Video](results/baseline/eval-03.mp4) | [Video](results/lora/eval-03.mp4) |

Teacher prompts, seeds, sampling settings and SHA-256 hashes are in the [held-out manifest](results/teacher/manifest.json). Videos were generated from prompts and contain no filmed people. The model cards for [Wan2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) and [Wan2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) state Apache-2.0; the actual teacher files came from the [Comfy-Org adaptation](https://huggingface.co/Comfy-Org/Wan-2.2). Follow the applicable license and attribution terms. The repository contains experimental video outputs, not model weights.

The [historical frame statistics and hashes](results/frame-comparison.json) remain available. In a subsequent environment, all eight file hashes matched, but mean paired frame differences were approximately 12.74, 2.44, 8.86 and 11.50, versus the historical 12.72, 2.35, 8.92 and 11.43. The cause is unresolved, so those pixel values are not strictly reproduced. Values are on the 0–255 scale and measure output differences, not quality. `compare.py` requires identical video filename sets and prints current hashes; it does not automatically verify them against the historical record. The videos have not been decoded again since the filename-set check changed.

## Observations on the four original pairs

These are unblinded human observations of all 49 frames, not a rating study or statistical test:

| Case | Before and after |
| --- | --- |
| 00 | Expanding rings appear in both; after LoRA they are thinner and fade toward the end. |
| 01 | Neither output has a clearly recognizable circular ripple. |
| 02 | Both have concentric patterns; after LoRA the patterns are smaller and fainter. |
| 03 | Both show water-surface texture; motion direction and physical plausibility are difficult to judge. |

Pixel differences do not establish more natural motion. At 24 fps, 49 frames last about two seconds and cannot establish long-term consistency. More independent scenes, blinded reviews, repeated seeds and explicit motion-quality measures would be needed for a stronger evaluation.

## Additional held-out comparison

Four new water-ripple descriptions, each with two new seeds, produced eight base-versus-LoRA pairs. One AI reviewer compared anonymized sampled frames and, after unblinding, preferred LoRA in 3 pairs, the base model in 2, marked 1 tie and 2 inconclusive. The prompts are related and the reviewer did not watch complete videos, so these observations do not establish motion continuity or improved quality. See the [videos, prompts and individual review records](results/heldout-followup/README.md).
