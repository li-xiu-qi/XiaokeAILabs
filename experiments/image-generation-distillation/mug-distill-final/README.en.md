# Training a smaller image model on teacher-generated data

Can a larger model's synthetic images improve a smaller model on one narrow task? We compare SDXL teacher images with SD 1.5 self-generated images for 128 ceramic-mug prompts. Both LoRA students start from the same base and train for the same budget. A held-out test covers 32 unseen attribute combinations with two seeds each. Two AI review rounds favored the teacher-data student over the self-data control; see [results and limits](RESULTS.en.md).

This is synthetic output-data supervision, not direct matching of internal denoising predictions. `train.jsonl`, `development.jsonl`, and `test.jsonl` record prompts, seeds, mug color, background, view, and lighting. Training and testing use disjoint attribute combinations; the models may still have encountered mugs during pretraining.

## Method

1. On the development set, compare 1024 px SDXL images resized to 512 px with 512 px base-student images. Both use 30 steps and guidance 7.5. The continuation gate requires at least 28 valid pairs, 20 teacher successes, and an advantage of at least 4. The locally frozen protocol has no independent trusted timestamp.
2. Generate both training sets with the same 128 prompts and seeds. `prepare_mug_training_data.py` verifies IDs, prompts, seeds, dimensions, and hashes and retains every image without selective quality filtering.
3. Train two SD 1.5 LoRAs using batch 1, gradient accumulation 4, 400 steps, learning rate 1e-4, seed 42, and FP16. `run_experiment.sh` serially generates data, trains both arms, and generates all four held-out groups. It refuses existing outputs and is not a checkpoint-resume mechanism.
4. Two independent AI review rounds score each image for a single recognizable mug, mug color, background color, and severe artifacts. A success requires all three positive criteria and no severe artifact; any unknown makes the composite unknown. Round 1 was split among eight AI agents; a different AI agent completed round 2. Two seeds from one scene are not independent scenes.

## Run and recompute

The recorded run used NVIDIA GB10, driver 580.173.02, PyTorch 2.14.0+cu130, diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0, datasets 5.0.1, peft 0.20.0, and Pillow 12.3.0. This is not a complete dependency lock. The base snapshots were [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` and [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) `462165984030d82259a11f4367a4eed129e94a7b`; comply with their licenses and derivative distribution terms.

Set `PYTHON`, `WORK`, `STUDENT_MODEL`, `TEACHER_MODEL`, and `TRAIN_SCRIPT` to local paths as shown in the Chinese README, then run `bash run_experiment.sh`. The actual diffusers training script SHA-256 was `d1958e329d01f5878a545de675b94e0b3dc86b6625d2db1eaffd08eee5684da5`. The parameterized public script passed a shell syntax check but has not itself completed a new GPU run; the real run used equivalent core commands on the device. The 50 GiB guard measures system MemAvailable, not GPU-specific memory. Base weight bytes were not individually hashed.

This directory contains ratings, post-review mappings, and image SHA-256 values. [Download all 256 held-out original images, manifest and hashes](https://github.com/li-xiu-qi/XiaokeAILabs/releases/tag/evidence-mug-final-20261107); the teacher originals are 1024 px and were resized to 512 px with LANCZOS for review. Both rounds were AI reviews and the black image is retained. Thumbnail screening cannot rule out tiny text. Use the `python recompute.py` command in the Chinese README to reproduce the score mapping and 32-scene summary with Python 3.10 or later. Training images and LoRA weights are not included.
