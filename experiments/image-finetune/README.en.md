# Image Fine-Tuning: Teaching a Model a Specific Person from a Few Photos

Starting from a small set of photos, this experiment adds the ability to generate a specific person to an existing image-generation model, then uses quantitative metrics to check whether the identity was learned. See [Image Fine-Tuning and Quantitative Evaluation](../../docs/zh/multimodal/人脸身份LoRA微调与量化评估.md) for the explanation and conclusions.

| Directory | Contents |
|---|---|
| [face-identity-lora](face-identity-lora/README.md) | Fine-tunes an SDXL LoRA on 20 photos of one CelebA identity and evaluates identity specificity with ArcFace similarity plus positive and negative controls |

The [assets](assets/) directory contains an SDXL LoRA workflow diagram.

Before running, enter the specific sub-experiment and follow its environment setup instructions to install dependencies and prepare the model and data. See the [Chinese README](README.md).

## Notes

- Training data is not included in the repository. CelebA's data terms restrict use to non-commercial research and prohibit redistribution of images. The source and terms are described in the [face-identity-lora README](face-identity-lora/README.md). Keep training data only on the machine where training runs.
- This directory retains only the converged scripts, configurations, report, and evidence figures. Deprecated sub-experiments are no longer kept in the repository.
