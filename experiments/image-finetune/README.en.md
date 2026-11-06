# Fine-tuning image generation with LoRA for an identity

[中文](README.md) · [Experiment and reproduction](face_identity_lora/README.en.md)

A general image generator may not consistently produce a particular person from a description alone. With authorized photographs, LoRA can associate a trigger phrase with that identity. This experiment presents data preparation, SDXL training and face-embedding evaluation using 20 training photographs and 10 held-out reference photographs.

In the historical record, the 1,000-step checkpoint generated at 512 pixels has a mean cosine similarity of 0.3232 to the target reference across seven detected faces; the 2,000-step checkpoint scored 0.2461. **The base-model baseline was generated at 768 pixels, so subtracting these scores does not isolate the effect of fine-tuning.** This is one identity, and images with no detected face are excluded from the similarity mean. See the [experiment](face_identity_lora/README.en.md) for conditions and limits.

Obtain the necessary data permissions and consent before training on a real person's images. The [CelebA agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) restricts use to non-commercial research and prohibits further publication or distribution of the dataset. The model and evaluation weights have their own terms. Historical summaries do not capture every revision needed for exact reproduction.

Images containing dataset photographs were removed from the current tree; existing Git history still contains those files.
