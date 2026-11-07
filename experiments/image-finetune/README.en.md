# Fine-tuning image generation with LoRA for an identity

[中文](README.md) · [Experiment and reproduction](face_identity_lora/README.en.md)

A general image generator may not consistently produce a particular person from a description alone. With authorized photographs, LoRA can associate a trigger phrase with that identity. This experiment presents data preparation, SDXL training and face-embedding evaluation using 20 training photographs and 10 held-out reference photographs.

In the historical record, the 1,000-step checkpoint generated at 512 pixels has a mean cosine similarity of 0.3232 to the target reference across seven detected faces; the 2,000-step checkpoint scored 0.2461. **The base-model baseline was generated at 768 pixels, so subtracting these scores does not isolate the effect of fine-tuning.** This is one identity, and images with no detected face are excluded from the similarity mean. See the [experiment](face_identity_lora/README.en.md) for conditions and limits.

The historical run used CelebA photographs; the surviving records do not establish that the depicted person consented to training or public release of generated likenesses. The [CelebA agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) limits use to non-commercial research, prohibits further publication or distribution of the dataset, and prohibits commercial use of its images and derived data. It states that the images came from the internet and are not owned by the distributor. Check applicable permissions separately before publishing generated likenesses or weights; dataset research permission is not evidence of the person's consent. The model and evaluation weights have their own terms. The surviving records do not support exact reproduction.

Images containing dataset photographs were removed from the current tree; existing Git history still contains those files.
