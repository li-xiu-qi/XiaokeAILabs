# Check the teacher first: a failed image-generation gate

Before using a larger model's images to train a smaller model, a basic question is whether the teacher demonstrates an advantage worth transferring in the target scenario. We set a development gate before scoring SDXL and an unmodified SD 1.5 student on camera product images.

Each model generated one image for each of 32 attribute combinations. One AI reviewer inspected each anonymized original image at 512 × 512. Composite success required a single recognizable camera, the requested camera color and background, and no severe artifact. A pair was excluded from the composite comparison when either image contained an unknown rating.

| Development result | SD 1.5 student | SDXL teacher |
| --- | ---: | ---: |
| One recognizable camera | 12/32, plus 6 unknown | 6/32, plus 3 unknown |
| Correct background | 11/32 | 18/32 |
| Severe artifacts | 21/32 | 21/32 |
| Composite success in valid pairs | 1/24 | 0/24 |

Eight of 32 pairs had at least one unknown and were excluded from the composite comparison. The frozen gate required at least 28 valid pairs, at least 20 teacher successes, and at least four more teacher successes than the student. **The gate failed, so we stopped the planned fine-tuning and final test.** This is not a general model ranking: the single AI review is subjective and the sample is small. The models used their respective PNDM and Euler schedulers, and SDXL was run at 512px; [Diffusers documentation says most SDXL checkpoints work best at 1024px](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl). The result does not establish how the teacher performs at another resolution or in other scenarios.

Prompts combine camera color, background, viewpoint and lighting. Both models used matching prompts and integer seeds, 30 steps and guidance 7.5. Matching integer seeds do not imply matching noise across models. Severe artifacts covered obvious geometric distortions, collages or merged objects; studio lights alone were not counted. Camera color was marked incorrect when the camera was absent. Generation and ratings are subject to randomness and judgment.

## Recompute

`revealed.csv` contains per-image ratings and `development.jsonl` contains the inputs. Run:

```bash
python check_development_gate.py --development development.jsonl --scores revealed.csv
```

All 68 generated device files were checked against local backup SHA-256 hashes. This directory does not include every generated original. Recomputing ratings is not a substitute for inspecting the images.

The gate and prompts were recorded before review; file timestamps are mutable and no independent trusted timestamp establishes their order from the public files alone.
