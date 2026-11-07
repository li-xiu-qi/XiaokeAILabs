# A native-resolution teacher check

The preceding camera-product development check failed its teacher-advantage gate with SDXL generating at 512px. Because [Diffusers recommends 1024px for most SDXL checkpoints](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl), we used new development combinations and screened the models under their respective output settings: SDXL generated at 1024px and was resized to 512px using Pillow LANCZOS; the SD 1.5 baseline generated at 512px. Both used matched prompts and integer seeds, 30 steps and guidance 7.5, with their respective schedulers; the teacher used model CPU offload.

One AI reviewer inspected all 64 anonymized 512px images individually. Of 32 pairs, seven contained an unknown rating. Among the remaining 25 valid pairs, composite success was 5 for the teacher and 0 for the student. Composite success requires one recognizable camera, correct camera color and background, and no severe artifact.

| Individual metric | Student | Teacher |
| --- | ---: | ---: |
| One recognizable camera | 12/32, plus 4 unknown | 20/32, plus 4 unknown |
| Correct camera color | 7/32 | 19/32 |
| Correct background | 13/32 | 25/32 |
| Severe artifacts | 24/32 | 20/32 |

The stated gate requires at least 28 valid pairs, at least 20 teacher composite successes, and at least four more than the baseline. Only the third condition was met, **so subsequent training was not started**. Teacher counts were higher on individual metrics in this sample, but a single AI reviewer, a small set of related scenes, and unknown ratings limit the conclusion. This round used different development combinations from the previous round, so the change between rounds cannot be attributed to resolution. Model, output resolution, scheduler and resizing form the compared pipelines; their contributions are not isolated.

New development and held-out sets each contain 32 complete attribute combinations disjoint from the preceding training, development and test combinations. The held-out set remains unused; combination disjointness does not imply semantic independence. Original 1024px images and resized review images were retained separately, and the blinded mapping, prompts and hashes of all 64 review images were checked. This directory includes the development manifest, per-image ratings and recomputation inputs, not all originals or the held-out manifest; recomputing counts does not replace image review. Local file and conversation records support the timing of the gate and split, but there is no independent trusted timestamp and this public package cannot independently prove that sequence.

```bash
python check_development_gate.py --scores revealed.csv
```

The ordering of the gate, split, and ratings relies on local records without an independent trusted timestamp. The held-out manifest and images are not public, so readers cannot independently verify the cross-round split or image hashes.
