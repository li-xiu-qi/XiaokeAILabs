# Ceramic mug generation: a held-out teacher-data control

On this small held-out set, an SD 1.5 student trained on SDXL-generated images scored higher in both AI blind-review rounds than the same base trained on its own generated images. Both LoRA runs used the same 128 training prompts and seeds, base weights, 400 steps, and hyperparameters. The difference is the source of the training images. This is synthetic output-data supervision, not direct matching of the teacher's denoising predictions.

The frozen test has 32 attribute combinations unseen in training and two seeds per combination, giving 64 requests. Each arm generated 64 images. The teacher produced 1024 px images, resized to 512 px with Pillow LANCZOS for review; the three student arms generated at 512 px. ID, prompt, seed, model and adapter, image size, decoding, and image hashes were checked before review.

| Arm | Round 1 successes / 64 | Unknown | Round 2 successes / 64 | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Untrained base | 20 | 2 | 24 | 5 |
| Student trained on self-generated images | 17 | 4 | 22 | 5 |
| Student trained on teacher-generated images | 32 | 2 | 32 | 3 |
| SDXL teacher pipeline | 36 | 2 | 35 | 4 |

Success requires exactly one recognizable ceramic mug, correct mug and background colors, and no severe artifact. Any unknown field makes the composite result unknown. On matched requests, teacher-data student versus self-data student produced 18 wins, 3 losses, 37 ties, and 6 unknowns in round 1; round 2 produced 15, 4, 38, and 7. Two seeds share one scene and must not be treated as independent scenes. The full per-scene and inter-reviewer results are in `mug-final-summary.json`.

Both rounds were AI reviews, not human reviews. The original plan assigned one AI agent to round 1; that agent reported insufficient capacity for all images. The work was split into sixteen 16-image batches, assigned to eight agents. A different AI agent completed round 2. The batch assignments and per-image reviewer identities are recorded in `mug-final-round1-assignments.json` and `mug-final-round1-reviewers.json`. Agents reported viewing the images; file hashes and complete ratings can be reproduced, but the files alone cannot independently prove visual inspection. Round 1 therefore includes variation across reviewers.

Pixel inspection found one completely black image in the teacher-data student arm, `test-007-1`. In round 1 its first three criteria were 0 and severe artifact was unknown; the frozen rule classifies any unknown composite as unknown. Round 2 classified it as failure. If the obviously mug-free image is treated as failure for round 1, teacher-data student successes remain 32 and unknowns decrease from 2 to 1. The original ratings are unchanged.

The results concern this product-image scenario and restricted attribute combinations. They do not establish generalization to other scenes or statistical significance. The teacher and student pipelines also differ in architecture, resolution, and scheduler. Only one training run per arm was performed; neither multiple training seeds nor human review was conducted. The locally hashed protocol has no independent trusted timestamp.

This package contains the test list, ratings, and mappings disclosed after review. `recompute.py` reproduces the scores and paired scene-level results. The mappings retain image hashes. [Download the 256 held-out original images and manifest](https://github.com/li-xiu-qi/XiaokeAILabs/releases/tag/evidence-mug-final-20261107); teacher originals are 1024 px and were resized to 512 px with LANCZOS for review. The black image is retained, both reviews were AI reviews, and thumbnail screening cannot rule out tiny text. Base-model weight files were not individually hashed, and the environment record is not a complete lock. The 50 GiB guard measures system MemAvailable, not dedicated GPU memory.
