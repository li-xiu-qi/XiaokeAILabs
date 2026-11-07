# Camera follow-up, 2026-11-07

Hypothesis: the earlier post-hoc camera improvement recurs on new prompts and seeds. This is a follow-up, not a confirmatory preregistration independent of the earlier analysis.

Eight distinct prompts; two fixed seeds each. Arms: original student, 400-step LoRA student, SDXL teacher. Same 512px, 30 steps and guidance 7.5 as prior experiment; model default schedulers may differ. Seeds do not imply identical noise across model families.

Primary outcome: a single image depicts exactly one recognizable camera. Secondary outcomes: required camera color; specified background color; severe artifacts / unusable output. Blind randomized independent AI review, complete all 48 images, scores 0/1/uncertain. Record reviewer, instructions, every score and mapping; do not silently drop failures. Compare student pair per prompt and seed with changes both ways. Report 16 pairs but 8 unique prompts, correlated repeats, one reviewer; no significance claim or generalization to all objects. Teacher is contextual comparator, not ground truth.

No data-dependent retuning or prompt replacement. Preserve all outputs including failures; do not retrain. If generation fails, retain logs and mark missing; limit inference to successful outputs and failure count. Sufficiently label post-hoc selection of camera and limited sample. Record exact revisions, scheduler, steps, seed, model and LoRA hashes, hardware, dependency versions and execution logs.
