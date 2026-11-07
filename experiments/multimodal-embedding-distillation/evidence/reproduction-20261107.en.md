# GPU end-to-end reproduction of the public scripts

[简体中文](reproduction-20261107.md) | English

On 2026-11-07, we reran public data preparation, teacher encoding and both student training conditions on a DGX Spark GB10. For the fixed 100 test queries, each of the baseline, supervised and distilled student rankings matched its [previously published record](student-test-ranks.json) item by item. Teacher R@1, R@5 and MRR were 0.94, 1.00 and 0.965833, matching the original aggregate. This is one repeat on the same device, cached model revisions and data; it does not establish portability across devices or versions.

## Conditions and procedure

The scripts correspond to commit `f1aa627`. The environment was Linux, Python 3.12, PyTorch 2.14.0+cu130, Transformers 5.16.1 and Sentence Transformers 6.0.1. The teacher revision was `5c54692f22e186fc12ea38a9193e3ff9e1cda2a3` and the student revision `cc4c733ed997dbee4ac70ccffb911e70c9c24b93`. The annotation SHA-256 was `395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137`. The new sample manifest SHA-256 was `c9aaf627f76439c479fcdc2b03595be3bdc62c829a7baa5651e982af4e83dfec`, matching the original.

In a fresh work directory, we ran `prepare.py`, `teacher.py`, `train.py` and `verify.py` in the order and with the arguments shown in the main README. Preparation selected 512 / 128 / 100 images; all 740 decoded and no selected images were byte-identical. The scheduler observed exit code 0 for teacher encoding and student training; separate exit codes were not preserved in the logs. `verify.py` recalculates metrics from the historical public ranks; separately, all three newly generated arrays of 100 test ranks matched those records item by item. Result JSON includes run-specific timing and has a different SHA-256; it was not byte-identical.

An independent process monitored system available memory, terminating the experiment process group if it fell below 50 GiB. The minimum recorded availability was 87.62 GiB for the teacher stage and 88.78 GiB for the student stage. The teacher stage took about 88.03 seconds including model loading; the guarded intervals were approximately 93 and 44 seconds and are not pure model-compute times. Other hardware, concurrent loads and datasets were not evaluated.

Flickr30k images and text remain subject to the original publisher's terms; the Jina teacher model card specifies CC BY-NC 4.0. This record distributes no images, captions, teacher vectors or model weights. See the [experiment README](../README.en.md) for provenance and licensing. Visual near-duplicates, pretraining overlap and unlabeled relevant images remain unresolved.