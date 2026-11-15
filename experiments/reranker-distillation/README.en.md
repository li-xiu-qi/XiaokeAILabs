# Can a Small Reranker Learn a Teacher's Ranking Preferences?

[中文](README.md)

Before starting distillation, use `--min-teacher-gap` to declare the minimum absolute advantage of the teacher over the student on development-set nDCG@10. The paired query-bootstrap 95% confidence interval must also have a lower bound above zero. Training does not start if the threshold is missing or not met, or if smoke-test data is used. Interpret historical results according to the gate in effect when they were produced.

This directory contains experiment code, configuration, run instructions, and result evidence. See the [reranker distillation guide](../../docs/zh/retrieval/reranking/重排序蒸馏.md) for the experiment's purpose, method, and results. Run the commands below from a working directory outside the repository.

[scripts](scripts/) · [assets](assets/) · [evidence](evidence/)

In one experiment on a fixed SciFact candidate set, the Qwen3-Reranker-8B teacher achieved test nDCG@10 of 0.7635. The same TinyBERT student scored 0.6336 in the supervised group and 0.6356 in the distillation group. The confidence interval for the difference between distillation and supervision includes zero, so these results do not establish an additional benefit from distillation. The [per-query evidence for this run](evidence/qwen3-8b/) supports recalculating the statistics. Both the development and test sets had been used in earlier experiments; this was not a fresh blind test.

## Reproduction and historical records

### Five training-signal comparison

With fixed candidates and a budget of 180 training steps, the comparison uses original labels, the teacher's hard ranking, the teacher's soft scores, and mixtures of original labels with the hard ranking or soft scores. Each group has three training random seeds; training-sample selection uses the same fixed seed. See the [results guide](../../docs/zh/retrieval/reranking/重排序蒸馏.md#后续对照教师硬排序和软分数有什么不同) for the data and limitations. [Per-query metrics, per-candidate scores, training losses, and configuration](evidence/hard-soft-five-arms/) are available for inspection. The test set had previously been exposed, so this is an exploratory comparison.

The five-group comparison requires completing the setup below and scoring with the Qwen teacher before running the five-group command.

Linux, CUDA-enabled PyTorch, Transformers, NumPy, and Hugging Face Hub are required. The Qwen 8B teacher in BF16 needs roughly 16 GB for weights alone, plus memory for the runtime and batches; actual requirements vary by device and configuration. In an isolated Python environment, first install a CUDA-enabled PyTorch build compatible with the device, then install the remaining dependencies:

```bash
REPO=/path/to/XiaokeAILabs
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# Install PyTorch matching the system, architecture, and GPU from https://pytorch.org/get-started/locally/
python -m pip install -r "$REPO/experiments/reranker-distillation/requirements.txt"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Training requires an available CUDA GPU'"
python -c "import numpy, transformers, accelerate, huggingface_hub; print('Dependencies imported successfully')"
hf --help
```

Replace `REPO` with the absolute path to the repository. The [dependency list](requirements.txt) pins Transformers to historical version 4.56.2 for the Qwen teacher and includes Accelerate, required to load the Qwen teacher with `device_map`. Other dependencies are not pinned, so this is not a complete lock file and has not been verified from a clean environment. `hf download` uses the Hugging Face Hub CLI. Follow the data preparation steps in the [reranker fine-tuning tutorial](../reranker-finetuning/README.en.md#prepare-data-check-the-pipeline-and-train); prepare `scifact/` and `prepared/pools.json`, but do not run its training. The SHA-256 of this candidate file should be `e7cac48e7083bdd948e2773136c1df9672b4ca8f787ee5b774697b32d689c5f8`. Results from a different version or candidate pool are not directly comparable with the numbers here.

Replace `REPO`, `WORK`, and `DATA_ROOT` with absolute paths to the repository, a working directory outside it, and the prepared data. Models and results are saved under `WORK`:

```bash
WORK=/path/to/private-reranker-work
DATA_ROOT=/path/to/prepared-reranker-data  # contains scifact/ and prepared/pools.json
mkdir -p "$WORK"
cd "$WORK"
hf download cross-encoder/ms-marco-MiniLM-L12-v2 --revision 7b0235231ca2674cb8ca8f022859a6eba2b1c968 --local-dir "$WORK/teacher"
hf download cross-encoder/ms-marco-TinyBERT-L2-v2 --revision 81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc --local-dir "$WORK/student"
python "$REPO/experiments/reranker-distillation/scripts/run.py" \
  --data-root "$DATA_ROOT" --teacher "$WORK/teacher" \
  --student "$WORK/student" --out "$WORK/run" --min-teacher-gap 0.05
```

On 2026-11-14, the fixed MiniLM teacher, TinyBERT student, and SciFact candidate pool above were used for real-data teacher scoring and the development gate on Linux ARM64 and NVIDIA GB10. On 89 development queries, nDCG@10 was 0.6370 for the teacher and 0.6138 for the student. The 0.0232 advantage was below the prespecified 0.05, and the lower bound of the paired bootstrap 95% interval was -0.0071. The program therefore stopped as designed without training or testing. Model weight hashes were checked. The separate copied and adjusted environment passed `pip check` with Python 3.12.3, PyTorch 2.15.0+cu132, and Transformers 4.56.2; it was not installed from scratch. This does not reproduce the Qwen teacher run below.

To use the Qwen teacher from this run, download it as follows and use it in place of the earlier teacher. Keep the TinyBERT student downloaded above:

```bash
hf download Qwen/Qwen3-Reranker-8B --revision 77d193c791ed757ca307ee72715aa132723da912 --local-dir "$WORK/qwen-teacher"
python "$REPO/experiments/reranker-distillation/scripts/run.py" \
  --data-root "$DATA_ROOT" --teacher "$WORK/qwen-teacher" \
  --teacher-type qwen3 --student "$WORK/student" --out "$WORK/qwen-run" --min-teacher-gap 0.05
```

On 2026-11-14, the public entry point ran on Linux ARM64 and NVIDIA GB10 with the fixed Qwen teacher and TinyBERT student; model weight hashes were checked. On 89 development queries, teacher nDCG@10 was 0.7363 and student nDCG@10 was 0.6138. The 0.1225 advantage exceeded the prespecified 0.05 and the paired bootstrap lower bound was 0.0749. After 180 steps each of supervised and distillation training, final evaluation on 300 test queries gave nDCG@10 of 0.7618 for the teacher, 0.6267 for the original student, 0.6336 for supervision, and 0.6356 for distillation. The paired difference of 0.0019 has a 95% bootstrap interval of [-0.0028, 0.0068], so this does not establish a stable improvement. The test set was used in earlier experiments; query IDs are disjoint across splits, but some test-positive documents occur in training candidates. Only one seed was run. The separate copied environment passed `pip check` with Python 3.12.3, PyTorch 2.15.0+cu132, and Transformers 4.56.2; it was not installed from scratch.

The Qwen teacher defaults to a maximum of 512 tokens and batch size 4; adjust these with `--qwen-max-length` and `--qwen-batch-size`. The student maximum length is 192 tokens. The teacher uses the difference between the raw `yes` and `no` logits and scores candidates sampled during the actual training run. This run used DGX Spark, a BF16 teacher, an FP32 student, PyTorch 2.14.0+cu130, and Transformers 4.56.2. The new public entry point was adapted from the device-side script and has passed CPU simulation, an old-teacher smoke check, and the synthetic GPU check below; the real-data Qwen 8B end-to-end rerun above has now completed. It computes test results after both training groups finish, whereas the original run generated Qwen test scores before training; those scores were not used to select the teacher or train. Changes to configuration, truncation, or execution order require reporting results again.

The default `--stage all` runs teacher scoring and student experiments in sequence. You can instead run `--stage teacher` first and then `--stage student` with the same arguments to reuse teacher scores. The script checks SHA-256 hashes for candidate pools, corpus, and query files, model paths, and teacher-output completeness. You must still verify model directory contents yourself; a path alone does not establish the weight revision. For new runs, declare the teacher's minimum absolute advantage on the development set in advance and require the lower bound of the paired per-query bootstrap 95% interval to exceed zero. Otherwise training stops. The example threshold of 0.05 is specific to this experiment, not a universal threshold. Once the gate passes and training completes, output includes per-query metrics, per-candidate scores for each model, training loss, parameter listings, and model weights. If the gate fails, only completed teacher and development-set evaluations and the stop reason are retained. Hardware, software versions, and nondeterministic computation can affect results.

### Rerun the five-group comparison using existing teacher scores

The following entry point reuses existing teacher scores and rechecks the teacher's development-set advantage before training. Use the same training and development candidate scores as this run. They can be generated by running the Qwen teacher-scoring stage above, or by producing scores with the same configuration. The teacher-score file hash is recorded, but a hash alone does not prove which model produced the scores.

```bash
python "$REPO/experiments/reranker-distillation/scripts/compare_signals.py" \
  --data-root "$DATA_ROOT" --student "$WORK/student" \
  --teacher-scores "$WORK/qwen-run/teacher-scores.json" \
  --out "$WORK/five-arm-run" --min-teacher-gap 0.05
```

The five-group entry point requires `--min-teacher-gap` to be at least 0.05; this restriction applies only to that entry point. On Linux devices, append `--min-available-gib 50` to stop if available memory is below 50 GiB. Do not use an existing result directory. The teacher, student, and candidate pool must match the versions described in this section. The public entry point completed a five-group rerun on the same DGX Spark using existing teacher scores; all 15 test metrics matched the original run to nine decimal places. Generating teacher scores again from Qwen weights and running end to end has not been verified.

For either two-stage command, append `--device cpu --smoke` to check the scoring path: each split uses only two queries, so use a **separate output directory** and do not overwrite formal results. Smoke data is insufficient to assess the teacher advantage and will not proceed to training; it is not an effectiveness comparison. Formal runs should use the specified model versions and full data.

To check the GPU without downloading models or data, use the randomly initialized tiny BERT and synthetic data:

```bash
python "$REPO/experiments/reranker-distillation/scripts/smoke_gpu.py" \
  --out "$WORK/synthetic-gpu-check"
```

The output directory must be empty. This check first runs the public entry point and confirms that smoke data stops at the teacher-advantage gate. It then separately calls the teacher scoring and student training functions, runs two steps each for supervision and distillation, and checks that the weights changed. It does not pass the formal gate or reproduce the real data or Qwen teacher. One check used Linux ARM64, NVIDIA GB10, Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 4.56.2, and NumPy 2.5.4. It used an isolated dependency overlay with an existing PyTorch environment; the overlay's Hugging Face Hub conflicts with `datasets` in that base environment, so this is not a clean, fully verified environment installation.

[The earlier MiniLM teacher experiment](evidence/) and [the Qwen teacher experiment](evidence/qwen3-8b/) preserve per-query metrics separately. Recompute the paired interval for the Qwen distillation and supervision results with `python "$REPO/experiments/reranker-distillation/scripts/recompute-qwen3.py"`. These records do not contain per-candidate student scores, so the metrics cannot be independently recalculated from rankings. The BEIR data card specifies CC BY-SA 4.0; the original SciFact annotations and abstract corpus have their own licenses. See the [original license](https://github.com/allenai/scifact/blob/master/LICENSE.md). Public evidence does not include original paper text or model weights.
