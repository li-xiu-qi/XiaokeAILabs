[中文](README.md)

# Embedding Distillation for Scientific Literature Retrieval

This directory preserves the historical experiment and reproduction code. The scripts now include an independent teacher–student comparison on a development set: training continues only if the teacher's nDCG@10 is at least 0.05 higher than the student's and the lower bound of the paired, per-query bootstrap 95% confidence interval is above zero. The 0.05 threshold is preset for this experiment, not a universal standard for distillation. The historical run did not use this gate; interpret its results under the original run conditions.

This directory contains experiment code, configuration, run instructions, and result evidence. See the [Embedding Distillation walkthrough](../../docs/zh/retrieval/embedding/Embedding蒸馏.md) for the purpose, principles, and results. Run the commands below from this directory.

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

## Reproduction and Historical Results

Run `python scripts/verify.py` to recompute the table and paired bootstrap from the supplied per-query metrics. The evidence contains query IDs and metrics, but not raw text, models, or datasets. This recomputation checks aggregation and confidence intervals; it cannot independently validate the original model inference or labels. The published scripts were parameterized from device-side scripts and have completed one real-data run using the steps below; they are not a one-click workflow.

Prepare a Linux CUDA environment with Python 3.11 or later, PyTorch, NumPy, and Sentence Transformers. The data-preparation script uses only the Python standard library. Its candidate-pool configuration matches the reranking experiment; the generated `pools.json` determines training positives and BM25 candidates. Obtain datasets and models from their original sources under their respective licenses; they are not included in this repository. The BEIR dataset card states CC BY-SA 4.0. The original SciFact labels and corpus have different licenses; consult the [original license](https://github.com/allenai/scifact/blob/master/LICENSE.md) before use.

Create an isolated Python environment, install a PyTorch build that matches the device, then install the other direct dependencies:

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# Install PyTorch matching the system, architecture, and GPU from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Training requires an available CUDA GPU'"
python -c "import torch; assert torch.cuda.is_bf16_supported(), 'The teacher requires BF16 support'"
python -c "import numpy, sentence_transformers, huggingface_hub; print('Dependencies imported successfully')"
```

On 2026-11-14, `pip check` passed in a separately copied and adjusted Python 3.12.3 environment. Preparing the original SciFact data produced 390 training, 89 development, and 300 test queries; the SHA-256 of `pools.json` was `e7cac48e7083bdd948e2773136c1df9672b4ca8f787ee5b774697b32d689c5f8`. The pinned student model was prepared. The copied environment is not a verified clean installation.

A formal run completed on the same day. Across 89 development queries, the teacher exceeded the initial student by 0.0704 nDCG@10, with a paired bootstrap 95% interval of [0.0250, 0.1192], passing the preset gate. The supervised and distilled students each trained for 128 steps. On 300 test queries, nDCG@10 was 0.6654 for baseline, 0.6702 supervised, 0.6712 distilled, and 0.7341 teacher. Distilled minus supervised was 0.0010, with a paired bootstrap 95% interval of [-0.0180, 0.0208], insufficient to establish a stable improvement. This result depends on the fixed candidate pool, one training seed, and this environment. Test queries were separate, but some test-positive documents also appeared among training candidates, so this is not an evaluation on an entirely new corpus.

On the same day, a CUDA smoke run used the pinned Qwen3-Embedding-8B teacher and MiniLM student on real data: the teacher encoded and scored 4 queries and their 68 candidate documents, and the student scored candidates for 2 development queries and calculated the gate. The smoke run stopped before training and test evaluation as designed. Each teacher weight file passed SHA-256 verification. Two development queries cannot establish a teacher advantage or experimental effectiveness.

[`requirements.txt`](requirements.txt) pins the direct dependencies; install PyTorch separately with a build compatible with the target device. The data preparation commands below also require `curl`, `unzip`, `tar`, and `sha256sum`, and sufficient disk space for data, models, and outputs.

To check the GPU code paths without downloading a model or dataset, run:

```bash
python scripts/synthetic_gpu_smoke.py
```

This check creates a random tiny model and synthetic data, then runs teacher scoring and the student-stage development gate. The student `--smoke` stage stops before training. The script then calls the training function directly for two supervised and two distilled steps and checks that parameters change. This does not show that the formal experiment passes its gate or reproduce its results on real data. On 2026-11-13, the check completed on Linux with an NVIDIA GB10, Python 3.12.3, PyTorch 2.15.0+cu132, NumPy 2.5.4, Sentence Transformers 6.3.0, Transformers 5.22.2, and huggingface_hub 2.4.0. That environment's `pip check` reported `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`, so this is limited runtime evidence, not a complete environment verified from scratch.

To install dependencies in a fresh environment and run the same GPU smoke:

```bash
export XL_REQUIRE_EXTERNAL_ENV=1
export XL_RUN_ID="$(date +%Y%m%d-%H%M%S)"
export XL_DATA_DIR="$HOME/xl-data"
export XL_OUT_DIR="$HOME/xl-output"
# Set only if an inherited proxy blocks downloads: export XL_CLEAR_PROXY=1
bash scripts/clean_env_gpu_smoke.sh
```

On 2026-11-15, a fresh environment on Linux ARM64 / NVIDIA GB10 installed the pinned dependencies; `pip check`, CUDA, and BF16 checks passed. The synthetic GPU smoke completed, updating parameters for two steps under each of the supervised and distilled losses. See [`evidence/clean-env-gpu-smoke-2026-11-15.json`](evidence/clean-env-gpu-smoke-2026-11-15.json). This run did not download real models or data and did not run the real-data teacher gate, full distillation, or test evaluation. Full reproduction from an empty environment remains unverified.

Run the following from this experiment directory. Set `WORK` to a directory outside the repository. Data, models, and outputs will be written there.

```bash
export WORK=/path/to/new-working-directory
mkdir -p "$WORK/data"
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o "$WORK/data/scifact.zip"
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o "$WORK/data/scifact-original.tar.gz"
(cd "$WORK/data" && sha256sum -c <<'EOF'
536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165  scifact.zip
11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be  scifact-original.tar.gz
EOF
)
unzip -q "$WORK/data/scifact.zip" -d "$WORK/data"
tar -xOzf "$WORK/data/scifact-original.tar.gz" data/claims_train.jsonl > "$WORK/data/claims_train.jsonl"
python scripts/prepare_scifact.py --data "$WORK/data/scifact" \
  --original-claims "$WORK/data/claims_train.jsonl" --config configs/pool-config.json --out "$WORK/prepared"
DATA="$WORK/data/scifact"
POOLS="$WORK/prepared/pools.json"
```

Download both models at the recorded revisions, for example:

```bash
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-8B', revision='1d8ad4ca9b3dd8059ad90a75d4983776a23d44af', local_dir=os.path.join(os.environ['WORK'], 'models/teacher'))"
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir=os.path.join(os.environ['WORK'], 'models/student'))"
```

Choose `--min-teacher-gap` in advance and use the same value in both stages. For this experiment, it must be at least 0.05:

```bash
python scripts/train.py --stage teacher --data "$DATA" --pools "$POOLS" --out "$WORK" --teacher "$WORK/models/teacher" --student "$WORK/models/student" --min-teacher-gap 0.05
python scripts/train.py --stage student --data "$DATA" --pools "$POOLS" --out "$WORK" --teacher "$WORK/models/teacher" --student "$WORK/models/student" --min-teacher-gap 0.05
```

The teacher stage computes scores only for training candidates and the development set. The student stage first compares teacher and student on the same development set and candidate pool. If the gate passes, both the supervised and distilled runs are trained and saved, after which test-set scores are computed. `--smoke` checks only the small development-set workflow; it does not train or evaluate the test set. The scripts require Linux `/proc/meminfo` and about 50 GiB of available system memory at minimum; the full experiment uses CUDA. The real-data gate, training, and test workflow previously completed in a copied and adjusted environment. Fresh-environment validation currently covers only the synthetic GPU smoke.

In the historical workflow, the teacher stage generated scores, and the student stage evaluated the original model, supervised run, and distilled run. It checks SHA-256 values for the corpus, queries, and candidate pool, as well as model paths and teacher-output integrity. A model's content changing at the same path is not detected, so verify model revisions and weights yourself. Hardware, dependency versions, and randomness can affect results.
