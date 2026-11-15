[中文](README.md)

# Fine-Tuning Chinese Text Embeddings for Retrieval

The [full experiment walkthrough](../../docs/zh/retrieval/embedding/Embedding监督微调.md) explains the problem, principles, and results. This page provides the code and reproduction entry point. Run from a working copy; keep data and outputs outside the public repository.

## Environment Setup and Quick Start

The commands below use Linux Bash and are run from this experiment directory. Formal training requires an available CUDA GPU. Memory use depends on the model, batch size, and text length; the batch size must be at least 2. Create an isolated environment outside the repository, choose a PyTorch build compatible with the system architecture, driver, and GPU using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), and install the script's direct dependencies:

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# Install PyTorch matching the system, architecture, and GPU from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Training requires an available CUDA GPU'"
python -c "import transformers, sentence_transformers, tqdm; print('Dependencies imported successfully')"
```

[`requirements.txt`](requirements.txt) and [`requirements-eval.txt`](requirements-eval.txt) pin the direct dependencies used in the latest validation; install PyTorch separately with a build compatible with the target GPU. On 2026-11-15, a fresh Linux ARM64 / NVIDIA GB10 virtual environment completed GPU fine-tuning and evaluation on real data; see the run record below. The first use of the default model downloads [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5); you can instead download and verify it in advance, then use a local path. Network access and disk space for the model, data, and checkpoints are required. The script enables `trust_remote_code=True`; inspect the selected model's code and license before loading it.

To run the validated workflow from an empty environment, run this script from the experiment directory. It creates a fresh virtual environment, checks CUDA and dependencies, verifies the data files, trains, and evaluates. Use a new `XL_RUN_ID` for each run, and keep data and output directories outside the repository. The default PyTorch build is for the tested GB10; for other devices set a compatible `TORCH_INDEX_URL` and `TORCH_PACKAGE`.

```bash
export XL_REQUIRE_EXTERNAL_ENV=1
export XL_RUN_ID="$(date +%Y%m%d-%H%M%S)"
export XL_DATA_DIR="$HOME/xl-data"
export XL_OUT_DIR="$HOME/xl-output"
# If needed for network access: export HF_ENDPOINT=https://hf-mirror.com
# If the inherited proxy is broken, also set: export XL_CLEAR_PROXY=1
bash scripts/clean_env_gpu_smoke.sh
```

By default, the script downloads T2Retrieval from its pinned revision. If that revision is already available, set `T2RETRIEVAL_DATA_DIR` to its directory; the script still verifies every file's SHA-256. It downloads the model from a pinned revision. This small development-set workflow is not a substitute for full training or a formal test.

Set data and output paths outside the repository. Point `--output_dir` to a new, nonexistent directory or an empty directory. Existing files, nonempty directories, or training data located inside the output directory are rejected to protect existing results. The following runs the bundled demo data. If the default `demo_data.jsonl` is missing, the script creates it. To avoid writing into the repository, first change to an external working directory and use the script's absolute path:

```bash
REPO=/path/to/XiaokeAILabs
WORK=/path/to/new-working-directory
mkdir -p "$WORK"
cd "$WORK"
python "$REPO/experiments/embedding-finetuning/scripts/train_embedding.py" --output_dir "$WORK/model"
```

From the experiment directory, run `python -m unittest discover -s tests -v` to check output-directory protection. The tests neither download a model nor train.

The default model is `BAAI/bge-base-zh-v1.5`; the first run may download it. You can also pass a local model path through `--model_name_or_path`. The script selects CUDA automatically, falling back to CPU if unavailable. It trains for 3 epochs by default with a batch size of 4 and saves a checkpoint each epoch. The command above writes the final model to `$WORK/model/final_model`.

The default data file is `demo_data.jsonl`; if missing, the script creates the built-in demo data. A missing custom data path causes an error to prevent demo data from being mistaken for real training data. The batch and dataset must each contain at least two records. If the last batch has only one record, the script drops it to avoid a zero-loss update without a negative example.

On 2026-11-13, a CUDA smoke run completed on a DGX Spark. It used an existing Python 3.12.3, PyTorch 2.15.0+cu132, and Transformers 5.22.2 environment, with Sentence Transformers 6.3.0, scikit-learn 1.9.1, Pillow 13.0.0, joblib 1.6.0, threadpoolctl 3.8.0, cloudpickle, and narwhals isolated for that run. Using tiny-random-bert and four built-in synthetic samples, the run used batch size 4 and one epoch with one batch, exited with code 0, and saved a checkpoint and `final_model`. This was not an environment installed from this directory's requirements, nor a test with the default model, real data, or model quality. The base environment's `pip check` reported `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`, so the dependency check cannot be considered fully successful.

On 2026-11-14, a copied and adjusted isolated environment ran one CUDA epoch with batch size 4 on the four built-in samples, using the default `BAAI/bge-base-zh-v1.5` model at revision `f03589ceff5aac7111bd60cfc7d497ca17ecac65`. It exited with code 0 and saved a checkpoint and final model. The environment used Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.16.1, and Sentence Transformers 6.0.1; `pip check` passed. The model was obtained through a mirror. This was not a fresh installation, and the single training batch did not test real data or independent retrieval quality.

## Small-scale Run on Real Chinese Retrieval Data

On 2026-11-14, the copied and adjusted isolated environment above ran on a subset of [T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval): 256 training queries and 100 test queries selected from its original `dev` split, with 409 candidate documents. Query texts and training-positive versus test-relevant document IDs did not overlap. The default model completed 32 CUDA training steps in one epoch. Hit@10 was 0.99 both before and after training; MRR changed from 0.9753 to 0.9736.

On 2026-11-15, a fresh virtual environment on Linux ARM64 / NVIDIA GB10 installed the pinned dependencies from scratch; `pip check` passed. Using the pinned BGE model and T2Retrieval revisions, it completed one 32-step CUDA epoch. Hit@10 on 100 test queries was 0.99 before and after training; MRR changed from 0.9753 to 0.9805. The dataset came from an existing copy whose three Parquet SHA-256 values matched the ones listed below; the model was downloaded from a mirror during this run. See [`evidence/clean-env-gpu-smoke-2026-11-15.json`](evidence/clean-env-gpu-smoke-2026-11-15.json) for the run summary.

Both runs used a small candidate pool, a near-ceiling baseline, and one random seed. The training script does not fix its random seed, and semantic near-duplicates or pretraining overlap have not been ruled out. These scores do not establish a generalizable improvement, and they are not official test-set results.

Install `pandas`, `pyarrow`, and the Hugging Face download tool in the isolated environment. The three fixed-revision files `corpus/dev-00000-of-00001.parquet`, `data/dev-00000-of-00001.parquet`, and `queries/dev-00000-of-00001.parquet` have SHA-256 values `8f0400530c6ba664e580e2ae7bb5ed41a908b8abc17c3b3934f42cc65992ac1e`, `c4c6ffd715faf151e0832a26dfcf2d7ec1edc17d9620a9aa99e721cc2436d7d2`, and `186b4d0a13cc4d8339eccfc6d465d1b2d4d1f6572defa8d39f3c8da3b66b0210`, respectively. Run from this experiment directory, with a new `WORK` outside the repository:

```bash
python -m pip install -r requirements-eval.txt
python -m pip check
WORK=/path/to/new-work-directory
mkdir -p "$WORK"
hf download mteb/T2Retrieval --repo-type dataset --revision 921dd3af6e78d1ae7ee0368aa8d7eaee02c8f08e --include '*.parquet' --local-dir "$WORK/dataset"
hf download BAAI/bge-base-zh-v1.5 --revision f03589ceff5aac7111bd60cfc7d497ca17ecac65 --local-dir "$WORK/base-model"
sha256sum "$WORK/dataset"/*/*.parquet
python scripts/t2retrieval_subset.py prepare --source "$WORK/dataset" --output "$WORK/subset"
python scripts/t2retrieval_subset.py evaluate --data "$WORK/subset/eval.json" --model "$WORK/base-model" --output "$WORK/baseline.json"
python scripts/train_embedding.py --model_name_or_path "$WORK/base-model" --train_dataset "$WORK/subset/train.jsonl" --output_dir "$WORK/model" --epochs 1 --batch_size 8
python scripts/t2retrieval_subset.py evaluate --data "$WORK/subset/eval.json" --model "$WORK/model/checkpoint-epoch-1" --output "$WORK/finetuned.json"
```

The split uses seed 42, but the training script does not fix its random seed. The pool contains all labeled positives for the selected test queries plus 200 random documents. Hit@10 is the fraction of queries with at least one labeled relevant document in the top ten.

## Using Your Own Data

Prepare a UTF-8 JSONL file with one JSON object per line, containing `query` and `pos_doc`:

```jsonl
{"query":"如何学习 Python？","pos_doc":"学习 Python 可以从安装环境、变量和循环等基础开始。"}
{"query":"什么是机器学习？","pos_doc":"机器学习研究如何让计算机从数据中学习。"}
```

Pair each query with a document passage that answers it. Cover realistic queries, domain-specific phrasing, and easily confused candidates. Split training and evaluation data in advance, and check for duplicate or highly similar queries and documents to reduce data leakage. Documents and queries are truncated according to `--passage_max_len` and `--query_max_len`, respectively; verify during data preparation that important content is not truncated.

During training, the model separately encodes each query and its positive passage into vectors. The loss updates the model using similarities among vectors in the same batch. At retrieval time, encode candidate passages with the trained model and build a vector index. Encode each incoming query and compare it with the candidate vectors by similarity to return the highest-scoring passages. The model outputs vectors; ranking comes from the subsequent similarity calculation, not from the model directly generating an answer.

```bash
python "$REPO/experiments/embedding-finetuning/scripts/train_embedding.py" \
  --model_name_or_path BAAI/bge-base-zh-v1.5 \
  --train_dataset path/to/train.jsonl \
  --output_dir path/to/output \
  --epochs 3 \
  --batch_size 4
```

Other configurable parameters are `--lr`, `--query_max_len`, and `--passage_max_len`; see [`train_embedding.py`](scripts/train_embedding.py) for all defaults. The script does not support explicit hard negatives, validation-set evaluation, or automatic best-checkpoint selection; implement these separately if needed.

## How to Tell Whether Fine-Tuning Helps

On an independent evaluation set, test the base and fine-tuned models with the same document corpus, chunking, queries, and retrieval configuration. Compare task-relevant metrics such as Recall@K, MRR, or NDCG, and inspect real failure cases: whether relevant documents enter the candidate set and where they rank, whether domain-specific queries improve, and whether queries that previously worked well regress. Choose metrics and K values based on how many candidate passages the downstream application actually reads.

The script's `acc` is the ranking accuracy of positive examples within the current training batch, and `loss` is also measured on training batches. Neither is an independent retrieval evaluation, so neither supports claims about generalization or business impact. The script does not perform the evaluation above; confirm any benefit with a separate comparison.

## Scope of the Current Example

The built-in dataset has four demo samples from different topics. With the default batch size of 4, each epoch has only one batch. It demonstrates the code path for loading data, computing contrastive loss, updating parameters, and saving a model. It is not a sufficiently broad training set and includes no independent validation, baseline comparison, or real-world retrieval result. Before deployment, add suitable data, evaluation, and a comparative effectiveness study, then decide whether to use the fine-tuned model.
