# Fine-Tuning a Sparse Retrieval Model (SPLADE)

[中文](README.md)

> **Technical guide:** [Learned sparse representations and SPLADE](../../docs/zh/retrieval/sparse/学习型稀疏检索与SPLADE.md)

---

## 1. Experiment card

| Item | Specification |
|---|---|
| **Purpose** | Build a full-vocabulary sparse-representation fine-tuning pipeline based on pretrained language models (BERT/MiniLM/BGE). Quantify the trade-off between term expansion and inverted-index sparsity constraints (FLOPS regularization), and test whether retrieval accuracy can exceed traditional BM25 while retaining inverted-index efficiency. |
| **Success criteria** | 1. Build a sparse encoder with an MLM-head projection and sequence max-pooling. <br>2. Jointly train ranking contrastive loss and FLOPS regularization; on a validation set, improve MRR@10 by more than 15% over standard BM25. <br>3. Keep the average number of activated nonzero terms per document (Active Non-Zero Length) between 120 and 180 to control inverted-list retrieval latency. |
| **Stop conditions** | 1. If FLOPS regularization causes representation collapse (fewer than 15 average nonzero terms, degrading to severely incomplete matching without expansion), stop immediately and adjust the $λ_{reg}$ weighting schedule. <br>2. If the number of nonzero terms exceeds 350 (degrading to a pseudo-dense vector) and the model has not converged for 500 consecutive steps, stop the experiment. |
| **Budget ceiling** | Single GPU (DGX Spark or a local 16 GB GPU): no more than 6 hours of training and 20,000 total steps. |

---

## 2. Core architecture and algorithm

### 2.1 Forward pass (`src/model.py`)

The code below is illustrative. The actual implementation loads a pretrained model with an MLM prediction head rather than initializing a separate linear vocabulary head.

The input sequence passes through a Transformer encoder. The MLM vocabulary projection maps it to full-vocabulary logits, followed by ReLU and a logarithmic transform; max pooling is then applied over the sequence dimension:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpladeEncoder(nn.Module):
    def __init__(self, transformer_backbone, vocab_size=30522):
        super().__init__()
        self.transformer = transformer_backbone
        # Vocabulary projection head (usually shares pretrained MLM weights)
        self.vocab_head = nn.Linear(transformer_backbone.config.hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Hidden-state sequence [B, L, D]
        hidden_states = outputs.last_hidden_state
        # Vocabulary projection [B, L, |V|]
        logits = self.vocab_head(hidden_states)

        # Sequence max-pooling and nonlinear mapping
        # w_j = max_i log(1 + ReLU(y_{i, j}))
        relu_log = torch.log(1.0 + F.relu(logits)) * attention_mask.unsqueeze(-1)
        sparse_vec, _ = torch.max(relu_log, dim=1) # [B, |V|]
        return sparse_vec
```

### 2.2 FLOPS regularization loss (`src/loss.py`)

To constrain average query cost in the inverted lists, the loss combines the ranking objective with a FLOPS penalty:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ranking}}(\mathbf{q}, \mathbf{d}^+, \mathbf{d}^-) + \lambda_{\text{reg}} \cdot \sum_{j=1}^{|V|} \left( \frac{1}{B} \sum_{k=1}^B |w_{k, j}| \right)^2$$

```python
class FlopsRegularizer(nn.Module):
    def forward(self, sparse_weights):
        # sparse_weights: [B, |V|]
        # Mean activation magnitude for each term in this batch
        mean_abs = torch.mean(torch.abs(sparse_weights), dim=0)
        # Penalize long inverted lists and accumulation on frequent terms
        flops_loss = torch.sum(mean_abs ** 2)
        return flops_loss
```

---

## 3. Directory layout and quick start

```text
sparse-retrieval/
├── README.md              # Experiment card and conventions
├── requirements.txt       # Pinned direct dependency; install PyTorch separately
├── src/
│   ├── model.py           # SPLADE model definition
│   ├── loss.py            # Ranking + FLOPS regularization loss
│   ├── dataset.py         # Triplet data pipeline
│   └── train.py           # End-to-end training script
```

Run commands from this experiment directory. Training data is not included; prepare a JSONL file with text fields `query`, `pos` and `neg` on each line. The output directory is set with `--output_dir`; if omitted, the script uses Git-ignored `.work/artifacts/`. At startup, the script claims a new or empty output directory and saves model weights and training metrics. It stops if the directory is nonempty, so specify a fresh directory for every run. A retained `.training-run` marker prevents a later run from overwriting the directory even if model loading or training fails.

### Environment

Training is recommended on an NVIDIA GPU with CUDA and a PyTorch build matching its driver. If CUDA is not detected, the script falls back to CPU; check `[Device]` in the startup log to confirm the actual device. The 16 GB in the experiment card is a planning budget, not a verified minimum VRAM requirement. Memory needs depend on batch size and sequence length. Reserve disk space for the model, training data and output. Full training on real data has not been validated from a clean installation.

Use an isolated Python environment. Linux example:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install CUDA-enabled PyTorch suited to the device and driver: https://pytorch.org/get-started/locally/
# requirements.txt omits torch so pip will not replace the installed CUDA build
python -m pip install -r requirements.txt
python -c "import torch, transformers; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'transformers:', transformers.__version__)"
python -m pip check
```

[`requirements.txt`](requirements.txt) pins the direct Transformers dependency to 5.16.1. Install PyTorch separately for the target device; pip resolves transitive dependencies. The training and evaluation scripts do not directly use datasets, scipy or NumPy, so the dependency list excludes them. `pip check` and import checks inspect the environment only and do not validate training results.

### Model and data

The example uses [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased). Transformers downloads the model and tokenizer on first use. Alternatively, set `--model_name_or_path` to a complete local model directory. Record the model revision used; the current command does not pin one. Training data is not included. Each JSONL line has this form:

```json
{"query":"What is SPLADE?","pos":"SPLADE learns sparse representations.","neg":"This passage concerns an unrelated topic."}
```

Check the sources and splits for training, validation and test data to avoid query or document leakage across splits; the current script does not perform these checks automatically. Built-in synthetic examples are in Chinese; to use them for a pipeline check, set the model to `bert-base-chinese`. The first model download requires network access; a local model does not.

The training script records loss and the number of activated terms. It does not automatically evaluate on a validation set or enforce the experiment card's stopping criteria. A separate evaluation entry point is provided below. The repository contains no real evaluation data, and the small check below cannot establish that the experiment goals were achieved.

### Train

```bash
python src/train.py \
  --model_name_or_path bert-base-uncased \
  --data_path /path/to/triplets.jsonl \
  --lambda_flops 0.0008 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir /path/to/new-output/splade-v1
```

To check the training pipeline, the following downloads a fixed revision of a tiny random model and explicitly uses four built-in synthetic examples. Neither the model nor the data can measure real retrieval quality. The output directory must be new or empty:

```bash
MODEL_DIR=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("hf-internal-testing/tiny-random-bert", revision="f171d7baecaf37b5da5a3616d8833b9969753535"))')
python src/train.py --model_name_or_path "$MODEL_DIR" --synthetic_data \
  --epochs 1 --batch_size 2 --max_len_q 16 --max_len_d 32 \
  --output_dir /path/to/new-output/splade-smoke
```

On NVIDIA GB10/Linux aarch64, this entry point creates a clean environment, installs pinned dependencies, downloads a fixed revision of a random tiny model and trains on the four bundled synthetic examples using CUDA. Its default PyTorch build is for this device; set `TORCH_SPEC` and `TORCH_INDEX_URL` for other devices. Choose new paths outside the repository for the environment, model cache and outputs:

```bash
export XL_RUN_ID="sparse-$(date -u +%Y%m%dT%H%M%SZ)"
export XL_DATA_DIR="/path/outside/repository/sparse-data/$XL_RUN_ID"
export XL_OUT_DIR="/path/outside/repository/sparse-runs/$XL_RUN_ID"
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/verify_clean_gpu.sh
```

On 2026-11-15, the script completed from a clean virtual environment on NVIDIA GB10/Linux aarch64. `pip check` passed with Python 3.12.3, PyTorch 2.15.0+cu132 and Transformers 5.16.1; CUDA training completed two optimizer updates and saved weights and metrics. The model was a random tiny BERT at a fixed revision, and the data were four bundled synthetic triplets. This verifies clean installation through GPU model saving; it does not establish retrieval quality or meet the experiment card's metric thresholds. See the [clean-environment GPU training record](evidence/clean-env-training-2026-11-15.json).

On 2026-11-13, two CUDA training steps completed on Linux ARM64 with an NVIDIA GB10, Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.22.2 and the cached model above; weights and metrics were written. BM25 and CUDA SPLADE evaluation also ran on a separate synthetic set of one query and two documents. `pip check` reported `nvidia-cusparselt-cu13` as unsupported on that platform, so this is not a dependency-consistent, locked environment. Loading the random model reports unused weights for other task heads.

On 2026-11-14, a copied and adjusted isolated environment on the same class of device ran one epoch and eight CUDA training steps on 32 SNLI triplets using `bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`. The environment passed `pip check` with Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.16.1 and Datasets 5.2.0.

For a functional evaluation, 20 sentence pairs rated at least 4 were selected from the first 100 STS-B validation examples; one sentence was used as a query and the other as a candidate document. Exact matches with training text were excluded, but near duplicates were not checked. With 20 candidates and only one labeled relevant document per query, BM25, the initial model and the trained model achieved MRR@10 of 0.975, 0.7375 and 0.85 respectively. The trained model averaged 447.35 active terms per document, above the experiment card's target. This artificial small pool has incomplete labels and unpinned dataset revisions; it checks the real-text training and evaluation paths, not general retrieval quality, goal attainment or installation from scratch.

If the data path does not exist, the script raises an error and does not switch to synthetic data automatically. After training, inspect `splade_weights.pt` and `training_metrics.json` in the output directory. The latter records the loss and activated-term count at each step; written artifacts alone do not establish that retrieval metrics meet the target.

### Independent retrieval evaluation

`src/evaluate.py` computes MRR@10 for BM25 and the model on the same candidate document set. Prepare three UTF-8 JSONL files:

- `corpus.jsonl`: one `{"doc_id":"d1","text":"document text"}` per line.
- `queries.jsonl`: one `{"query_id":"q1","text":"query text"}` per line.
- `qrels.jsonl`: one `{"query_id":"q1","doc_id":"d1","relevance":1}` per line. Values above zero indicate relevance; zero means not relevant. Every query needs at least one positive. IDs must be unique and refer to existing records.

For a real evaluation, obtain licensed data and labels, pin versions and splits, and check for leakage of queries, documents or near duplicates between training and evaluation. The four training examples are not a substitute for an independent test set. The following commands evaluate both methods using the same candidates and labels:

```bash
python src/evaluate.py --method bm25 --corpus /path/to/corpus.jsonl \
  --queries /path/to/queries.jsonl --qrels /path/to/qrels.jsonl --out /path/to/new-run/bm25.json
python src/evaluate.py --method splade --model /path/to/local-base-model \
  --checkpoint /path/to/trained/splade_weights.pt --device cuda \
  --corpus /path/to/corpus.jsonl --queries /path/to/queries.jsonl \
  --qrels /path/to/qrels.jsonl --out /path/to/new-run/splade.json
```

The model directory must match the base-model architecture and tokenizer used for training. Evaluation loads the local model and tokenizer; `--checkpoint` is the complete weight file produced by training. Omit it to evaluate the base model's initial representations. BM25 tokenizes lowercase Unicode words and individual Chinese characters, with `k1=1.2` and `b=0.75`. This is a defined baseline, not an optimal tokenization configuration for every language. Both methods score the same candidate set, include only documents with positive scores, and break score ties by ascending document ID. MRR@10 is the mean reciprocal rank of the first relevant document within the top 10. SPLADE also reports the average number of active document terms with weights above `1e-4`, matching the threshold in training logs.

Output includes input-file hashes, hashes of every file in the model directory, parameters and per-query rankings. Do not overwrite existing output. BM25 reads full text; SPLADE truncates queries and documents to 64 and 256 tokens by default, so the methods may receive different effective text lengths. Consider this truncation when comparing them. The implementation stores scores for every query-document pair and is suitable only for evaluation sets that fit in memory; it does not measure inverted-index latency. Verify model file versions, data sources and labels before running. Using the same candidate set does not cover every potentially relevant document.

To rerun the synthetic evaluation described above after training the tiny model:

```bash
EVAL_DIR=$(mktemp -d)
printf '%s\n' '{"doc_id":"d1","text":"SPLADE learns sparse representations"}' '{"doc_id":"d2","text":"The weather forecast predicts rain"}' > "$EVAL_DIR/corpus.jsonl"
printf '%s\n' '{"query_id":"q1","text":"sparse representations"}' > "$EVAL_DIR/queries.jsonl"
printf '%s\n' '{"query_id":"q1","doc_id":"d1","relevance":1}' > "$EVAL_DIR/qrels.jsonl"
python src/evaluate.py --method bm25 --corpus "$EVAL_DIR/corpus.jsonl" \
  --queries "$EVAL_DIR/queries.jsonl" --qrels "$EVAL_DIR/qrels.jsonl" --out "$EVAL_DIR/bm25.json"
python src/evaluate.py --method splade --model "$MODEL_DIR" \
  --checkpoint /path/to/new-output/splade-smoke/splade_weights.pt --device cuda \
  --corpus "$EVAL_DIR/corpus.jsonl" --queries "$EVAL_DIR/queries.jsonl" \
  --qrels "$EVAL_DIR/qrels.jsonl" --max-len-q 16 --max-len-d 32 --out "$EVAL_DIR/splade.json"
```

Replace the checkpoint path with the actual output path from the previous step. A single query cannot establish comparative retrieval quality.
