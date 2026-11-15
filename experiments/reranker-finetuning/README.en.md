# Fine-Tuning a Reranker on SciFact: Move Better Results Up

[中文](README.md)

The [full experiment guide](../../docs/zh/retrieval/reranking/重排序监督微调.md) explains the problem, method and results. This page covers code and reproduction. Run from a working copy; keep data and outputs outside the public repository.

## Recompute the saved results offline

Python's standard library is sufficient. The script reranks 9,000 saved candidate records and checks per-query and aggregate nDCG@10 and MRR@10, as well as candidate Hit@30 and Recall@30. BM25 uses the saved candidate order; ties in model scores preserve that order.

```bash
python scripts/verify_ranking.py
```

## Prepare data, check the pipeline, and train

The recorded environment was Linux aarch64, Python 3.12.3, CUDA 13.0, PyTorch 2.14.0+cu130, Transformers 5.17.0 and NumPy 2.5.3. Data preparation uses `hashlib.file_digest` and therefore requires Python 3.11 or newer. The training script requires a CUDA GPU with native BF16 support and checks for at least 24 GiB of available system memory; GPU memory needs depend on the configuration. Data preparation uses only the Python standard library. Run the following in this experiment directory. Create an isolated environment, then use the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to choose a CUDA build matching the operating system, architecture and GPU. The model is downloaded on first use using the configured model name and revision; alternatively, pass `--model-path` to a prepared local model. Downloads and training require network access and disk space outside the repository.

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# Install a PyTorch build matching the OS, architecture and GPU: https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Training requires an available CUDA GPU'"
python -c "import torch; assert torch.cuda.is_bf16_supported(including_emulation=False), 'GPU lacks native BF16 support'"
```

Next download and verify the data, construct training examples, run a smoke check, then train. BEIR supplies the retrieval corpus and relevance labels; original SciFact supplies sentence-level evidence used to select positive training examples. Set `WORK` to a location outside the repository. This directory can also be reused by the reranker-distillation experiment.

```bash
WORK=/path/to/new-working-directory
mkdir -p "$WORK"
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o "$WORK/scifact.zip"
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o "$WORK/scifact-original.tar.gz"
(cd "$WORK" && sha256sum -c <<'EOF'
536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165  scifact.zip
11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be  scifact-original.tar.gz
EOF
)
unzip -q "$WORK/scifact.zip" -d "$WORK"
tar -xOzf "$WORK/scifact-original.tar.gz" data/claims_train.jsonl > "$WORK/claims_train.jsonl"

python scripts/prepare_scifact.py --data "$WORK/scifact" \
  --original-claims "$WORK/claims_train.jsonl" --config config/config.json --out "$WORK/prepared"

python scripts/train_scifact_reranker.py --data "$WORK/scifact" --prepared "$WORK/prepared" \
  --config config/config.json --out "$WORK/smoke" --smoke

python scripts/train_scifact_reranker.py --data "$WORK/scifact" --prepared "$WORK/prepared" \
  --config config/config.json --out "$WORK/formal" --save-model
```

The [dependency list](requirements.txt) pins NumPy and Transformers from the historical run. PyTorch and transitive dependencies are not pinned, so this is not a complete environment lockfile; the full workflow has not been validated in a clean environment. Data preparation also requires `curl`, `unzip`, `tar` and `sha256sum`. Training scripts and configuration are in this directory. The model revision is fixed in [the config](config/config.json).

On 2026-11-14, a Linux ARM64 NVIDIA GB10 completed 180 CUDA training steps, model saving, and evaluation on 300 test queries using the pinned RoBERTa-base and hash-checked SciFact data and candidate pool. BM25 and the fine-tuned model had nDCG@10 of 0.6617 and 0.6606 respectively; the paired per-query bootstrap 95% interval for the difference was [-0.0342, 0.0317], so this run does not establish a stable advantage over BM25. Test queries were held out, but some relevant documents occurred in training; candidate coverage and the single seed also limit the conclusion. The separate copied and adjusted environment passed `pip check` with Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.17.0, and NumPy 2.5.3; it was not installed from scratch.

Replotting the result requires `python -m pip install matplotlib`. Run `python scripts/plot_ranking.py evidence/summary.json "$WORK/ranking.png"` to avoid overwriting the example figure in the repository. This experiment retains scripts, configuration and evaluation evidence keyed by document ID. The BEIR data card specifies CC BY-SA 4.0. Original SciFact claim and evidence annotations use CC BY 4.0; the abstract corpus uses ODC-By 1.0. Obtain data and model weights from their respective sources: [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) · [AllenAI SciFact](https://github.com/allenai/scifact) · [original license](https://github.com/allenai/scifact/blob/master/LICENSE.md) · [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base).
