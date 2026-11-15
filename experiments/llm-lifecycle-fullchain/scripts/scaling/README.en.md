# Scaling: Data-Scale Comparison (Machine-Independent Components)

This experiment asks how loss, BPB, throughput, and memory change as pretraining data grows for the same model architecture and held-out validation set. The retained measurements are single-GPU runs; the two-node DDP comparison is not complete. This page preserves the procedure for reproducing the comparison. Read [中文](README.md).

Only machine-independent components are kept here:

| File | Purpose |
|---|---|
| `collect_scaling.py` | Aggregates a local results directory (`--results-dir`, default `$XL_OUT_DIR/scaling`) into JSON, a comparison table, and three plots. Makes no remote calls. |
| `profile_vram_batch.py` | Benchmarks memory, data scale, and batch throughput. Model dimensions use `--d-model/--n-layers/--n-heads/--d-ff`; defaults match this scaling experiment: d_model=320, n_layers=6, n_heads=8, d_ff=1280. |
| `inspect_parquet.py` | Inspects Parquet corpus structure (row count, fields, and sampled character counts). |

## Multi-node orchestration and cross-device execution

The comparison is intended to measure throughput and convergence for single-GPU (`single`) and two-node, two-GPU (`dual`) runs across two separate devices. Because cross-device launch commands depend on each environment's network topology, authentication, host addresses, and interface names, this repository does not include a remote launcher tied to a particular cluster. It keeps general local/single-node training, metric aggregation, memory benchmarking, and corpus inspection tools instead.

- Before cross-device runs, make sure both devices use the same code revision and PyTorch/CUDA environment, and can access the same Parquet corpus and fixed validation slice under `$XL_DATA_DIR`.
- The nodes must communicate over `MASTER_ADDR:MASTER_PORT`. On multi-interface systems, explicitly set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`. For `torchrun --nnodes=2`, interface selection, and artifact locations, see the “GPU execution” section in the experiment's top-level [`README.md`](../../README.en.md).
- Only the main node (`node-rank=0`) writes `{pretrain,sft}_metrics.json`. After a run, copy each tier's metric files into a common directory for offline tables and plots.

## Environment and data setup

Run commands from the experiment root, `llm-lifecycle-fullchain/`. Install dependencies and CUDA-enabled PyTorch as described in [the top-level environment section](../../README.en.md#requirements-and-setup), then install Parquet and download tools:

```bash
python -m pip install pyarrow 'huggingface_hub>=0.34,<1.0'
python -c "import torch, pyarrow; print('CUDA:', torch.cuda.is_available(), 'pyarrow:', pyarrow.__version__); assert torch.cuda.is_available()"
```

Historical runs used `shard_00000.parquet` from [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) for training, `shard_06542.parquet` for fixed validation, and `data/train-00000-of-00004.parquet` from [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) for SFT. These files require network access and disk space. Download only the needed shards rather than the full datasets:

```bash
hf download karpathy/climbmix-400b-shuffle shard_00000.parquet shard_06542.parquet --repo-type dataset --local-dir datasets/climbmix
hf download HuggingFaceTB/smol-smoltalk data/train-00000-of-00004.parquet --repo-type dataset --local-dir datasets/smol-smoltalk
python scripts/scaling/inspect_parquet.py datasets/climbmix/shard_00000.parquet
python scripts/scaling/inspect_parquet.py datasets/climbmix/shard_06542.parquet
```

Historical results do not record dataset revisions or file hashes, so these downloads may not be byte-identical to the historical files. Record the revision and hashes for a rerun, and follow the upstream licenses. ClimbMix uses separate training and validation shards. The script splits loaded SFT examples into train and validation sets by seed; this does not guarantee there is no content overlap across sources. The download commands have not been verified end to end.

## Reproduction steps

### 1. Full-chain entry point and memory check

```bash
# Run normally on a single machine or external GPU server; set XL_OUT_DIR / XL_DATA_DIR / XL_RUN_ID as needed
bash run.sh --config configs/full.json

# Memory and throughput benchmark on the target GPU
python scripts/scaling/profile_vram_batch.py --device cuda --amp bf16 --out "${XL_OUT_DIR:-.work/out/local}/scaling"
```

`run.sh --config configs/full.json` runs the full chain. It does not run the data-scale comparison below or automatically use the downloaded Parquet files. If `corpus.txt` is missing, it generates toy data; see the top-level README.

### 2. Single-node data-scale comparison

The example below runs the 100K-character tier. It trains a shared tokenizer on the first 30 million characters of the training shard, then pretrains on the first 100,000 characters and continues from that checkpoint with SFT. For the other tiers, use the same tokenizer, change `--max-chars` to `2000000` or `20000000`, and write to a new output directory. The training subsets are nested prefixes, not independent samples.

```bash
export XL_RUN_ID=scaling-single-t1
export XL_DATA_DIR="$PWD/datasets"
export XL_OUT_DIR="$PWD/.work/scaling/single/t1_micro_100k"
TOKENIZER="$PWD/.work/scaling/tokenizer"
TRAIN="$PWD/datasets/climbmix/shard_00000.parquet"
VAL="$PWD/datasets/climbmix/shard_06542.parquet"
SFT="$PWD/datasets/smol-smoltalk/data/train-00000-of-00004.parquet"

python scripts/01_train_tokenizer.py --corpus-path "$TRAIN" --max-chars 30000000 --vocab-size 8192 --output-dir "$TOKENIZER"
python scripts/03_pretrain.py --device cuda --amp bf16 --tokenizer "$TOKENIZER/tokenizer.json" \
  --train-source "$TRAIN" --val-source "$VAL" --max-chars 100000 --val-max-chars 2000000 \
  --d-model 320 --n-layers 6 --n-heads 8 --d-ff 1280 --block-size 256 \
  --steps 600 --batch-size 16 --eval-every 100 --eval-windows 64 --no-baseline
python scripts/05_sft.py --device cuda --amp bf16 --tokenizer "$TOKENIZER/tokenizer.json" \
  --init "$XL_OUT_DIR/checkpoints/pretrain.pt" --sft-path "$SFT" --test-source "$VAL" \
  --steps 200 --batch-size 16 --eval-every 50 --hit-eval-rows 24 --val-eval-rows 256
```

Historically, the 20M tier used 1,800 pretraining steps; the other two used 600 each. In the example, `--test-source` reuses the pretraining validation shard, so BPB/PPL after SFT are tracking metrics on that same shard, not independent test-set metrics. Results and weights go under `XL_OUT_DIR`; use a new directory for each run to avoid overwriting. These commands were assembled from the current script interfaces and have not fully rerun the historical configuration; they do not guarantee the historical numbers.

### 3. Two-node scaling comparison (`dual`) and aggregation

To reproduce the scale comparison across `single` and `dual` using two devices:

1. **Organize output directories:** Assign each data tier `<tier>` its own `<results-dir>/<mode>/<tier>` output directory, where `<mode>` is `single` or `dual`.
2. **Run each tier:**
   - `single`: On one machine, set `XL_OUT_DIR=<results-dir>/single/<tier>` and run `03_pretrain.py` for each tier, controlling scale and held-out validation with `--max-chars` and `--train-source` / `--val-source`. Then continue with `05_sft.py`, using `--init` for the pretrained checkpoint and `--sft-path` / `--test-source` for SFT data and the external test stream.
   - `dual`: On device A (`node-rank=0`, with `XL_OUT_DIR=<results-dir>/dual/<tier>`) and device B (`node-rank=1`), simultaneously run the same tier with `torchrun --nproc_per_node=1 --nnodes=2 --master_addr=<NODE0_IP> --master_port=29500`. Decide from the comparison protocol whether to halve per-GPU `--batch-size` to keep global batch size constant.
3. **Aggregate and plot:** Put the tier metric files from `<results-dir>/single/` and `<results-dir>/dual/` in the expected common results tree (only `{pretrain,sft}_metrics.json` are needed), then run on any machine:

```bash
python scripts/scaling/collect_scaling.py --results-dir "<results-dir>" --modes single,dual
```

## Artifacts

Outputs include `scaling_summary.json`, `comparison_table.md`, `fig_data_scaling_{single,dual}.png`, `fig_speedup_single_vs_dual.png`, `fig_sft_downstream_{single,dual}.png`, and per-tier raw metrics and loss curves. Sanitized copies are in the experiment's `evidence/` directory.

## Hardware profiles

- Smoke profile (`smoke.json`): CPU-only, a few minutes.
- Full profile (`full.json`): one GPU supporting BF16. Required memory depends on model, batch size, and device; minimum requirements are unverified. See the scans from `profile_vram_batch.py` for specific measurements.
- CUDA communication for multi-GPU DDP depends on the PyTorch build and backend. Linux can use NCCL; gloo/CUDA paths in other environments require separate verification.
