# Reproducible LLM Lifecycle Training and Fine-Tuning Experiment

This experiment adapts the full-stack workflow in [karpathy/nanochat](https://github.com/karpathy/nanochat) into an eight-stage, from-scratch LLM lifecycle: **BPE tokenizer (GPT-4-style regex pre-tokenization) → modern causal language model → Muon pretraining (with an AdamW baseline) → chat-format midtraining → supervised fine-tuning (SFT) → GRPO reinforcement learning → KV-cache generation → full-chain evaluation**. Core algorithms and training scripts do not hard-code a corpus. `scripts/dataset.py` streams training data; `examples/` independently generates offline smoke data, and fixed domain slices for tokenizer-bias evaluation are in `fixtures/`.

The same code supports **CPU, single-GPU CUDA, and multi-GPU DDP**: `torchrun --nproc_per_node=2` for two GPUs on one host or `nnodes=2` across two hosts. If the official Windows wheel lacks NCCL, backend auto-selection uses gloo. CUDA DDP has limitations in the tested Windows build; see [GPU execution](#gpu-execution-single-gpu-and-two-gpu-ddp). The code also supports mixed precision (`--amp bf16/fp16`), gradient accumulation (`--grad-accum`), and batch scaling (`--scale`); the default small CPU smoke profile remains unchanged. See [中文](README.md).

## Quick Start

```bash
# 1. Clone repository and install requirements
git clone https://github.com/li-xiu-qi/XiaokeAILabs.git
cd XiaokeAILabs/experiments/llm-lifecycle-fullchain
pip install -r requirements.txt

# 2. End-to-end execution (choose a profile):
# Option A: Fast CPU smoke run (approx. 2 minutes, pure CPU, code flow validation)
bash run.sh --config configs/smoke.json

# Option B: Single-GPU run (CUDA + BF16, standalone workstation scale)
bash run.sh --config configs/gpu-smoke.json
```

| Stage | Script | Description |
|---|---|---|
| **1. Data Prep** | `scripts/prepare_data.py` | One-click download of real open datasets (Wikitext-2 / Alpaca / GSM8K), or `--source toy` for offline smoke testing |
| **2. Tokenizer** | `scripts/01_train_tokenizer.py` | Train Byte-Level BPE tokenizer, export `tokenizer.json` and `token_bytes.pt` |
| **3. Pretraining** | `scripts/03_pretrain.py` | Pretraining loss convergence, Muon vs. AdamW baseline |
| **4. Midtraining** | `scripts/04_midtrain.py` | Low-lr dialogue format continual pretraining |
| **5. SFT** | `scripts/05_sft.py` | ChatML template with prompt loss masking (`-100`) |
| **6. GRPO** | `scripts/06_rl_grpo.py` | Critic-free group sampling, exact match reward, relative advantage optimization |
| **7. Inference** | `scripts/07_generate.py` | Generation sampling, KV cache speedup benchmark |
| **8. Evaluation** | `scripts/08_eval_report.py` | Full-chain multi-stage evaluation report |

## Run contract

This experiment separates source configuration from runtime artifacts (implemented in `scripts/llm_common.py`):

| Item | Environment variable | Behavior |
|---|---|---|
| Output directory | `XL_OUT_DIR` | Metrics, checkpoints, plots, and reports go here. Defaults to `.work/out/<run_id>`. |
| Data directory | `XL_DATA_DIR` | Corpus reads and writes go here. Defaults to `.work/data`. |
| Run identifier | `XL_RUN_ID` | Added to each stage's metrics and manifest. `run.sh` defaults to `local-<timestamp>`; individual scripts default to `local`. |

Clone and run locally; `.work/` is Git-ignored. To put data and artifacts on a separate disk or NAS, explicitly set `XL_OUT_DIR`, `XL_DATA_DIR`, and `XL_RUN_ID`. In CI or a controlled environment, `XL_REQUIRE_EXTERNAL_ENV=1` disables local fallback and requires all three variables to be injected.

This directory contains experiment code, configuration, and evidence. Theory, derivations, and analysis are in the companion guide, [Building and Training an LLM from Scratch](../../docs/zh/inference/大语言模型从零构建与训练全链路.md) (Chinese). For interpreting steps, tokens, and training time, see [Training Measurement Units](../../docs/zh/inference/模型训练的计量单位.md) (Chinese). A consolidated report copy is `evidence/fullchain-eval-report-toy-synthetic.md`.

For a separate alignment experiment using fixed preference pairs without sampling during updates, see [DPO preference alignment](../dpo-preference-alignment/README.en.md).

## How the LLM experiments fit together

| Level | Experiment | Guide | Focus |
|---|---|---|---|
| First: individual mechanisms | [`llm-mechanics`](../llm-mechanics/README.en.md) | [Generation Mechanics and KV Cache](../../docs/zh/inference/语言模型生成机制与KV缓存.md) (Chinese) | BPE/BBPE byte segmentation, next-token probabilities, numerical KV-cache equivalence, and low-bit quantization error. |
| Second: full lifecycle | This directory | [Building and Training an LLM from Scratch](../../docs/zh/inference/大语言模型从零构建与训练全链路.md) (Chinese) | Streaming tokenizer training, architecture checks, Muon pretraining, chat-format adaptation, SFT, and GRPO. |
| Third: system performance | [`llm-inference-perf`](../llm-inference-perf/README.en.md) | [Speculative Decoding Results](../../docs/zh/inference/投机解码收益与实测条件.md) (Chinese) | GB10 measurements, speculative decoding for 27B/125B models, same-model MTP and separate DFlash2 draft heads, long-context and task-type limits. |

## Correspondence with nanochat

nanochat trains a GPT-2-level chat model on one 8×H100 node in about three hours. This experiment scales its workflow and key techniques down to a CPU toy setting (about 2.5M parameters and 37 KB of synthetic text) while retaining an inspectable teaching pipeline.

| nanochat | This experiment | Correspondence |
|---|---|---|
| `nanochat/dataset.py` | `scripts/dataset.py` | Streaming batches from Parquet or text, with `max_chars` / `doc_cap` truncation. |
| `scripts/tok_train.py` + tiktoken | `01_train_tokenizer.py` | Byte-Level BPE and GPT-4-style regex pre-tokenization through `tokenizers`, caching `token_bytes`. |
| `nanochat/gpt.py` | Model code in `scripts/llm_common.py` | RoPE, QK norm, relu² MLP, no bias, untied weights, post-embedding norm, logit softcap, and GQA code path. |
| `nanochat/optim.py` MuonAdamW | `llm_common.Muon` + `build_optimizer` | Muon with Newton-Schulz orthogonalization for hidden matrices; AdamW for embeddings and `lm_head`. |
| `scripts/base_train.py` | `03_pretrain.py` | Pretraining plus an AdamW baseline using the same steps and seed. |
| Midtraining stage | `04_midtrain.py` | Chat-format adaptation with lower-learning-rate continuation. |
| `scripts/chat_sft.py` | `05_sft.py` | Instruction tuning with prompt masking. |
| `scripts/chat_rl.py` + `tasks/` | `06_rl_grpo.py` | Group sampling, verifiable exact-match reward, and KL anchoring to a frozen reference. |
| `nanochat/engine.py` KV cache | `llm_common.KVCache` | KV-cache generation checked against naive token-by-token forward passes in stage 02. |
| `scripts/base_eval.py` / report | `07_generate.py` + `08_eval_report.py` | Decoding comparisons, temperature effects, speed comparison, and a four-checkpoint evaluation matrix. |

Large-scale nanochat infrastructure (FSDP, FP8, FlashAttention-3, and a web chat UI) is out of scope. DDP is supported at toy scale: two GPUs on one host or four GPUs across two hosts.

## Pipeline overview

```text
$XL_DATA_DIR/corpus.txt ──► 01_train_tokenizer.py ──► $XL_OUT_DIR/tokenizer/tokenizer.json
   (+ sft.jsonl / rl.jsonl)                                │
                                                           ▼
                                                  llm_common.py (model + Muon + KV cache)
                                                           │
      ┌──────────────────┬───────────────────────────────┼──────────────────────┐
      ▼                  ▼                               ▼                      ▼
 03_pretrain.py     04_midtrain.py                  05_sft.py            06_rl_grpo.py
 Muon vs AdamW      Chat-format adaptation          Instruction tuning  Verifiable-reward GRPO
      │                  │                               │                      ▼
      ▼                  ▼                               ▼                    rl.pt
 pretrain.pt         midtrain.pt                      sft.pt                   │
 (pretrain_adamw.pt)          └───────────────┬───────────────┘                   │
                                             ▼                                   ▼
                                  07_generate.py (greedy / temperature / cache) ◄──┘
                                             │
                                             ▼
                                  08_eval_report.py ──► $XL_OUT_DIR/report.md + summary.json
```

## Requirements and setup

- Python ≥ 3.10; dependencies are in `requirements.txt`. Recorded CPU environment: torch 2.15.0+cpu, tokenizers 0.23.3, matplotlib 3.11.2.
- GPU runs need CUDA-enabled torch. Recorded GPU environment: torch 2.14.0+cu126 on an RTX 3080 Laptop. With `--device auto`, a detected GPU is used; otherwise the script uses CPU.
- Backend auto-selection uses NCCL when CUDA and NCCL are available, otherwise gloo. Official Windows wheels commonly lack NCCL. CUDA DDP depends on the exact environment; see the limitations below.
- Standard public dataset streams are supported. `examples/generate_toy_dataset.py` independently creates offline smoke data.
- `$XL_OUT_DIR/run_manifest.json` records `run_id`, profile, device, AMP mode, process count, and other run details.
- Four CPU threads were fastest for the small model in the recorded multi-core test; eight threads were slower due to contention. Override with `LLM_THREADS`.

Create an environment in this directory. The example uses Linux or WSL Bash. For GPU, first install a PyTorch build suitable for the platform and CUDA version using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the remaining requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
# For GPU, install a CUDA-enabled torch build here, following the PyTorch guide
python -m pip install -r requirements.txt
# Optional: enable staged reading of Parquet corpora
python -m pip install pyarrow
python -c "import torch, tokenizers; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'tokenizers:', tokenizers.__version__)"
```

The requirements include Matplotlib for plots; if unavailable, PNG generation is skipped. The two topic-specific plotting scripts also use NumPy, usually installed with Matplotlib. `matplotlib_layout_audit` is an optional external layout checker; without it, audit is skipped. For multiprocess runs, make sure `torchrun` uses the same Python environment as the installed PyTorch.

The one-command runner reads data from `$XL_DATA_DIR` (default `.work/data`). If it does not find `corpus.txt`, it generates toy data and writes `corpus.txt`, `sft.jsonl`, and `rl.jsonl`, which may overwrite existing files with those names. For custom corpora, prepare all three files; JSONL fields are illustrated by `examples/generate_toy_dataset.py`. `run.sh` does not accept an external data-source argument. To use Parquet, invoke the supporting stage scripts individually.

### Recorded CPU smoke run

On 2026-11-12, the `smoke` profile completed with recorded exit code 0. Logs reached stage `[8/8]` and wrote a report. The manifest records Python 3.12.3, PyTorch 2.11.0+cu130, tokenizers 0.22.2, and Matplotlib 3.11.0; device `cpu`, `nproc=1`, `amp=none`. The device was a GB10 with CUDA available, but the run used CPU. This verifies only that CPU smoke run, not GPU training or the `full` profile. See `run_manifest.json` and `report.md` in its output directory.

### Recorded CUDA GPU smoke run

On 2026-11-13, `configs/gpu-smoke.json` completed all eight stages on an NVIDIA GB10 with exit code 0. Environment: Ubuntu 24.04 ARM64, Python 3.12.3, PyTorch 2.15.0+cu132, tokenizers 0.23.3; training used `cuda:0`, BF16, and one process. This profile differs from CPU smoke: each of two pretraining runs used 60 steps, midtrain 40, SFT 80, and RL 20.

### Two-node CUDA toy-data workflow

On 2026-11-14, a toy-data workflow was run stage by stage using one GPU on each of two machines with NCCL and BF16. Both isolated environments used Ubuntu 24.04 ARM64, Python 3.12.3, PyTorch 2.15.0+cu132, and tokenizers 0.23.3. SHA-256 checks covered the independently generated corpus, tokenizers, and subsequently copied checkpoints; weight broadcast, gradient consistency, and checkpoint loading checks passed. Muon and AdamW each completed 60 pretraining steps, followed by 40 midtraining, 80 SFT, and 20 GRPO steps on both GPUs; both processes exited with code 0 for each stage. Tokenizer preparation, generation, and evaluation used single processes. Held-out arithmetic accuracy was 9% of 47 examples after both SFT and GRPO; this does not establish improved generalization. The environments were copied from an existing dependency set and passed `pip check`, not installed from scratch. Stages were launched manually rather than by a two-node one-command script, and the result does not establish performance on real data.

SFT recorded 80 training-history entries, a PyTorch allocated-memory peak of 73.99 MiB, and mean loss over the first/last four steps decreasing from 7.7149 to 0.0873. Held-out accuracy was 25% on 8 examples, in addition to 49 training examples. These toy-data results validate a workflow, not model quality or performance. Matplotlib was not installed, so no plots were generated; the manifest and report were still written.

## One-command reproduction

```bash
# Local run; output defaults to Git-ignored .work/. XL_* variables may be exported first.
# Minimal CPU smoke profile, about 2-3 minutes
bash run.sh --config configs/smoke.json

# Single-GPU CUDA + BF16 smoke run on toy data
bash run.sh --config configs/gpu-smoke.json

# Full profile (tokenizer vocabulary target 8192 and standard step counts); input data determines content
bash run.sh --config configs/full.json

# Select the interpreter / two-process CPU DDP on one host / append multi-domain tokenizer evaluation
PYTHON=/path/to/python bash run.sh --config configs/smoke.json
NPROC=2 bash run.sh --config configs/smoke.json
bash run.sh --config configs/smoke.json --with-tok-eval
```

Profiles in `configs/<name>.json` define `device`, `amp`, `scale`, `nproc`, `tokenizer.vocab_size`, `tokenizer.max_chars`, and the `stages.{pretrain,midtrain,sft,rl}` step counts. Stages can also be run individually; defaults write to `.work/out/local` and `.work/data`, unless overridden by `XL_*` variables.

`gpu-smoke` explicitly requires CUDA and exits with an error if no CUDA device is available; it does not fall back to CPU. You can set the positive environment variable `XL_MIN_AVAILABLE_GIB` to check available Linux system memory during training. This check supports only one process. Invalid thresholds, unavailable readings, or insufficient memory stop training. It is not a GPU-memory reservation and is disabled by default.

### Run on external compute or separate storage

To run on a separate GPU server or keep corpora and weights outside the repository, set the three contract variables:

```bash
export XL_RUN_ID="exp-20261112-full"
export XL_DATA_DIR="/data/llm-corpus"
export XL_OUT_DIR="/data/runs/$XL_RUN_ID"
# Optional: require explicit external variables instead of falling back to .work/
export XL_REQUIRE_EXTERNAL_ENV=1
bash run.sh --config configs/full.json
```

Keep large checkpoints on the compute device under `$XL_OUT_DIR/checkpoints/`. To bring back a comparison bundle, collect only `report.md`, `summary.json`, `*_metrics.json`, `run_manifest.json`, and plots.

The individual stages are:

```bash
python examples/generate_toy_dataset.py # Optional: create smoke data under $XL_DATA_DIR
python scripts/01_train_tokenizer.py    # Streaming corpus input; BPE training/export (CPU)
python scripts/02_model.py              # Model checks: QK norm/RoPE/cache/softcap/GQA/AMP
python scripts/03_pretrain.py           # Pretraining (Muon vs AdamW)
python scripts/04_midtrain.py           # Chat-format midtraining
python scripts/05_sft.py                # Instruction tuning
python scripts/06_rl_grpo.py            # GRPO reinforcement learning
python scripts/07_generate.py           # Generation and KV-cache speed comparison
python scripts/08_eval_report.py        # Full-chain evaluation and report.md
```

## GPU execution: single GPU and two-GPU DDP

Training scripts 02–06 share device options parsed by `llm_common.init_distributed`. Under `torchrun` they initialize a process group; launched directly with `python`, they run as a single process without a separate code path.

| Option | Values | Meaning |
|---|---|---|
| `--device` | `auto` / `cpu` / `cuda` / `cuda:N` | Auto uses a detected GPU; under DDP, each process binds to its `local_rank` GPU. |
| `--amp` | `none` / `bf16` / `fp16` | Mixed precision. FP16 uses GradScaler and is CUDA-only; CPU use is rejected explicitly. |
| `--backend` | `auto` / `nccl` / `gloo` | Auto uses NCCL with CUDA + NCCL, otherwise gloo (common on Windows). |
| `--grad-accum` | Integer, default 1 | Number of gradient-accumulation microbatches; increases effective batch when memory is limited. |
| `--scale` | Float, default 1.0 | Scales per-GPU batch in stages 03/04/05 or prompts per step in stage 06. |

Single-GPU commands were measured on an RTX 3080 Laptop with torch 2.14.0+cu126:

```bash
# Model checks: BF16/FP16 forward pass and expected rejection of FP16 on CPU
python scripts/02_model.py --device cuda
# Single GPU, BF16, double batch
python scripts/03_pretrain.py --device cuda --amp bf16 --scale 2
# Full end-to-end profile
bash run.sh --config configs/full.json
```

The following single-host CUDA DDP commands have not been verified end to end in the recorded environment:

```bash
# Two-GPU DDP model checks: broadcast consistency, same-batch gradient consistency, clean checkpoint keys
torchrun --nproc_per_node=2 scripts/02_model.py --device cuda
# Two-GPU DDP training; ranks sample different batches, all-reduce averages gradients, rank 0 evaluates/saves
torchrun --nproc_per_node=2 scripts/03_pretrain.py --steps 60 --device cuda --amp bf16
# One-command two-GPU chain: stages 02-06 use torchrun; 01/07/08 remain single-process
NPROC=2 bash run.sh --config configs/gpu-smoke.json
```

On Windows without NCCL, replace `--device cuda` with `--device cpu` to check the same two-rank semantics using gloo on CPU; this was measured locally:

```bash
torchrun --nproc_per_node=2 scripts/02_model.py --device cpu
```

### Two nodes and two GPUs

Under `torchrun`, two nodes with one GPU each use the same semantics as two GPUs on one node and can spread computation across separate hosts. Before launch:

1. **Align code and environment:** Use the same code version and PyTorch/CUDA dependencies on both nodes.
2. **Align data and tokenizer:** Both nodes must access identical training and validation data through `$XL_DATA_DIR` or `--train-source` / `--val-source`, either from matching local copies or shared storage. Ranks sample different training examples using `rank_seed`; manual data-file sharding is unnecessary. First prepare toy corpus and tokenizer on node A, using new data/output directories:

```bash
export XL_DATA_DIR=/path/to/data XL_OUT_DIR=/path/to/out
python examples/generate_toy_dataset.py
python scripts/01_train_tokenizer.py --corpus-path "$XL_DATA_DIR/corpus.txt" --vocab-size 2048
```

Copy `corpus.txt` and all of `$XL_OUT_DIR/tokenizer/` to the same paths on node B, or use shared storage. At minimum, both `tokenizer.json` and `token_bytes.pt` must match. The toy-data record above uses separate stage launches; subsequent two-node stages also require matching input data and the preceding checkpoint on both nodes.

3. **Check network and select interfaces:** The nodes must reach one another at `MASTER_ADDR:MASTER_PORT` (for example TCP 29500) and over NCCL/Gloo dynamic communication ports. On multi-interface hosts, set the same directly reachable interface on each side; otherwise initialization may hang:

```bash
# Set the actual interface name reachable between the nodes (e.g. eth0, enp1s0, tailscale0)
export NCCL_SOCKET_IFNAME=<reachable-interface>
export GLOO_SOCKET_IFNAME=<reachable-interface>
```

4. **Launch on both nodes and collect artifacts:** Pass the same `--steps`, `--scale`, model configuration, and `XL_RUN_ID`. On Linux with CUDA, backend auto selects NCCL. Only `node-rank=0` (global rank 0) evaluates, saves checkpoints, writes metrics, and plots. The other node performs forward/backward passes and gradient all-reduce. Collect outputs from the main node's `$XL_OUT_DIR`:

```bash
# Node A (master, node-rank=0; writes metrics and checkpoints)
XL_RUN_ID=dual-run-01 XL_DATA_DIR=/path/to/data XL_OUT_DIR=/path/to/out \
torchrun --nproc_per_node=1 --nnodes=2 --node-rank=0 \
  --master_addr=<NODE0_REACHABLE_IP> --master_port=29500 \
  scripts/03_pretrain.py --steps 250 --device cuda --amp bf16

# Node B (worker, node-rank=1; match node A's settings and step count)
XL_RUN_ID=dual-run-01 XL_DATA_DIR=/path/to/data XL_OUT_DIR=/path/to/out \
torchrun --nproc_per_node=1 --nnodes=2 --node-rank=1 \
  --master_addr=<NODE0_REACHABLE_IP> --master_port=29500 \
  scripts/03_pretrain.py --steps 250 --device cuda --amp bf16
```

To reproduce the toy-data run above, set pretraining to `60` steps. Then run `04_midtrain.py` for 40 steps, `05_sft.py` for 80, and `06_rl_grpo.py` for 20, in that order. Run the following on both nodes for each stage, setting `NODE_RANK` to `0` or `1` and `MASTER_ADDR` to the reachable main-node address. Both nodes must use the same `STAGE`, `STEPS`, and `XL_RUN_ID`, with their own `XL_DATA_DIR` and `XL_OUT_DIR` paths:

```bash
STAGE=scripts/04_midtrain.py STEPS=40  # then use 05_sft.py / 80 and 06_rl_grpo.py / 20
torchrun --nproc_per_node=1 --nnodes=2 --node-rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  "$STAGE" --steps "$STEPS" --device cuda --amp bf16
```

Before each new stage, copy its preceding checkpoint (`pretrain.pt`, `midtrain.pt`, or `sft.pt`) from the main node's `$XL_OUT_DIR/checkpoints/` to the corresponding location on the other node and compare SHA-256 hashes with `sha256sum`, or use shared storage. Both nodes also need matching `sft.jsonl` and `rl.jsonl`. After GRPO, run `python scripts/07_generate.py --device cuda` and `python scripts/08_eval_report.py --device cuda` on the main node with the same `XL_*` paths. Use a task manager with a total time limit so communication failures do not leave the nodes waiting indefinitely.

For a speed comparison at equal global batch size, use half the single-GPU `--batch-size` (or `--scale`) per GPU on two GPUs. Keeping per-GPU batch unchanged doubles the global batch per step.

DDP implementation details:

- Rank-offset seeds (`batch_generator` / `rank_seed`) ensure ranks train on different batches.
- Evaluation, sample generation, checkpoint saving, and plotting run only on rank 0; metrics such as stage 06 mean reward are averaged across ranks.
- Checkpoints strip the DDP `module.` prefix. `load_checkpoint` accepts older prefixed weights and supports `map_location`.
- Stage 06 samples under `no_grad` using the unwrapped policy; only policy forward/backward uses DDP. Each group has one forward and one backward (mathematically equivalent to summing then backpropagating in a single process). If ranks have different numbers of valid groups, zero-gradient backward calls pad to the global maximum to avoid mismatched all-reduce calls and deadlock.
- KV cache, RoPE cache, and sampling Generator stay on the same device as their input tensors. Muon's Newton-Schulz operation accumulates in FP32, so AMP needs no special change.

Known environment limitations, measured with the Windows torch 2.14.0+cu126 wheel:

- Its TCPStore requests libuv, but the wheel was built without libuv. `torchrun` rendezvous fails with `use_libuv was requested but PyTorch was built without libuv support`; `USE_LIBUV=0` has no effect (known upstream issue). To check the same two-rank semantics, use `LLM_INIT_METHOD` with a `file://` FileStore and start two processes manually:

```bash
# Both processes share the same FileStore path (Windows example)
LLM_INIT_METHOD=file:///C:/temp/xk_store WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 python scripts/02_model.py --device cpu &
LLM_INIT_METHOD=file:///C:/temp/xk_store WORLD_SIZE=2 RANK=1 LOCAL_RANK=1 python scripts/02_model.py --device cpu &
```

- In the same wheel, gloo backward on CUDA tensors segfaulted (minimal reproduction: wrap a Linear layer in DDP and call `loss.backward()`). The two-rank run was therefore completed with CPU gloo: broadcast weights, gradient consistency, checkpoint keys, and all four stages 03/04/05/06 passed. CUDA two-rank requires a wheel with working gloo/CUDA transfers or Linux/NCCL; the code path is the same and backend auto selects NCCL when available.

## Tokenizer domain-bias evaluation and scaling experiment

Two companion experiments share the same run contract.

### Multi-domain tokenizer compression evaluation

`eval_tokenizer_domains.py` evaluates multiple tokenizers on fixed slices in `fixtures/` (Python code, LaTeX, Chinese technical text, mixed Chinese/English code, and arithmetic). It reports bytes/token, roundtrip, token inflation, and audits the `token_bytes.pt` formula. Add real corpus domains with `--extra-domain NAME=PATH`; no local path is hard-coded:

```bash
python scripts/eval_tokenizer_domains.py \
    --tokenizer-path "$XL_OUT_DIR/tokenizer/tokenizer.json" --tokenizer-name current \
    --extra-domain climbmix_val=/path/to/shard_06542.parquet \
    --audit-source /path/to/shard_06542.parquet --baseline-domain climbmix_val \
    --out "$XL_OUT_DIR/tokenizer/domain_compression_eval.json"
```

Sanitized result copies: `evidence/tokenizer-domain-compression-eval.md` and `.json`.

### Data-scale expansion (`scripts/scaling/`)

Compares training at increasing data scales with the same architecture and held-out validation set. Existing valid runs are single-GPU; a two-node preflight could not reach the second node and was replaced by single-GPU runs, so there is no dual-GPU DDP comparison. Cluster-specific launch scripts are excluded; general aggregation and benchmark tools are documented in [`scripts/scaling/README.en.md`](scripts/scaling/README.en.md). Result copies: `evidence/scaling-summary.json` and `evidence/scaling-comparison-table.md`.

## Hardware profiles

| Profile | Entry point | Hardware | Approximate runtime |
|---|---|---|---|
| Smoke | `configs/smoke.json` | CPU only (4 threads sufficient) | All eight stages, about 2–3 minutes. |
| GPU smoke | `configs/gpu-smoke.json` | One CUDA GPU + BF16 | Toy-data workflow check; runtime varies. |
| Full | `configs/full.json` | One BF16-capable GPU; memory depends on model, batch, and data. Minimum requirement unverified. | Minutes to hours, depending on steps and data. |
| Two-GPU DDP | `NPROC=2 bash run.sh --config configs/gpu-smoke.json` | Two GPUs on one host (Linux NCCL or working gloo/CUDA) | Depends on communication and workload; command and two-GPU comparison are unverified. |

Two-node launch (`nnodes=2`) can be done manually with `torchrun` as above. `run.sh` covers one host with one or more GPUs; cluster-specific multi-host orchestration is not included.

## Directory structure

```text
llm-lifecycle-fullchain/
├── README.md                    This experiment guide
├── requirements.txt
├── run.sh                       One-command runner; profiles are in configs/
├── configs/                     CPU smoke, CUDA+BF16 smoke, and full profiles
├── examples/generate_toy_dataset.py  Offline toy smoke-corpus generator
├── fixtures/                    Fixed tokenizer-domain slices and generator
├── evidence/                    Converged report and evidence copies; raw output stays in $XL_OUT_DIR
└── scripts/
    ├── dataset.py               Shared streaming loader (like nanochat/dataset.py)
    ├── llm_common.py            Model, Muon, cache, data, and generation utilities
    ├── 01_train_tokenizer.py    BPE training/export/checks with GPT-4-style regex pre-tokenization
    ├── 02_model.py              Model checks: QK norm / RoPE / cache / softcap / GQA
    ├── 03_pretrain.py           Muon + AdamW pretraining and AdamW baseline
    ├── 04_midtrain.py           Chat-format midtraining
    ├── 05_sft.py                Instruction tuning with prompt masking
    ├── 06_rl_grpo.py            GRPO with verifiable reward and KL anchoring
    ├── 07_generate.py           Greedy/temperature generation and cache speed comparison
    ├── 08_eval_report.py        Four-stage evaluation matrix and report.md
    ├── eval_tokenizer_domains.py  Multi-domain tokenizer compression evaluation
    └── scaling/                 Data-scale comparison tools (see its README)
```

`$XL_OUT_DIR` contains `tokenizer/tokenizer.json`, `tokenizer_info.json`, and `token_bytes.pt`; `checkpoints/{pretrain,pretrain_adamw,midtrain,sft,rl}.pt`; `{pretrain,midtrain,sft,rl}_metrics.json`; loss/reward plots; `samples.md` / `samples.json`; `report.md` / `summary.json`; and `run_manifest.json`.

## Implementation details by stage

### 01 Tokenizer: Byte-Level BPE + GPT-4-style pre-tokenization

The older `token_bytes.pt` miscounted bytes in the ByteLevel vocabulary, and the BPB numerator once included special-token loss. Current code corrects old caches in memory from the vocabulary. Older evaluation windows also omitted the final prediction position of each complete window. Re-evaluate old BPB/PPL with the original model and validation set; do not convert by a fixed ratio. For pipeline details and vocabulary-size tradeoffs, see [Tokenizer Training and Vocabulary-Size Tradeoffs](../../docs/zh/inference/分词器训练与词表规模权衡.md) (Chinese); for domain evaluation and cross-vocabulary BPB calibration, see [Tokenizer Evaluation and Cross-Vocabulary BPB](../../docs/zh/inference/分词器效果评测与跨词表BPB计算.md) (Chinese).

The template-generated corpus contains capital-city facts, fruit colors, occupations, syntax combinations, English number-word arithmetic, Arabic-digit equations (108 groups), identity/chat samples, and factual-dialogue records: 37,824 bytes across 170 EOS-separated records. There are 83 SFT pairs (68 factual Q&A and 15 arithmetic examples that seed RL capability). The 155 verifiable RL tasks are single-digit addition/subtraction: 108 equations appeared in the corpus and 47 were fully held out, split by equation group.

Pre-tokenization uses GPT-4-style regex with the `tokenizers` library (`Split` isolated + ByteLevel), following nanochat's tiktoken route: split into words, numbers, and punctuation, then run byte-level BPE per piece. Special tokens are `<|endoftext|>` (id=0) and `<|pad|>` (id=1). On this tiny corpus, merges saturate quickly and the vocabulary stops at 775 instead of its target of 2048; this is expected.

### 02 Model architecture (aligned with nanochat `gpt.py`)

| Component | Implementation |
|---|---|
| Normalization | Parameter-free RMSNorm (`F.rms_norm`): no bias, mean-centering, or learned weight. |
| Position encoding | RoPE (base=100000), applied before QK norm as in nanochat. |
| Feed-forward | relu²: `down(relu(fc(x))²)`, hidden size 768; optional SwiGLU comparison via `--mlp-type`. |
| Attention | Causal self-attention with `F.scaled_dot_product_attention(is_causal=True)`; GQA code path retained. |
| Structure | Four pre-norm residual blocks, no model biases, untied token embeddings and `lm_head` by default. |
| Other | Norm after embeddings, logits softcap=15, KV-cache generation engine. |

At default vocab=2048 the model has about 2.56M parameters. With the actual 775-token vocabulary, it has **2,067,072 (2.07M)** parameters; `tok_emb` and `lm_head` have 599,550 each (untied), and block matrices have 868,352. `python scripts/02_model.py` checks shapes, QK-norm scale invariance, RoPE relative positions, cache vs naive forward, softcap, both MLP paths, GQA, and tied/untied weights.

### 03 Pretraining: Muon + AdamW baseline

Split EOS-separated records 95/5 into train/validation and randomly crop 128-token windows. Optimizer groups follow nanochat: **Muon** for block matrices (momentum 0.95, five Newton-Schulz orthogonalization steps, matrix_lr 0.02), AdamW for embeddings and `lm_head` (lr 0.3 / 0.004), warmup + cosine decay, gradient clipping 1.0, batch 16, and 250 steps. Run pure AdamW again with the same steps and seed; both validation loss and runtime go into `pretrain_metrics.json`. See [LLM Pretraining and Scaling Laws](../../docs/zh/inference/大模型预训练原理与扩展律.md) (Chinese) for theory and overfitting analysis.

### 04 Midtraining: chat-format adaptation

Load pretrained weights and continue for 120 steps at 0.2× learning rate using only `User:` / `Assistant:` dialogue records and full-sequence loss. This corresponds to nanochat midtraining between base training and chat SFT: first adapt to the conversation format, then instruction-tune.

### 05 Supervised fine-tuning

Load midtrain weights. Format examples as `User: {question}\nAssistant:` + ` {answer}` + eos. **Compute loss only on answer tokens** (prompt positions use `ignore_index=-100`). Use 0.2× learning rate, batch 16, and 250 steps. After training, generate greedily on held-out instruction pairs and score answer-prefix match rate.

### 06 Reinforcement learning: GRPO with verifiable reward

Use `sft.pt` as the policy and a frozen copy of the same weights as reference for GRPO on single-digit arithmetic:

- Sample groups of 8 answers per prompt at T=1.0; reward is verifiable answer matching.
- Advantage is `(r - mean) / std` within each group. Skip groups with no reward variation because they provide no gradient.
- Loss is `-advantage · logprob(answer tokens) + β·KL(π ‖ ref)`, with β=0.04.
- Reward tiers: exact match 1.0; only punctuation after answer 0.6; answer starts with a digit 0.2; otherwise 0. The latter tiers shape cold start; all-zero groups have no reward variation and therefore no GRPO gradient.
- Evaluate training equations (108, present in pretraining corpus) and held-out equations (47, unseen) separately to observe memorization and generalization. Monitor chat Q&A hit rate for catastrophic forgetting.

### 07 Generation and KV cache

Greedy decoding (`argmax`) and temperature sampling (`softmax(logits/T)` + top_k/top_p truncation + multinomial) are implemented; stop at eos. Fixed per-prompt seeds make sampling reproducible. By default compare midtrain and pretrain checkpoints on the same prompts and compare T=0.8 with T=2.5. Compare KV-cache and naive full-forward decoding in tokens/s.

### 08 Evaluation

Run the same evaluation on pretrain, midtrain, SFT, and RL checkpoints: chat Q&A hit rate (SFT held-out set) plus arithmetic exact match (train/held-out). Write `$XL_OUT_DIR/report.md` and `summary.json`. See [Pretraining-Stage Evaluation and Capability Benchmarks](../../docs/zh/inference/预训练阶段评测与能力基准.md) (Chinese) for base-model vs post-training evaluation, continuous `val_bpb`, and continuation-style multiple-choice benchmark design.

## Measured results

The following values come from one complete CPU run with the default profile. Its `run_manifest.json` is generated in `$XL_OUT_DIR`; converged report and comparison copies are in `evidence/`.

| Stage | Metric | Result |
|---|---|---|
| 01 Tokenizer | Vocabulary / eos_id / pad_id | 775 (target 2048; merges saturate on tiny corpus) / 0 / 1 |
| 01 Tokenizer | Corpus compression | 37,824 bytes → 9,684 tokens (3.906 bytes/token) |
| 01 Tokenizer | Encode/decode roundtrip | 0/50 failures |
| 02 Model | Parameters with vocab=775 | 2,067,072 (2.07M); all stage-02 checks passed |
| 03 Pretraining | Muon train loss, mean first 5% → last 5% of steps | 4.4863 → 0.0924 (↓ 97.9%) |
| 03 Pretraining | Muon best validation loss | 1.2501 at step 49 |
| 03 Pretraining | AdamW best validation loss (same steps/seed) | 1.2775 at step 149 |
| 03 Pretraining | Final validation loss (Muon / AdamW) | 2.3770 / 1.3380 (toy-corpus overfit) |
| 04 Midtrain | Train loss change / final validation loss | 0.4366 → 0.0382 (↓ 91.3%) / 0.9505 |
| 05 SFT | SFT loss | 9.7020 → 0.0005 (↓ 100.0%) |
| 05 SFT | Answer hit rate | Train 100% (71/71), held out 58% (7/12) |
| 06 RL | Arithmetic exact match, train (SFT → RL) | 17% → 23% |
| 06 RL | Arithmetic exact match, held out (SFT → RL) | 13% → 9% (unseen equations not learned) |
| 06 RL | Chat Q&A hit rate, regression check | 60% → 60% (no regression) |
| 07 Generation | KV cache vs naive decoding speed | batch=1: 1.4×; batch=8: 4.9× (96 generated tokens) |

**How to read Muon vs AdamW:** With the same steps and seed, Muon reached a lower best validation loss three times earlier (step 49 vs 149), indicating faster optimization. But after 250 steps Muon's final validation loss (2.3770) was higher than AdamW's (1.3380): on this tiny ~9K-token corpus, Muon memorized training data faster (train loss 0.09 vs 0.83) and overfit earlier. This reflects data scale, not an optimizer defect; nanochat's Muon advantage is expected to show fully at billions of tokens.

Greedy generation examples (full set in `evidence/generation-samples-toy-synthetic.md`):

| Prompt | RL model | SFT model before RL |
|---|---|---|
| `User: What is the capital of Japan?` + `Assistant:` | `Tokyo` | `Tokyo` |
| `User: What color is a banana?` + `Assistant:` | `The banana is yellow.` | `The banana is yellow.` |
| `User: What does Carol do?` + `Assistant:` | `Carol is a engineer.` | `Carol is a engineer.` |
| `User: What is 7 plus 5?` + `Assistant:` | `7` | `7` |

Pretrain and midtrain checkpoints only continue corpus fragments and do not follow chat format; after SFT, the model answers based on question entities. This directly demonstrates that instruction tuning changes behavior distribution. After RL, arithmetic answers consistently take numeric form (for example, `7`), but most exact results remain wrong, as expected from a 2M-parameter toy model. T=0.8 produced stable, coherent output; at T=2.5 diversity increased as outputs began to break down. See `evidence/fullchain-eval-report-toy-synthetic.md` for the four-stage evaluation matrix.

## Reproducibility and determinism

- Random seeds are fixed: corpus generation 1234, training 1234, validation evaluation 1235, and generation sampling 1234 + prompt index. Reruns on the same machine were bitwise identical.
- DDP offsets seeds by rank (`rank_seed`). Single- and multi-GPU runs share the same code; cross-rank gradient averaging means multi-GPU loss curves are not expected to match single-GPU runs bit for bit.
- Stage 02 has structural checks for shapes, QK-norm scale invariance, RoPE relative positions, cache vs naive token-by-token output (including batch=3), softcap, relu² and SwiGLU paths, and tied/untied weights. It also checks AMP forward, Muon device updates, and under torchrun DDP weight broadcast, gradient consistency, and checkpoint keys. Stages 03/04/05 assert loss decreases and exit nonzero when unmet.
- Each stage's full configuration is recorded under `config` in `*_metrics.json`, including distributed settings (`device`, `world_size`, `backend`, `amp`, `grad_accum`, `scale`) and `run_id`.

## Scope and limitations

- Vocabulary is only 775: BPE merges quickly saturate on this tiny corpus. Real use requires more data.
- The model has only 2.07M parameters and 128-token context. It can answer simple facts in its training distribution but does not have general language ability.
- The corpus is synthetic English templates; Chinese and open-domain tasks are out of scope.
- Muon overfits the 9K-token corpus within 250 steps (best validation at step 49); the AdamW baseline decays more slowly. This is a data-scale limitation.
- GRPO reward simplifies punctuation tolerance and numeric-form shaping. Measured RL gains were mainly on training equations (17%→23%); 47 held-out equations declined (13%→9%). The model did not learn addition; it memorized training equations. The generalization limit is reported directly.
- RL can regress on chat format. This experiment tracks Q&A hit rate; it remained 60%→60% in this run.
- Midtrain dialogue is also part of pretraining data. With this small dataset, validation loss rose after continued training (overfitting). Midtrain is only for format adaptation; its hard gate checks only that training loss falls.
- KV-cache benefit grows with batch and output length: 4.9× at batch=8 but only 1.4× at batch=1. For smaller models or shorter generations, fixed per-step cache overhead can erase the gain.
