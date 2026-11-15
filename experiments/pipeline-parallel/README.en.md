# Running One Large Model Across Two Computers: Exploring Cross-Machine Communication

[中文](README.md)

If one computer cannot fit or process a target model, can two computers each handle part of the computation? This directory preserves the communication tests and model-splitting prototypes created to explore that question. For related concepts and the approach used in this Windows/Linux mixed environment, see [Dual-GPU Pipeline Parallel Inference](../../docs/zh/inference/双卡流水线并行推理.md).

**Current status:** Two-device CPU/Gloo and GPU/NCCL communication checks passed with zero error across 256 round-trip elements. On November 14, 2026, the same weight file was copied to both devices and its SHA-256 verified before Qwen2.5-0.5B-Instruct generation; both Gloo and NCCL completed two-device GPU generation, and all eight output tokens for one prompt matched the single-device baseline. The cause of two older copies with differing hashes remains undetermined. Timeline evidence narrows the interval: the hashes matched on November 13, and the mismatch was first observed on November 14. This identifies when the discrepancy became observable, not whether it arose during transfer, receipt or writing, or from an earlier source state. A later transfer with checks at the sender, during receipt, and after writing produced matching hashes, showing that this transfer path can copy the file correctly; it does not explain the older differences. This is a single-prompt functional check, not a general correctness or performance claim.

The block-by-block comparison showed that two older files differed from the verified copy in one and two regions spanning 32 bytes, with 27, 27, and 29 bytes actually different. This rules out whole-file truncation, but the original audit could not identify where the differences arose.

On November 15, 2026, a further read-only audit found that the current sender-side cached `model.safetensors` is 988,097,824 bytes and has SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`, matching the [official checksum for the pinned Hugging Face revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors). Two older files retained on the receiving side are also 988,097,824 bytes, but have different hashes: `71ef72df5184a32e01792df388b187a035ffa75c1f1df5a57699829ab7b5d1e3` (`model.safetensors`) and `a591bf149b8e53f19184af06e24f80e7bef86822f31f6218ee55036fcceb4605` (`model-check.safetensors`). A byte comparison found 27 differing bytes in the first file, spanning byte positions 673,989,287–673,989,318; the second has 27 differing bytes at 220,871,871–220,871,902 and another 29 at 583,115,647–583,115,678. Positions use the one-based numbering reported by `cmp -l`. The observed mismatch is confined to these two retained receiving-side copies; however, the original transfer has no contemporaneous sender/receiver hashes or step-by-step logs, so it is not possible to determine whether the differences arose during transfer, while writing on receipt, or in an earlier source state. A later controlled transfer had matching sender, received-stream, and on-disk hashes, proving that transfer completed intact but not explaining the older files. The two model files used for the current generation, as well as their configuration, tokenizer, and vocabulary files, match individually. The complete directory lists still differ: the more complete snapshot also contains `.gitattributes`, `LICENSE`, `README.md`, and `configuration.json`, so only the inference-required files are confirmed identical. This was a checksum and byte audit; generation was not rerun.

## Goals, success criteria, and budget

There are two levels, in priority order. The criteria were fixed in advance rather than selected after seeing the results.

**Level 1: element-wise tensor communication between two machines** (priority). Each machine runs one process. A probe tensor is sent over the network, returned, and compared element by element. Success requires `probe_element_mismatches == 0`, `probe_max_abs_diff == 0`, and no communication error during one complete run. Preserve failed records and investigate. Set an external overall time limit before running to avoid waiting indefinitely if communication fails.

**Level 2: deterministic generation comparison between one and two machines.** Both machines must already have a usable local copy of a small model. With the same prompt, seed, and generation settings (greedy decoding), compare a full single-machine forward pass with generation split across two machines. Success means the token sequences match exactly. Stop if they match; if not, record the first differing position and stop without tuning. Budget: one model, at most three prompts, and one complete comparison.

Passing the communication test proves only that the tensor transfer is correct; it does not prove that the model runs. A successful model run validates only this particular path and does not imply a performance result. Speculative decoding is out of scope and is not tested.

## What the code attempts

- `scripts/pipeline_stage.py` and `scripts/run_smoke.sh`: start one process on each device, send a random tensor, and perform simulated computation to probe cross-machine communication. They do not load a model or generate text.
- `scripts/real_pp2_inference.py`: attempts to split model layers across two devices and recompute the required values at each generation step.
- `scripts/real_pp2_kv_cache.py`: attempts to save and reuse attention caches during split computation.
- `scripts/real_pp2_speculative.py`: an attempt at multi-candidate generation. It does not implement candidate acceptance checks and rollback, so it is not a correct speculative decoding implementation. To prevent misuse, direct execution stops.

The model prototypes load the full model on each device before trimming layers. They therefore do not show that a model too large for one device can run. A single comparison with a small model does not establish compatibility with all models.

## Run the communication test

First run the CPU/Gloo communication check that has passed. The two devices must be able to connect to each other and have PyTorch with Gloo support and NumPy installed. The launcher uses Bash. Windows users can run it in a Linux environment. Model tests additionally require a usable GPU and local model files on each device.

Prepare the environment on both devices and use the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to install a suitable version for each:

```bash
cd experiments/pipeline-parallel
python3 -m venv .venv
source .venv/bin/activate
# Install PyTorch with CUDA and NCCL as required by the target devices
python -m pip install -r requirements.txt
python -c "import torch, numpy, transformers; print('torch:', torch.__version__, 'numpy:', numpy.__version__, 'transformers:', transformers.__version__, 'CUDA available:', torch.cuda.is_available(), 'NCCL available:', torch.distributed.is_nccl_available())"
```

`requirements.txt` contains NumPy for the communication scripts, Transformers for the model prototypes, and the Hugging Face Hub client for the model download. Install PyTorch separately for the target system to avoid a default package that does not support its CUDA/NCCL setup. The list is not version-pinned, so the installed versions may differ from the tested environment below.

Before running a model prototype, download the same revision of [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on each device and review its model card and license. This example pins a public model commit; the historical comparison did not record a complete model revision, so it cannot be assumed to use identical files. The download requires network access and disk space; choose a new directory for `MODEL_PATH`:

```bash
export MODEL_PATH="$HOME/models/Qwen2.5-0.5B-Instruct"
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
if [ -e "$MODEL_PATH" ]; then
  echo 'Choose a new MODEL_PATH' >&2
else
  python -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", revision=os.environ["MODEL_REVISION"], local_dir=os.environ["MODEL_PATH"], allow_patterns=["*.json", "*.safetensors", "*.txt"])'
fi
```

The paths may differ between devices, but the model files and revision must match. **Before comparing generation across devices, create and compare SHA-256 manifests for the entire snapshot on both machines**; checking only `model.safetensors` is insufficient because configuration, tokenizer, and sharded files also affect loading. Use a dedicated model snapshot directory. These commands exclude `.cache/` metadata generated by Hugging Face `local_dir`; run them on both devices, then copy the first device's manifest to the second:

```bash
cd "$MODEL_PATH"
find . -type f ! -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum > "$HOME/qwen-model-sha256.txt"
```

Copy the first device's manifest to the second, then run this in the second model directory:

```bash
cd "$MODEL_PATH"
find . -type f ! -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum > "$HOME/qwen-model-sha256.local.txt"
diff -u "$HOME/qwen-model-sha256.txt" "$HOME/qwen-model-sha256.local.txt"
```

An empty diff and exit code 0 mean both filenames and file-content hashes match. If they differ, stop the model comparison and redownload or transfer the snapshot. The communication test does not require a model download.

Place the repository on both devices and run the commands from each device's `experiments/pipeline-parallel/` directory. Replace `STAGE1_ADDRESS` with an address for the first device that the other device can reach. Start the first device, then the second:

```bash
# Device 1; replace IFACE1 with its actual network interface name
MASTER_ADDR=STAGE1_ADDRESS GLOO_SOCKET_IFNAME=IFACE1 \
  bash scripts/run_smoke.sh stage1 --backend gloo --device cpu --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --warmup 1 --iters 2 --timeout-min 1 \
  --out .work/smoke-pipeline-metrics.json
```

```bash
# Device 2; replace IFACE2 with its actual network interface name
MASTER_ADDR=STAGE1_ADDRESS GLOO_SOCKET_IFNAME=IFACE2 \
  bash scripts/run_smoke.sh stage2 --backend gloo --device cpu --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --warmup 1 --iters 2 --timeout-min 1
```

Both devices must use the same master address and port and allow the required network connections. The default port is 29500; change it with `MASTER_PORT`. Specify a network interface on each device that can reach the other device. In the tested environment, hostnames resolved to loopback addresses, and initialization timed out without an explicit interface. The test runs only two measured iterations, which is not enough to compare performance.

The first device prints JSON and writes the `.work/` file shown above, including status, configuration, and timing information. The metrics use schema `xiaokeailabs/smoke-pipeline-metrics/v2`:

- `nodes.stage1` and `nodes.stage2` record device and environment details, send/receive and computation timings, and peak memory on each node.
- `results.numerical` records the element-wise comparison of the returned probe on stage 1; stage 2 receives that result.
- `status=passed` means the probe matched and stage 1 received stage 2's metrics. Check both process exit codes as well to confirm stage 2 received the final acknowledgement. This still does not prove that a model runs.
- Effective throughput is reported in GB/s (1 GB = 1e9 bytes). Round-trip latency includes computation and waiting on both devices, so it is not network bandwidth.
- Gloo stage synchronization uses `--barrier-timeout`; NCCL synchronization uses the process-group timeout `--timeout-min`. Exit behavior when communication fails still needs testing.

`.work/` contains local run output and is not included in the repository or provided as historical results. The script errors if the specified output file already exists; choose a fresh path.

## GPU/NCCL communication test

Both devices need a usable NVIDIA GPU, PyTorch with CUDA and NCCL support, and a network connection to each other. The small-tensor test passed on November 14, 2026, between a DGX Spark running Linux and an RTX Spark N1X running Windows/WSL2. Both used Ubuntu 24.04 ARM64, Python 3.12.3, PyTorch 2.15.0+cu132, and NCCL 2.30.7. This is a tested combination, not a minimum requirement.

Activate the separate environment on each device, start the first command, then promptly start the second. Replace the address and interface names with actual values and ensure the first device's port is reachable from the second. This example disables InfiniBand and uses the specified interfaces; parameters must match on both sides. Use a task supervisor that enforces an overall deadline and cleans up all child processes so that a communication failure cannot leave a task running indefinitely.

```bash
# Device 1
MASTER_ADDR=STAGE1_ADDRESS NCCL_SOCKET_IFNAME=IFACE1 NCCL_IB_DISABLE=1 \
  bash scripts/run_smoke.sh stage1 --backend nccl --device cuda --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --local-layers 1 --warmup 1 --iters 2 --timeout-min 1 \
  --out .work/nccl-metrics.json
```

```bash
# Device 2
MASTER_ADDR=STAGE1_ADDRESS NCCL_SOCKET_IFNAME=IFACE2 NCCL_IB_DISABLE=1 \
  bash scripts/run_smoke.sh stage2 --backend nccl --device cuda --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --local-layers 1 --warmup 1 --iters 2 --timeout-min 1
```

Check both exit codes and the first device's JSON. In this run, `status` was `passed`, all 256 probe elements matched, and the maximum absolute difference was zero. Recheck on different devices or networks; this result does not establish that split-model inference works over NCCL.

## Lightweight self-test

`scripts/test_pipeline_stage_gloo.py` is a protocol self-test that does not require a GPU. It starts two local Gloo CPU processes for one real communication round and checks the pure-function definitions and metrics schema. It can run on any machine with PyTorch, including the CPU build:

```bash
python scripts/test_pipeline_stage_gloo.py
```

The local two-process self-test passed on Windows AMD64 with Python 3.13.0, torch 2.15.0+cpu, and NumPy 2.5.4. This does not replace validation between two machines.

## Model prototype boundaries

The prototypes also require Transformers and local model files accessible from both devices, as well as the role, model path, and first-device address. A small-model comparison was completed between Linux and a Linux environment on Windows: the model was loaded in BF16, each device computed 12 of the 24 layers, and activations were converted to CPU FP32 tensors for Gloo transfer. Compatibility with other model architectures and dependency versions has not been confirmed.

`real_pp2_inference.py` provides the `baseline` single-machine entry point and the two-machine `stage1` and `stage2` entry points. For a comparison, explicitly set `--temperature 0`, and use the same `--seed`, prompt, and generation length on both paths. `--out` saves the baseline or stage 1 inputs and generated token IDs as JSON; existing files are not overwritten. Check both process exit codes and compare token IDs element by element. Numerical differences between devices or framework versions may produce different outputs for other inputs.

For example, prepare the same Qwen2.5-0.5B-Instruct model on both devices as shown above and replace the address and interface names. Run the baseline on device 1 and wait for it to finish before starting the two stages in sequence:

```bash
mkdir -p .work
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/real_pp2_inference.py --role baseline --model-path "$MODEL_PATH" \
  --prompt 'The capital of France is' --max-new-tokens 8 --temperature 0 --seed 0 \
  --out .work/baseline.json

# Device 1
GLOO_SOCKET_IFNAME=IFACE1 python scripts/real_pp2_inference.py \
  --role stage1 --model-path "$MODEL_PATH" --master-addr STAGE1_ADDRESS \
  --prompt 'The capital of France is' --max-new-tokens 8 --temperature 0 --seed 0 \
  --out .work/stage1.json
```

```bash
# Device 2: MODEL_PATH must point to the same model on this device
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
GLOO_SOCKET_IFNAME=IFACE2 python scripts/real_pp2_inference.py \
  --role stage2 --model-path "$MODEL_PATH" --master-addr STAGE1_ADDRESS \
  --prompt 'The capital of France is' --max-new-tokens 8 --temperature 0 --seed 0
```

Create `.work/` before the first run. The default `--master-port` is 29505 for both stages; if changed, use the same port on both. The script falls back to CPU when CUDA is unavailable, so check the CUDA status on both devices first.

To use NCCL for GPU tensor communication, verify CUDA and NCCL support on both devices, replace the Gloo interface variables with `NCCL_SOCKET_IFNAME=IFACE1` and `NCCL_SOCKET_IFNAME=IFACE2`, and add `--backend nccl` to both stage commands. Gloo is the default. NCCL requires CUDA; set `NCCL_IB_DISABLE=1` on both sides if InfiniBand must be disabled. Before a model comparison, verify every file's SHA-256 on both devices using the steps above.
