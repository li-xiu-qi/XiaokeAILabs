# Single-GPU Inference of DeepSeek-V4-Flash 2-bit on DGX Spark

[中文](README.md)

## Questions

iggerask's [DeepSeek-V4-Flash-2bit-GB10](https://huggingface.co/iggerask/DeepSeek-V4-Flash-2bit-GB10) is a 299B MoE checkpoint quantized with 2-bit VQ. Its published claim is about 22 tok/s on one DGX Spark (GB10, 128 GB unified memory). We ran it from a clean setup on a GB10 to answer:

1. Can the checkpoint run end to end on a clean GB10, including environment setup, downloading 101 GB of weights, loading, and inference?
2. What are the actual decode speed, memory allocation, and output coherence, and how do they compare with the published figures?
3. Does the model's MTP speculative decoding work here, and in which direction does it affect speed?

**Result first:** it ran. The non-speculative run recorded 20.16 tok/s and 94.62 GiB allocated during model loading; one short Q&A completed. MTP K=2 with FR-Spec recorded 13.42 tok/s, about 33% below the non-speculative run. Logs show a CUDA Graph configuration downgrade, but current measurements do not isolate its cause; see the report.

## Experiment setup

| Item | Configuration |
|---|---|
| Model | `iggerask/DeepSeek-V4-Flash-2bit-GB10` (based on DeepSeek-V4-Flash 299B MoE; experts use 2-bit VQ k=1024/vdim=4/group-64 Hadamard, attn/shared/lm_head use NVFP4 W4A16, routing and embeddings use BF16, KV cache uses fp8) |
| Serving plugin | `vq2_vllm` from `iggerask/deepseek-v4-flash-2bit-gb10` (Apache-2.0; auto-registered through the vLLM general_plugins entry point, not a vLLM fork) |
| Hardware | DGX Spark, GB10, 20-core ARM64, 121.6 GiB unified memory, sm_121 |
| Software | Ubuntu 24.04, CUDA 13.0 (V13.0.88), driver 580.173.02, torch 2.11.0+cu130, vLLM 0.23.0, Python 3.12 |
| Context | 2,048 tokens, one request |
| Sampling | temperature 0, greedy |
| Metrics | decode tok/s (diff method excludes prefill), resident memory, finish_reason |

## Version record

Results are not directly comparable if component versions change. The plugin was recorded only as its master branch, without a commit ID. The model revision below is pinned, but the plugin clone is not, so the exact software state may differ.

| Component | Version identifier |
|---|---|
| Model checkpoint | `iggerask/DeepSeek-V4-Flash-2bit-GB10`, revision `df5ebc8fa0e70f36542ca710fcd5e2ac24fae0bb`, lastModified 2026-06-24 |
| Plugin | `iggerask/deepseek-v4-flash-2bit-gb10` master (includes the mhc_post sm_121 torch-reference patch) |
| vLLM | 0.23.0 (stock, not forked) |
| torch | 2.11.0+cu130 |
| CUDA / driver | 13.0 (V13.0.88) / 580.173.02 |
| Experiment date | 2026-10-27 |

The publisher's README/ARCHITECTURE.md (2026-06-24 version) reports 17.4 tok/s without speculation and 22.5 tok/s with MTP K=2; PPL 4.64 (3.66 for source FP8+MXFP4); GSM8K 95.0%, MMLU-Pro 66.4%, and HumanEval 65.2%.

## Results

The two runs share the memory quota, context length, and shared-experts settings. The non-speculative run disables FR-Spec, while the MTP run enables it, so this compares two configurations:

| Run | Decode tok/s | ms/token | Coherence |
|---|---:|---:|---|
| Non-speculative (K=0) | **20.16** | 49.61 | Pass (Paris + haiku, finish_reason=stop) |
| MTP K=2 | **13.42** | 74.52 | Pass (Paris + three-line haiku, finish_reason=stop) |

The non-speculative load log records 94.62 GiB allocated across 45 safetensors shards; the MTP run records 96.86 GiB. These are not file sizes or full-system peak memory. The non-speculative summary separately records 94.69 GiB; the discrepancy has not been resolved. The speed difference is about −33%. Further measurement is needed to determine whether graph capture, FR-Spec, acceptance rate, or other overhead explains it.

## Environment

The run used a single DGX Spark with no other large model service occupying the device. The 101 GB weights were downloaded through a Hugging Face mirror. Before the first inference, the vq2 CUDA kernel (`vq2_grouped.cu`, sm_121a gencode) took about 3 minutes to JIT compile; later runs use the cache.

The steps below require ARM64 Ubuntu, GB10, a working CUDA compiler toolchain, Python 3.12, Git, curl, and uv. Reserve at least about 101 GB for the model files, plus space for dependencies, caches, and build outputs. These steps describe the recorded setup; recheck dependency conflicts if the plugin or external download sources have changed. Before installation, check the architecture, driver, CUDA compiler, and Python with `uname -m`, `nvidia-smi`, `nvcc --version`, and `python3.12 --version`. In the isolated environment, check versions and CUDA availability with `python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"`. Passing dependency checks does not prove that model loading or performance has been reproduced.

## Reproduction

Dependencies are managed by the external plugin repository's `requirements-serve.lock.txt` and the plugin itself; this directory does not duplicate a requirements file that could drift. The lockfile edits below reflect the historical environment. Recheck conflicts if upstream's default branch has changed. These commands create an environment and model directory in the plugin repository and write to `/tmp/torch-wheels` and `/tmp/req-rest.txt`; first check that those paths do not contain files you need to keep.

```bash
# 1. Plugin repository (vLLM 0.23.0 + vq2 plugin + bench script)
git clone https://github.com/iggerask/deepseek-v4-flash-2bit-gb10
cd deepseek-v4-flash-2bit-gb10

# 2. torch packages (cu130 build; download.pytorch.org connected but
#    transferred no data on the tested domestic network; an Aliyun mirror was used)
mkdir -p /tmp/torch-wheels && cd /tmp/torch-wheels
for f in torch-2.11.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
         torchvision-0.26.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
         torchaudio-2.11.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl; do
  curl -sL -O "https://mirrors.aliyun.com/pytorch-wheels/cu130/$f"
done
cd - && uv venv --python 3.12 .venv-vllm && source .venv-vllm/bin/activate
uv pip install --find-links /tmp/torch-wheels/ \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130

# 3. Remaining dependencies. Three conflicts in requirements-serve.lock.txt
#    need adjustment: fsspec 2026.6.0 -> 2026.4.0 (datasets 5.0.0 upper bound);
#    remove llmcompressor (not needed for inference and conflicts with the
#    compressed-tensors/accelerate pins); keep compressed-tensors at vLLM's 0.17.0.
grep -ivE '^(torch|torchvision|torchaudio|vq2-vllm)==' requirements-serve.lock.txt \
  | sed 's/^fsspec==2026.6.0$/fsspec==2026.4.0/; /^llmcompressor==/d' > /tmp/req-rest.txt
UV_HTTP_TIMEOUT=15 uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /tmp/req-rest.txt
uv pip install -e .

# 4. Download the pinned historical model revision (~101 GB; allow extra cache space)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='iggerask/DeepSeek-V4-Flash-2bit-GB10', revision='df5ebc8fa0e70f36542ca710fcd5e2ac24fae0bb', local_dir='models/DeepSeek-V4-Flash-2bit')"

# 5. Bench. Key environment variables:
#    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 bypassed an MTP capture error here
#    GPU_UTIL=0.85 was the memory quota used; available space varies by system
# Non-speculative:
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 GPU_UTIL=0.85 \
  python scripts/bench.py --spec-mtp 0 --no-frspec --max-model-len 2048
# MTP K=2:
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 GPU_UTIL=0.85 \
  python scripts/bench.py --spec-mtp 2 --max-model-len 2048
```

The download command pins the model repository revision. On 2026-11-13, all 89 safetensors files in an existing copy were hashed and matched the LFS SHA-256 values returned by the official Hugging Face API for that revision. Independent hashes for the historical run and a fixed plugin revision were not recorded, so this does not prove the copy is identical to the historical weights or guarantee strict reproduction. To use another model directory, pass `--model /path/to/model` to the upstream bench or this repository's `min_repro.py`. The installation and pinned-revision download commands above have not been rerun from scratch.

A minimal check on 2026-11-13 was stopped by memory protection during loading and did not generate text. Its reused environment reported dependency conflicts involving `llmcompressor` and `fsspec` in `pip check`; this does not establish that the installation above works. A later inspection found plugin repository commit `abdfcce6d01269eeb2aff3d451b98e183c2de1f4`, but this does not establish which commit was used in the successful 2026-10-27 experiment, nor does it make this commit a validated reproduction. A GPU memory quota is not a guarantee of available system memory: check headroom and whether task protection covers all child processes before running.

On 2026-11-14, a further attempt used a separate environment with a passing dependency check, a shorter context, and a plugin copy with modified weight loading. Available system memory still fell rapidly during initialization, and memory protection stopped the run before it generated text. The change has not been shown to reduce the full loading peak, and this attempt did not validate the clean installation procedure above. The historical result and this incomplete reproduction are separate observations.

The first bench compiles `vq2_grouped.cu` with nvcc; this took about 3 minutes in the recorded run. If a later process waits for the compile lock, first confirm there is no active compiler process, then inspect the lock in the actual cache. To use this repository's script from the plugin repository root, set `XL_REPO` to the XiaokeAILabs clone directory and run `bash "$XL_REPO/experiments/moe-2bit-fit/scripts/run_bench_pair.sh"`. Add `nospec` or `mtp2` to run just one configuration.

## Known limitations

- The MTP K=2 + FR-Spec configuration was slower than this run's non-speculative configuration. Logs show that unsupported FULL graph mode fell back to FULL_DECODE_ONLY. There is no per-stage timing or usable FULL-mode comparison to identify the cause, and the publisher's dependency combination was not tested.
- Context length was 2,048 tokens; the default 4,096 did not fit the available unified memory. Longer-context workloads were not tested.
- Each configuration was measured once on one machine, without repeated trials or variance estimates. The diff method times decode and excludes prefill.

## Contents

- [Single-GPU inference report](../../docs/zh/inference/299B-MoE单机2-bit推理实测.md): main report; use it for conclusions
- `scripts/run_bench_pair.sh`: run both bench configurations (environment variables and per-run options are documented in the script header)
- `scripts/min_repro.py`: minimal check for “loads and emits one token,” useful for isolating environment issues. It defaults to a `0.85` memory quota and a `2048` context; adjust them with `--gpu-memory-utilization` and `--max-model-len`. It can still fail if memory headroom is insufficient.
- `scripts/launch_repro2.sh`: launches this repository's min_repro.py with diagnostic options. Set `PLUGIN_REPO` to the installed plugin/model repository, then run `bash scripts/launch_repro2.sh` from this experiment directory. Logs default to the plugin repository's `.work/`; set `XL_OUT_DIR` to choose another location. The script reports the timeout wrapper's process ID; identify the actual Python child process before using SIGUSR1 to dump thread stacks.
- `evidence/`: raw bench outputs (`bench-output-*.txt`) and result JSONL for both runs
