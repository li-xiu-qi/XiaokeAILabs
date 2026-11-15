# Unified-Memory Inference with Strata on DGX Spark

[中文](README.md)

## Questions

On the same DGX Spark (GB10, unified memory), this experiment compares two ways to run Qwen3.8-Flash-Next IQ3_S:

1. Can the ARM64 community port of Strata build and run inference on GB10 (aarch64, sm_121)?
2. What generation speeds do the current Strata and llama.cpp configurations reach?
3. Is Strata's automatic expert-cache allocation reliable on unified memory, and what manual tuning is needed?

**Comparison caveat:** Strata's native IQ package requires MTP (`--spec >= 2`), while llama.cpp cannot load this model's MTP architecture. The comparison is therefore “Strata with MTP enabled vs. llama.cpp without MTP,” not a fair comparison with speculative decoding toggled under otherwise identical conditions. A three-prompt llama.cpp matrix for the same model family is documented in [llm-inference-perf](../llm-inference-perf/README.md); quantization, engine and MTP settings differ.

## Experiment setup

| Item | Configuration |
|---|---|
| Strata source | [Niko1221/Strata PR #3622](https://github.com/Niko1221/Strata/pull/3622), fetched from the pinned `refs/pull/3622/head` reference |
| Pinned commit | `d891f006aaec64024513a1e55856afd992baaa94` |
| Model | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S (83.62 GB, two GGUF shards) |
| Hardware | DGX Spark, GB10, 20 ARM64 cores, 121 GiB unified memory, sm_121 |
| Software | Ubuntu 24.04, CUDA 13.0, built from source |
| Context | 8,192 tokens, one request, vision disabled |
| Sampling | temperature 0.7, `max_tokens` 20–200 depending on scenario |
| Strata speculation | Existing record says `--spec 4 --spec-min-p 0.5 --mtp <path>`; this conflicts with `spec=6` in the available startup log. Actual benchmark arguments need verification. |
| Baseline | llama.cpp 0.4.0-dev (commit ccc3646), `-ngl 40 -t 8 -c 8192`, no MTP (the architecture does not support MTP for this model) |

## Pinned versions

These versions make the experiment traceable. Results may not be directly comparable if components differ in a later reproduction.

| Component | Version identifier |
|---|---|
| Strata engine | 0.1.47 (PR #3622, commit `d891f006aaec64024513a1e55856afd992baaa94`) |
| llama.cpp | 0.4.0-dev (build 1, commit `ccc3646`), GNU 13.3.0, Linux aarch64, build directory `build-force-cublas` |
| Model shard 1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`, SHA-256: `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| Model shard 2 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`, SHA-256: `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| CUDA | 13.0 (`/usr/local/cuda/bin/nvcc`) |
| Experiment date | 2026-10-26 |

## Protocol

Nine requests cover warm-up, short and longer inputs, and repeated requests. The engines' reported prefill/decode tokens per second, output token count, completion reason and Strata draft acceptance were recorded. Some generations differ in length across engines, and several reached the length limit, so these requests cannot evaluate full answer quality. The longest input was 131 tokens.

## Environment

DGX Spark (GB10, 121 GiB unified memory, 273 GB/s), Strata built from source (aarch64, CUDA 13.0), and llama.cpp `build-force-cublas`. Other large-model services were stopped before the test to avoid resource contention.

## Status

- [x] Engine build (2026-10-26): aarch64 + CUDA 13.0 + sm_121, Strata 0.1.47.
- [x] Model deployment: IQ3_S, 83.62 GB; SHA-256 checks passed.
- [x] Single-request semantic smoke (2026-11-15): Strata answered `17×23=391` on the GPU. The older benchmark outputs have not been individually rechecked for answer quality.
- [x] Strata benchmark: generation 20.5–47.3 tok/s, prompt 49.1–166.3 tok/s, MTP acceptance 50–70%.
- [x] llama.cpp comparison: generation 11.0–16.0 tok/s, prompt 21.3–83.2 tok/s.
- [ ] Memory safety: an existing record reports a minimum of 61 GiB available and no mem-guard events, but time-series sampling evidence still needs verification.
- [x] [Comparison report](../../docs/zh/inference/Strata与llama.cpp在GB10上的推理对照.md): per-request speed ratios for eight non-warm-up requests are about 1.6–3.0, with an arithmetic mean of about 2.4. Generation lengths differ in some cases; this reflects only these requests and configurations.
- [x] Qwen3.6-35B-A3B supplementary test (2026-10-26 22:22): UD-Q4_K_XL; Strata generation was stable at 51.0–55.1 tok/s.
- [x] Qwen3.6 cross-engine supplementary test (2026-10-26 22:28): llama.cpp + MTP reached 65.5–87.0 tok/s; this configuration differs from Strata's speculative configuration.

Raw data is in [evidence/](evidence/) (per-request JSON with timings and draft statistics; Qwen3.6 results are in `qwen36-results.jsonl` and `llama-qwen36-results.jsonl`).

### GPU semantic smoke on 2026-11-15

Strata 0.1.47 was restarted on DGX Spark (Ubuntu 24.04, CUDA 13.0, NVIDIA driver 580.173.02). `nvidia-smi` recorded GPU utilization changing from 0% before the request to 88% afterward. The model returned `391` and passed the semantic check. This run used `--expert-cache 4096`, removed `--ple-io ram` so the PLE table was read from SSD, and set context 8192, temperature 0 and `max_tokens=128`. Startup logs reported 46.84 GiB of main-model experts loaded and a 10.16 GiB GPU expert cache. GB10 uses unified memory; these figures are not discrete-GPU VRAM requirements.

The request generated 74 tokens. The server reported 60.8 tok/s generation speed, 78.7 tok/s prompt processing speed, and 53 accepted out of 62 draft tokens. This was one short request with a memory-conscious configuration. These values document the smoke run; they are not a benchmark and should not be compared directly with the historical benchmark above. See the reproduction steps and startup configuration below.

## Key findings

1. **The two configurations had different generation speeds on this machine.** For the eight non-warm-up Qwen3.8 requests, Strata with MTP was faster per request than the llama.cpp configuration without MTP. Since both engine and speculative decoding changed, the difference cannot be attributed to expert caching or MTP alone.
2. **The Qwen3.6 supplementary test showed the opposite direction.** llama.cpp used MTP; Strata's recorded draft count was zero. Generation speeds were 65.5–87.0 and 51.0–55.1 tok/s respectively. These observations compare different configurations and do not establish a general engine-selection rule.
3. **Resource figures and settings need further checking.** GB10 uses unified memory, so different allocation categories cannot be interpreted directly as separate memory requirements for a discrete GPU. The available startup log records `expert_slots=10701`, `expert_cache_mib=20798` and `spec=6`, which conflict with 4096 and `--spec 4` in this document. It is not confirmed that the log and benchmark belong to the same run. Primary evidence for OOM in auto mode, CPU bottlenecks and the range of discrete GPUs that can run the model is still missing.

## Limits of memory statistics

The startup log reports 46.84 GiB loaded, but that is insufficient to verify the previously listed resource-use figures for the two engines or their measurement methodology. Memory categories on a unified-memory machine must be interpreted in light of how they were collected. This experiment did not test a discrete GPU.

## Reproduction

The preparation script targets Linux aarch64 and CUDA. It requires executable `/usr/local/cuda/bin/nvcc`, Git, Python 3.10+ (including venv and pip), GNU `timeout` and `tee`. Building llama.cpp also requires CMake and a C/C++ compiler; the benchmark script requires `curl`. The historical environment is listed above; it does not define minimum requirements. The two model shards total about 83.6 GB. Reserve additional space for builds, converted artifacts and runtime memory, and check free disk and available memory before preparation.

From the repository root, enter this experiment directory and set a working directory. The following uses Hugging Face Hub to download the required shards. Existing shards may be reused, but their hashes must still be checked:

```bash
cd experiments/strata-arm64
export STRATA_WORK_ROOT="/path/to/strata-work"
python3 -m venv "$STRATA_WORK_ROOT/.venv"
"$STRATA_WORK_ROOT/.venv/bin/python" -m pip install -r requirements.txt
"$STRATA_WORK_ROOT/.venv/bin/python" -m pip check
"$STRATA_WORK_ROOT/.venv/bin/python" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", revision="ed59f92082b1e93c0e96d60a8b11aab089b52f09", allow_patterns="IQ3_S/*.gguf", local_dir=os.path.join(os.environ["STRATA_WORK_ROOT"], "models"))'
(
  cd "$STRATA_WORK_ROOT/models/IQ3_S"
  sha256sum -c <<'EOF'
4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
EOF
)
```

The model-download dependency is pinned in [requirements.txt](requirements.txt). Strata's installation script manages its own environment and dependencies. Source the [pinned model repository](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S). If upstream files have changed, hash verification will fail; investigate the source rather than treating the new files as the historical model. After checking available space, run the preparation script:

```bash
bash scripts/prepare_dgx.sh
```

The script fetches code from the pinned `refs/pull/3622/head` reference in the [Strata upstream PR #3622](https://github.com/Niko1221/Strata/pull/3622) and verifies the commit hash in the table; it does not depend on a contributor-fork branch that is no longer advertised. It then adjusts build parallelism and invokes the upstream installer. Preparation may download dependencies or weights; inspect upstream scripts and check free space first. The model's two GGUF shards listed above are not stored in this repository. Preparation logs go to `$STRATA_WORK_ROOT/artifacts/prepare.log`. An existing checkout must match the pinned HEAD; the script does not replace it. If `models/IQ3_S/` exists, it is passed upstream via `--gguf-dir`; an incomplete directory causes an upstream error and is not automatically completed. Upstream also requires about 5 GB for the MTP draft layer and conversion artifacts; reserve space as instructed upstream.

After preparation, inspect the `args` array in `$STRATA_WORK_ROOT/upstream/strata-iq3_s.json`. The preparation script suggests setting the value after `--expert-cache` to `4096`; edit and save the file with a text editor if using that setting. This configuration still needs validation and is not confirmed as the historical benchmark setting. Check available memory and that port 8890 is free, then start the service in another terminal and run the benchmark from the experiment directory:

For a short semantic check on GB10, set `--expert-cache` to `4096` and remove `--ple-io` plus its following `ram` value from the `args` array to leave headroom for model loading and the task guard. This reads the PLE table from SSD instead of reserving about 29 GB of memory. It changes performance, so do not compare its speed directly with historical measurements.

```bash
export STRATA_WORK_ROOT="/path/to/strata-work"
cd "$STRATA_WORK_ROOT/upstream"
./run-iq3_s.sh
```

```bash
bash scripts/run_benchmark.sh
```

To check GPU inference and a basic arithmetic answer without running a benchmark, use this short smoke command. It starts Strata, waits for the API, sends one `17×23` request, saves the response summary and GPU state, then stops the server. NVIDIA's `nvidia-smi` utility is required. The launch configuration uses `--expert-cache 4096` and omits `--ple-io ram`:

```bash
export STRATA_WORK_ROOT="/path/to/strata-work"
bash scripts/smoke_gpu.sh
```

The summary is written to `$XL_OUT_DIR/summary.json` for managed runs or `$STRATA_WORK_ROOT/artifacts/gpu-smoke/summary.json` for local runs.

For the llama.cpp comparison, check out the historical source version and build `llama-server` with CUDA. The following commands follow the [upstream build instructions for the corresponding commit](https://github.com/ggml-org/llama.cpp/blob/ccc3646c63127e32ab8e2773290a1aded6dfb69c/docs/build.md), and pin the source and example options only. The full historical build options were not retained, and these commands have not been rerun:

```bash
git clone https://github.com/ggml-org/llama.cpp.git "$STRATA_WORK_ROOT/llama.cpp"
git -C "$STRATA_WORK_ROOT/llama.cpp" checkout --detach ccc3646c63127e32ab8e2773290a1aded6dfb69c
cmake -S "$STRATA_WORK_ROOT/llama.cpp" -B "$STRATA_WORK_ROOT/llama.cpp/build-cuda" \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build "$STRATA_WORK_ROOT/llama.cpp/build-cuda" --target llama-server --parallel 1
"$STRATA_WORK_ROOT/llama.cpp/build-cuda/bin/llama-server" --version
```

When resources are sufficient, the Strata service is stopped and port 8891 is free, start the server with the first verified shard; keep the second shard in the same directory. The historical benchmark used `-ngl 40 -t 8 -c 8192`, but the available material does not confirm all server and build arguments:

```bash
"$STRATA_WORK_ROOT/llama.cpp/build-cuda/bin/llama-server" \
  -m "$STRATA_WORK_ROOT/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  -ngl 40 -t 8 -c 8192 --host 127.0.0.1 --port 8891
```

Once the service is ready, run the following from the experiment directory in another terminal. Both benchmarks use Bash, curl and the Python standard library; they require no extra Python packages or external data. Prompts are built into the scripts. Results are written under `$STRATA_WORK_ROOT/artifacts/benchmark/` with second-resolution timestamps, so runs in the same second may append to the same file.

```bash
export STRATA_WORK_ROOT="/path/to/strata-work"
curl -fsS http://127.0.0.1:8891/health
bash scripts/run_llama_benchmark.sh
```

The preparation script suggests setting `expert-cache` to 4096, but the cache and speculation settings in the existing startup log differ from the historical record; see “Key findings” above. These steps provide run entry points, but cannot guarantee reproduction of the reported table. In particular, existing evidence cannot confirm that the llama.cpp build configuration above matches the historical build.
