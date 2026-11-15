# NInfer Deployment and Performance Validation on DGX Spark GB10

[中文](README.md)

This experiment tests a concrete question that is easy to misjudge: can the Qwen3.8 27B W4A4 + DFlash2 + Vision model package actually run on the DGX Spark GB10 and provide a reproducible high-throughput service? A separate three-prompt matrix for the 27B model on llama.cpp is in [llm-inference-perf](../llm-inference-perf/README.md); it uses different quantization, engine, and prompts. Acceptance covers more than loading: real generation, MTP, DFlash2, vision, CUDA Graph, OpenAI requests, and C1/C2/C4/C8 throughput comparisons.

> **With a fixed GB10 fork, NInfer completed all of these paths. At the screenshot reproduction settings, C8 reached 132.1 tok/s; the same-model SGLang reference reached 144.6 tok/s, showing that 100+ TPS is not limited to other hardware. Under the current protocol, NVFP4 W4A4 was also faster than FP8.**

## Why run this experiment

The current official NInfer source limits its build target to `sm_120a`, while DGX Spark uses `sm_121a`. Seeing only this error could lead to the mistaken conclusion that the model cannot run on DGX. Four questions need to be separated:

1. Does the current official engine directly support GB10?
2. Is there a verifiable GB10 porting path?
3. Should performance for the same model be described differently under NInfer, SGLang, and llama.cpp?
4. How do DFlash2 acceptance rate, quantization format, CUDA Graph, and memory settings affect actual speed?

The value of this experiment is evidence from actual builds, runs, and measurements, rather than model-card or community figures alone.

## Experiment setup and scope

| Item | Configuration |
|---|---|
| Model | Qwen3.8-27B Coder390 W4A4 + MTP + DFlash2 + Vision, V3 `.ninfer` |
| Model SHA256 | `3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808` |
| Fixed ModelScope revision | `196cc2ce161d7239976fcd5dc07adf06ae31a35c` |
| NInfer port | `baristahaus/ninfer-gb10`, commit `4c2fc2f9504d2f373c84e896bf18523906a646d7` |
| SGLang comparison | `NVFP4/W4A4/` and `FP8/` from the same model repository |
| Hardware | DGX Spark, NVIDIA GB10, aarch64, unified memory |
| Verified software | CUDA 13.0, NInfer GB10 fork, SGLang 0.5.19, PyTorch 2.13.0+cu130 |
| Performance workload | C1/C2/C4/C8, 2 rounds each, 512-token limit, xhigh reasoning |

This experiment does **not** cover 35B/125B MoE models, full GPQA/MMLU/LCB evaluations, long context, sustained concurrent stability, energy use, or production high availability. Small-sample quality probes are not a full accuracy evaluation. The hardware and prompts from the screenshot comparison are not controlled by this experiment.

## What we did

- Reproduced the `sm_120a` build rejection at official revision `64d10fb`, then inspected Qwen3.5 runtime gates, x86 host code, and unified-memory constraints.
- Built all 898/898 targets with a fixed GB10 fork, including CLI, server, tests, and benchmarks.
- Checked the V3 SHA256 and internal objects to confirm text, MTP, DFlash2, and Vision components are present.
- Ran text generation, MTP prefix, real DFlash2, Vision prefix, two repros, default CUDA Graph, and OpenAI text and image requests.
- Reproduced the screenshot configuration with draft 7, xhigh, and a 512-token limit, retaining per-request native timings and draft acceptance rates.
- Compared eager and decode-graph modes for FP8 and NVFP4 on SGLang with the same model, testing whether FP8 was necessarily optimal.

## Findings

### 1. Direct official support is limited, but the GB10 fork runs this model package

The fixed fork built and completed real runs. The architecture rejection at official revision `64d10fb` is not model-file corruption and does not mean GB10 can never run it; support requires actual ARM64, runtime-gate, and unified-memory porting rather than changing one CMake line.

The first full CTest run passed 142/144 tests. The remaining two only lacked a real-artifact environment variable; both passed when rerun with V3. Other skipped real tests are not counted as passes.

### 2. 100+ TPS is reproducible; about 22 tok/s describes a different experiment

| Concurrency | External screenshot | This experiment NInfer (draft 7) | This experiment SGLang reference (draft 8) |
|---:|---:|---:|---:|
| C1 | 48.3 | 32.8 | 31.2 |
| C2 | 76.9 | 58.4 | 63.7 |
| C4 | 97.7 | 95.2 | 99.0 |
| C8 | 146.8 | 132.1 | 144.6 |

The historical result of about 22 tok/s used UD-Q4_K_XL GGUF + llama.cpp + MTP2 + thinking disabled. It differs from this NInfer W4A4 + DFlash2 + xhigh setup in model container, engine, and speculative path, so it does not refute these results.

In this NInfer run, DFlash2 acceptance rates were C1 33.95%, C2 36.62%, C4 36.71%, and C8 32.57%; the screenshot reports 61.8%, 52.8%, 41.2%, and 44.6%. Acceptance rate depends strongly on task text distribution, so total TPS alone cannot establish which engine is better. Screenshot and SGLang C8 differ by only about 1.5%, but the screenshot hardware and prompts are not controlled here.

### 3. Under the current SGLang protocol, NVFP4 W4A4 is faster than FP8

| Concurrency | NVFP4 eager | FP8 eager | NVFP4 decode graph | FP8 decode graph |
|---:|---:|---:|---:|---:|
| C1 | 32.7 | 22.9 | 31.2 | 20.0 |
| C2 | 55.2 | 45.7 | 63.7 | 42.1 |
| C4 | 96.9 | 68.5 | 99.0 | 68.5 |
| C8 | 132.5 | 106.1 | 144.6 | 106.9 |

FP8 did not overtake NVFP4 at any of the four concurrency levels or in either graph configuration. Decode graph helped NVFP4 at C2/C8, but did not consistently speed up FP8. Draft acceptance rate does not explain the entire gap; kernel-level profiling is still needed to attribute it to specific compute or memory-read paths.

### 4. Small-sample quality and vision probes did not distinguish the formats

Both formats passed text arithmetic, image shape/color/text recognition, 10 deterministic automatic questions, and 2 programming questions. The automatic questions scored 10/10; both programming answers passed syntax checks (2/2), and after manual review each passed all 5 functional cases. This only means no obvious quality difference appeared in this small probe set; it does not establish equal full-scale accuracy.

## Reproduction

### Dependencies

Requires Linux aarch64, DGX Spark or compatible GB10, NVIDIA driver, CUDA Toolkit (including nvcc), CMake >=3.28, GCC/G++, Ninja, pkg-config, FFmpeg development files, libcurl >=7.85, and Python 3. The NInfer service script also requires curl, `ss`, `ps`, GNU grep, `setsid`, and `/proc` access to the current user's process environments. Install download-tool dependencies in an isolated environment. The service request and result-analysis scripts use only the Python standard library; image samples are included in the repository. The SGLang comparison uses a separate environment.

```bash
cd experiments/ninfer-serving
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_models.py --help
cmake --version
nvidia-smi
nvcc --version
pkg-config --modversion libcurl libavcodec libavformat libavutil
```

[`requirements.txt`](requirements.txt) does not pin complete versions; `modelscope` is used for downloads, along with helper libraries. These checks do not load a model and do not prove that the engine was built or GPU inference succeeded.

All three model-file groups come from the ModelScope repository `Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2`, fixed at revision `196cc2ce161d7239976fcd5dc07adf06ae31a35c`. Only directories needed for this experiment are fetched.

After download, the script verifies every file against the model repository's `SHA256SUMS`. The NVFP4 and FP8 verification ranges include their respective `DFlash2-FP8` subdirectories.

The optional V2 package in the NInfer `SHA256SUMS` manifest is not needed for this experiment and does not need to be downloaded.

```bash
python scripts/download_models.py ninfer --model-root ./models
python scripts/download_models.py nvfp4 --model-root ./models
python scripts/download_models.py fp8 --model-root ./models
```

If files are already present, add `--verify-only` to the corresponding command to skip downloading and check them. For NVFP4:

```bash
python scripts/download_models.py nvfp4 --model-root ./models --verify-only
```

### Build the GB10 fork

If Ninja, FFmpeg or libcurl development files are missing, an isolated environment can be created with an existing micromamba installation:

```bash
micromamba create -y -p "$PWD/.work/ninfer-deps" -c conda-forge \
  ninja=1.13.2 ffmpeg=9.0.2 libcurl=8.23.0
export NINFER_FFMPEG_PREFIX="$PWD/.work/ninfer-deps"
export NINFER_LIBCURL_PREFIX="$NINFER_FFMPEG_PREFIX"
export NINFER_NINJA="$NINFER_FFMPEG_PREFIX/bin/ninja"
PKG_CONFIG_PATH="$NINFER_FFMPEG_PREFIX/lib/pkgconfig" /usr/bin/pkg-config --modversion libcurl libavcodec libavformat libavutil
```

These dependencies built the engine on Linux aarch64 with CUDA 13.0, GCC 13.3 and CMake 3.28.3, and were used for the text and image smoke test below. They do not include the NVIDIA driver or CUDA Toolkit. The variables are unnecessary when suitable system development packages are already available.

```bash
git clone https://github.com/baristahaus/ninfer-gb10.git
cd ninfer-gb10
git checkout --detach 4c2fc2f9504d2f373c84e896bf18523906a646d7
cd ..

export NINFER_SRC=$PWD/ninfer-gb10
bash scripts/ninfer_gb10_configure.sh "$NINFER_SRC"
cmake --build "$NINFER_SRC/build-gb10-system-pkg" -j1
```

`MAX_JOBS=1` is an important safeguard on unified-memory systems. Concurrent `cicc` processes can have transient memory use far above the model weights themselves. Do not assume that a sufficient model memory quota also makes JIT compilation safe.

Run outputs default to the Git-ignored `.work/`. Service scripts accept `XL_OUT_DIR`, `NINFER_RUN_DIR`, or `SGLANG_RUN_DIR` to select an output root; examples below use the default.

### Start NInfer and smoke-test text and images

```bash
export NINFER_ARTIFACT=$PWD/models/NVFP4-NInfer/W4A4/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer
export NINFER_MAX_CONTEXT=2048 NINFER_KV_CAPACITY=2048 NINFER_MAX_CONCURRENCY=1
bash scripts/ninfer_gb10_serve.sh start
python scripts/ninfer_openai_smoke.py --max-tokens 128 --output .work/artifacts/smoke-ninfer-server.json
bash scripts/ninfer_gb10_serve.sh stop
```

The default server uses a 32K context, 262144 KV capacity, maximum concurrency 8, DFlash2 draft 7, Vision, and CUDA Graph. Adjust with `NINFER_MAX_CONTEXT`, `NINFER_KV_CAPACITY`, `NINFER_MAX_CONCURRENCY`, and `NINFER_DRAFT_TOKENS`.

On 2026-11-13, the above 2048-token context and KV capacity, concurrency 1, DFlash2 draft 7, Vision and CUDA Graph completed a real GPU smoke test. All six text arithmetic and image recognition checks passed. After stopping, the port and GPU process were released. See the [smoke record](evidence/rerun-2026-11-13-smoke.json).

On 2026-11-15, the previously built fixed-version server and SHA-256-checked 27B weight completed the NInfer concurrency benchmark with the default 32K context, 262144 KV capacity and draft 7. Two rounds each at concurrency 1, 2, 4 and 8 completed 30 requests with no errors; aggregate generation rates were 32.57, 64.79, 98.55 and 140.44 tokens/s, respectively. Twenty-seven requests reached the 512-token limit. The server was stopped afterward. These two-round throughput measurements do not assess answer quality; the four SGLang comparisons were not rerun.

### Reproduce screenshot performance

The smoke commands above stop the service. Restore the default configuration and restart it before the performance comparison:

```bash
unset NINFER_MAX_CONTEXT NINFER_KV_CAPACITY NINFER_MAX_CONCURRENCY
bash scripts/ninfer_gb10_serve.sh start
```

```bash
python scripts/bench_ninfer_concurrency.py \
  --base-url http://127.0.0.1:19410/v1 \
  --engine "NInfer GB10 fork 4c2fc2f" \
  --context-total 262144 \
  --enable-thinking --reasoning-effort xhigh \
  --model qwen3.8-27b-nvfp4-dflash2 \
  --levels 1,2,4,8 --rounds 2 --max-tokens 512 \
  --output .work/artifacts/bench-ninfer-w4a4-dflash2-draft7-xhigh.json
```

Stop the service with:

```bash
bash scripts/ninfer_gb10_serve.sh stop
```

Use the same `NINFER_RUN_DIR` to start and stop (otherwise `XL_OUT_DIR` or `.work/` is used). The script checks an instance identifier before signalling its own processes; it refuses to stop when records are missing, process environments are unreadable, or ownership cannot be established. Records made by older versions do not include an instance identifier; independently identify those processes before managing them. Startup exits if the port is occupied or old records remain in the run directory; check them and use a free port and a fresh run directory. If `NINFER_PORT` changes the port, also pass the corresponding `--base-url` to the smoke and benchmark clients.

### Run four SGLang performance comparisons with the unified script

The historical run used Linux aarch64, Python 3.12.3, CUDA 13.0, PyTorch 2.13.0+cu130, and SGLang 0.5.19; see [`environment.json`](evidence/environment.json). First check for and reuse an isolated environment with these components. If creating one, the following fixed-version example follows the [official SGLang pip installation instructions](https://docs.sglang.io/docs/get-started/install):

```bash
python3.12 -m venv /path/to/sglang-venv
export SGLANG_PYTHON=/path/to/sglang-venv/bin/python
"$SGLANG_PYTHON" -m pip install --upgrade pip
"$SGLANG_PYTHON" -m pip install --pre 'sglang==0.5.19' -r requirements.txt
"$SGLANG_PYTHON" -m pip check
"$SGLANG_PYTHON" -c 'import torch, sglang; print("sglang", sglang.__version__, "torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.is_available())'
```

Replace `/path/to/sglang-venv` with the actual environment path; the GPU check should print `True`. Version 0.5.19 and the experiment record are known, but dependency resolution, drivers, and lower-level components such as FlashInfer vary by installation platform. The clean-environment installation and four service runs above have not been rerun; they are not a fully pinned reproduction of the historical environment. Before running, also verify that NVFP4, FP8, and their respective `DFlash2-FP8` files have been downloaded and checked, and that the service port is available.

Set the model root and Python executable containing SGLang, then run the unified comparison script:

```bash
export SGLANG_MODEL_ROOT=$PWD/models
export SGLANG_PYTHON=/path/to/sglang-venv/bin/python
bash scripts/run_sglang_comparison.sh
```

The script runs four comparisons in sequence: NVFP4 and FP8, each with eager and decode-graph execution. The service and client use port 19410; do not set another `PORT`. Output files with the same names are overwritten. For another run, set `SGLANG_RUN_DIR` to a new empty directory.

Use the same `SGLANG_RUN_DIR` to start and stop SGLang (otherwise `XL_OUT_DIR` or `.work/` is used). Startup refuses an occupied port or a run directory with an existing server record. Stop verifies an instance identifier; older `current.env` records lack it, so identify those processes independently instead of guessing which service to stop.

Only the two eager stages additionally run text and vision smoke tests, 12 quality questions, and functional tests for 2 programming questions.

Results are written to `.work/artifacts/`. Four bench JSON files correspond to the two weight formats and two graph configurations. Smoke, quality, and code-functional JSON files are generated only during eager stages. Aggregate results are also saved as Markdown and JSON:

```text
.work/artifacts/
├── bench-nvfp4-dflash2.json
├── bench-fp8-dflash2.json
├── bench-nvfp4-dflash2-graph.json
├── bench-fp8-dflash2-graph.json
├── smoke-nvfp4-dflash2.json
├── smoke-fp8-dflash2.json
├── quality-nvfp4-dflash2.json
├── quality-fp8-dflash2.json
├── code-functional-nvfp4-dflash2.json
├── code-functional-fp8-dflash2.json
├── comparison-eager.md
├── comparison-eager.json
├── comparison-graph.md
└── comparison-graph.json
```

### Rerun quality and functional checks against an existing service

The following example uses NVFP4. With the service running and `SGLANG_PYTHON` set, first regenerate the raw quality JSON, then produce programming-function results from it:

```bash
"$SGLANG_PYTHON" scripts/quality_probe.py \
  --base-url http://127.0.0.1:19410/v1 \
  --model qwen3.8-27b-coder390-nvfp4-w4a4-dflash2 \
  --output .work/artifacts/quality-nvfp4-dflash2.json

"$SGLANG_PYTHON" scripts/verify_code_functional.py \
  --quality .work/artifacts/quality-nvfp4-dflash2.json \
  --output .work/artifacts/code-functional-nvfp4-dflash2.json
```

The functional script runs only model-generated functions that pass AST allowlist and restricted-builtins checks, in a subprocess with resource limits.

## Documentation and evidence

- [NInfer Port and Concurrency Validation on GB10](../../docs/zh/inference/NInfer在GB10上的移植与并发实测.md) records the official build blocker, fork choice, build, real-run results, screenshot reproduction, the difference from the roughly 22 tok/s metric, acceptance rates, and measurement protocol and generalization limits (unified memory, JIT, KV settings, sampling).
- [SGLang FP8 and NVFP4 inference comparison](../../docs/zh/inference/SGLang的FP8与NVFP4推理对照.md) records comparison design, acceptance rates, and interpretation.
- [`evidence/`](evidence/) contains environment, model checks, per-request performance, quality summaries, and OpenAI responses. Before publication, remove host identifiers and paths without changing values.

The model and engine remain subject to their respective licenses. This is a third-party GB10 port and performance validation; it does not mean NInfer has officially released equivalent GB10 support.
