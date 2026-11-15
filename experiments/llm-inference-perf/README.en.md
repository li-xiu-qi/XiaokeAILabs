# llm-inference-perf: Local LLM Inference Performance Experiment

This experiment measures speculative decoding on a GB10 across a 27B dense model, 125B MoE, and 35B MoE, using three prompts and several draft configurations. Results and applicability are discussed in [Speculative Decoding Gains and Measured Conditions](../../docs/zh/inference/投机解码收益与实测条件.md) (Chinese). An engine and 1-slot vs 8-slot concurrency comparison on the same device is in [engine-slot-concurrency](../engine-slot-concurrency/README.en.md). The two experiments use different protocols and prompts, so their results must not be combined directly. Read [中文](README.md).

## Related experiments

The three experiments focus on language-model generation mechanisms, training, and inference performance:

- [`llm-mechanics`](../llm-mechanics/README.en.md) explains tokenization, sampling, KV cache, and quantization at individual-step scale. Its guide, [Language-Model Generation Mechanics and KV Cache](../../docs/zh/inference/语言模型生成机制与KV缓存.md), is in Chinese.
- [`llm-lifecycle-fullchain`](../llm-lifecycle-fullchain/README.en.md) demonstrates a small model's path from tokenization and pretraining through alignment. Its guide, [Building and Training an LLM from Scratch](../../docs/zh/inference/大语言模型从零构建与训练全链路.md), is in Chinese.
- This experiment measures speculative decoding configurations for three models and three prompts on GB10. Its guide is [Speculative Decoding Gains and Measured Conditions](../../docs/zh/inference/投机解码收益与实测条件.md) (Chinese).

[scripts](scripts/) · [configs](configs/) · [log](log/) · [tasks](tasks/)

Run all commands below from this experiment directory.

## Environment setup

- **Hardware:** The report records an NVIDIA DGX Spark (GB10, 128 GB unified memory). Actual memory needs vary with model, quantization, context length, and cache settings. Performance from the report does not transfer directly to other devices.
- **System and tools:** The launch and matrix scripts use Bash. The matrix script also needs `curl`, `grep`, and Python 3. `bench.py` and the result parser use only the Python standard library; no third-party Python packages are needed for the benchmark client.
- **Inference server:** Prepare a CUDA-enabled `llama-server` and confirm that it supports the selected configuration's arguments and speculative-decoding mode. The reported version is llama.cpp commit [`a0e6dc7`](https://github.com/ggml-org/llama.cpp/commit/a0e6dc7). See the [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
- **Models:** Prepare GGUF main weights for the selected configuration. MTP and DFlash2 may also require draft weights compatible with the main model. The default Flash-Next main model is split into three files: place all shards in the same directory and pass the first shard to the server. Check model licenses, disk space, and device memory before running.

These public download pages correspond to some filenames in the scripts and may be useful when preparing weights. The original run did not record download sources or hashes; matching filenames do not prove the weights are identical:

- Flash-Next IQ4_XS main model: [three-shard files](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ4_XS).
- Flash-Next Q8_0 MTP draft head: [MTP files](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/MTP).
- 27B Q4_K_M DFlash2 draft model: [model files](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main). Prepare the 27B main model and other configurations separately.

For example, install the optional download tool in this experiment directory and download the three Flash-Next main-model shards plus the MTP draft head into a new directory. Downloads are large; check storage and memory first. Neither source is pinned to a revision, so record SHA-256 after downloading:

```bash
python -m pip install -r requirements.txt
export MODEL_ROOT=/path/to/new-models
hf download bartowski/Qwen3.8-Flash-Next-GGUF --include 'Qwen3.8-Flash-Next-IQ4_XS/*' --local-dir "$MODEL_ROOT/flash-next"
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --local-dir "$MODEL_ROOT/flash-next-mtp"
sha256sum "$MODEL_ROOT"/flash-next/Qwen3.8-Flash-Next-IQ4_XS/*.gguf "$MODEL_ROOT"/flash-next-mtp/MTP/*.gguf
```

[requirements.txt](requirements.txt) contains only the optional download tool and does not pin its version. It is not needed by the standard-library benchmark client. These public model sources are suggestions, not confirmed matches for historical weights, and compatibility in a fresh environment has not been verified.

After installing the NVIDIA driver, CUDA Toolkit, C/C++ compiler, CMake, and Git, you can try building the CUDA server at the recorded llama.cpp commit. Run in a new working directory:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout --detach a0e6dc7
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server -j 4
./build/bin/llama-server --version
./build/bin/llama-server --help
```

The original build command, CUDA version, full OS version, and weight hashes were not recorded. The build and script commands below are rerun entry points, but they are not enough to strictly reproduce the full reported experiment. After building, confirm that the server supports the speculative arguments in the configs. Changing versions or weights requires rechecking compatibility and remeasuring results.

## Run a request

After building, return to `experiments/llm-inference-perf`, replace paths with prepared files, and set these variables in the terminal that will run the server:

```bash
export LLAMA_BIN=/path/to/llama-server
export MODEL_DIR="$MODEL_ROOT/flash-next/Qwen3.8-Flash-Next-IQ4_XS"
export MAIN_MODEL=Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf
export DRAFT_MODEL="$MODEL_ROOT/flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
```

`MAIN_MODEL` is a filename under `MODEL_DIR`; `DRAFT_MODEL` is the draft-file path. Baseline mode does not need a draft model. These are placeholders and do not download weights. The default `LLAMA_BIN` differs between `start-server.sh` and `run-matrix.sh`; explicitly set the actual server path for either script.

Start one server mode at a time; stop the current service before switching modes:

```bash
bash scripts/start-server.sh base      # Baseline, no speculative decoding
bash scripts/start-server.sh mtp       # MTP
bash scripts/start-server.sh dflash    # DFlash2
```

When the service is ready, send a test request from another terminal:

```bash
python3 scripts/bench.py --host 127.0.0.1 --port 8835 --mode completion \
  --n-predict 400 --ignore-eos --prompt-file tasks/T1-自由生成.txt
```

This sends one request for one prompt; it is not the full matrix. The script defaults to the Flash-Next main model and its matching MTP draft head. For DFlash2, change `MODEL_DIR`, `MAIN_MODEL`, and `DRAFT_MODEL` to a compatible main/draft pair; changing only the server mode is insufficient. `PORT`, `CTX`, `SPEC_N_MAX`, and `SPEC_P_MIN` adjust the port, context length, and speculative parameters. The server script defaults to a 262144-token context and listens on `0.0.0.0`; this may use substantial memory and allow connections from other devices. Set `CTX` for the available memory and run on a trusted network or behind a controlled firewall. See `configs/` for per-configuration parameters; defaults are not the full experiment matrix.

## Batch matrix entry point

`scripts/run-matrix.sh` uses this experiment's `scripts/bench.py` and `tasks/`. It launches the server configuration by configuration, then runs three prompts with three repetitions each. Example (replace the model path, filename, and server binary for your environment):

```bash
LLAMA_BIN=/path/to/llama-server \
  bash scripts/run-matrix.sh 35b /path/to/model model.gguf 32768 configs/matrix-35b.cfg
```

The script stops only the server it started; check that the port is not already used by another process. Output defaults to the Git-ignored `.work/matrix-35b/`; set `XL_OUT_DIR` to choose another location. It refuses to overwrite existing files. With `XL_REQUIRE_EXTERNAL_ENV=1`, `XL_OUT_DIR` must be set explicitly. `PYTHON` can select a Python interpreter. `$HOME` in config files expands to the running user's home directory. Main and draft model paths must already exist. Config parameters are split on spaces, so draft-model paths must not contain spaces. The 35B MTP profile uses a draft head built into the model, unlike the external draft-head example above. On 2026-11-15, this 35B matrix completed on GB10: five configurations, three tasks and three runs each, for 45 successful requests. The Qwen3.6-35B-A3B UD-Q4_K_M weight had SHA-256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`. The CUDA server was built from the llama.cpp `ccc3646` source archive with CUDA Toolkit 13.0.88; its embedded Git commit was not recorded. Baseline mean generation rates across the three tasks were 67.84, 66.49 and 66.16 tokens/s; MTP rates were 85.37, 89.57 and 99.79 tokens/s. N-gram results varied by task. Historical weight hashes and complete build options are unavailable, so identical historical binaries and weights cannot be established; three runs do not establish a general speedup.

After a matrix run, inspect the summary (replace the path if you set `XL_OUT_DIR`):

```bash
python3 scripts/parse_matrix.py .work/matrix-35b/matrix_35b.txt
```

The summary's `n` counts successfully parsed runs. The parser skips malformed or incomplete records, so check that each group has three runs and inspect the raw log. In the historical records, repeated requests can reuse the prompt cache, making the `prefill` mean combine runs with different actual prefill token counts. Do not compare those means as if the same amount of input was processed; check `prefill_tokens` in the raw records.

## Results

Analysis is in [Speculative Decoding Gains and Measured Conditions](../../docs/zh/inference/投机解码收益与实测条件.md) (Chinese). Three-run raw records are in `log/`, with corresponding parameters in `configs/`.
