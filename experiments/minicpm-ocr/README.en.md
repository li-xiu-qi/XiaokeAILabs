# MiniCPM-OCR Dual-Model Deployment: 1.0 Parsing + Jev Quality Scoring

[中文](README.md)

This directory contains the experiment code, configuration, and result evidence. For the experiment's purpose, principles, data, results, and scope, see the [full experiment document](../../docs/zh/document-understanding/MiniCPM文档解析与质检.md).

[scripts](scripts/) · [assets](assets/) · [evidence](evidence/)

For the software environment, models, and alternative deployment paths used in the historical run, see [Historical Runtime Conditions and Deployment](#historical-runtime-conditions-and-deployment).

Run all commands below from this experiment directory.

## Environment Setup

The historical deployment used aarch64 Linux, Python 3.12, and a CUDA GPU with BF16 support. It also requires Bash, Git, curl, tar, and enough disk space for models, source code, the browser, and outputs. The steps below are assembled from repository scripts. The full dependencies for the historical OCR, scoring, and rendering environments are not locked, and a clean environment has not been verified end to end.

### OCR Service

Create an isolated environment. Following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), install a CUDA build of torch and matching torchvision for the device in the OCR virtual environment, then run:

```bash
mkdir -p "$HOME/minicpm-ocr"
python3.12 -m venv "$HOME/minicpm-ocr/.venv-ocr"
OCR_PY="$HOME/minicpm-ocr/.venv-ocr/bin/python"
# After installing CUDA-compatible torch and matching torchvision in the OCR_PY environment, install the pinned Transformers source and its dependencies
"$OCR_PY" -m pip install -r requirements-ocr.txt
bash scripts/01_download.sh
bash scripts/02b_fetch_sources.sh
bash scripts/04_opencv.sh
"$OCR_PY" -m pip check
"$OCR_PY" -c 'import torch, torchvision, transformers, cv2, minicpmvocr; print("CUDA:", torch.cuda.is_available(), "Torch:", torch.__version__, "torchvision:", torchvision.__version__, "Transformers:", transformers.__version__)'
```

[requirements-ocr.txt](requirements-ocr.txt) is a candidate OCR dependency list and pins the Transformers source version. Install torch, torchvision, and the SDK source separately as above; unpinned versions may change over time. The SDK installation script uses `--no-deps`. If `pip check` reports missing packages, install them in this environment according to the dependency list in the downloaded SDK source and check again; a successful import is not a complete runtime verification. `01_download.sh` uses `$HOME/minicpm-ocr` by default; set `MINICPM_OCR_HOME` to use a new isolated directory. It refuses to modify existing target files and exits if the recognition model's weight size or SHA-256 differs from the historical record. The model is several GB; the script does not download Jev or DSpark. This check covers only the main model weights and does not pin other file versions.

Parsing and the WebUI require a config file. The command below creates it only if it does not already exist and replaces the two path placeholders in the example. Check the format against the downloaded SDK before use:

```bash
mkdir -p "$HOME/minicpm-ocr/configs"
"$OCR_PY" -c 'from pathlib import Path; import os; p=Path.home()/"minicpm-ocr/configs/spark.yaml"; s=Path("scripts/spark.yaml").read_text(); p.open("x").write(s.replace("${MINICPM_OCR_HOME:-$MINICPM_OCR_HOME}", str(Path.home()/"minicpm-ocr")))'
```

### Jev Scoring and Rendering

Download the complete snapshot from the [MiniCPM-OCR-Jev model page](https://huggingface.co/openbmb/MiniCPM-OCR-Jev) to `$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev`. It must include `model.safetensors`, configuration, and processor files. Record the actual revision and hashes. Scoring reads `score.weight` directly from this file. Scoring and rendering use separate environments; these commands are an installation starting point:

```bash
python3.12 -m venv "$HOME/minicpm-ocr/jev/venv-score"
python3.12 -m venv "$HOME/minicpm-ocr/jev/venv-render"
export JEV_PYTHON="$HOME/minicpm-ocr/jev/venv-score/bin/python"
export RM_RENDER_PYTHON="$HOME/minicpm-ocr/jev/venv-render/bin/python"
export JEV_REPO="$HOME/minicpm-ocr/repo/scripts/minicpm-ocr-jev"
# Following the official PyTorch guide, install CUDA torch and matching torchvision in the JEV_PYTHON environment
"$JEV_PYTHON" -m pip install -r requirements-jev-score.txt
"$RM_RENDER_PYTHON" -m pip install -r requirements-jev-render.txt
"$RM_RENDER_PYTHON" -m playwright install chromium
"$JEV_PYTHON" -m pip check
"$JEV_PYTHON" -c 'import torch, torchvision, transformers; print("Torch:", torch.__version__, "torchvision:", torchvision.__version__, "Transformers:", transformers.__version__)'
"$RM_RENDER_PYTHON" -m pip check
node --version  # The official rendering preflight requires Node.js 18+
test -f "$JEV_REPO/rm_render.py" && test -f "$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev/model.safetensors"
"$JEV_PYTHON" -c 'import os, sys; sys.path.insert(0, os.environ["JEV_REPO"]); import rm_render; rm_render.check_assets(); print("Rendering assets and Node are available")'
```

Use [requirements-jev-score.txt](requirements-jev-score.txt) for scoring and [requirements-jev-render.txt](requirements-jev-render.txt) for rendering; do not merge them. The scoring environment also needs GPU-compatible torch and matching torchvision installed separately. The rendering environment needs Chromium, Node.js, and system libraries. The scorer imports official `rm_render.py` directly, and the rendering process also uses it. Check that this file and its dependencies exist in the current SDK and install missing packages based on actual errors. The browser may also need operating-system libraries and fonts. The Pillow and Playwright versions come from historical records and may not be compatible with current upstream source. Successful imports and `pip check` do not prove rendering, scoring, or GPU inference works; after startup, check the health endpoint and an actual sample.

On 2026-11-15, three official examples with reference renders were scored on an NVIDIA GB10 under Linux ARM64 using Jev revision `a0ed278eb6aa1f333a711665ab610055a4a816ab`. Scores for text, formula and table were 0.00253, 0.96544 and 0.00098; the example records report reference values of 0.00702, 0.96552 and 0.00116. Only one candidate per example was scored; this is not an accuracy evaluation and did not test rendering, the scoring service or the recognition-to-ranking loop. Python 3.12.3, PyTorch 2.15.0+cu132 and Transformers 5.23.0 were used; the copied isolated environment passed `pip check` but was not installed from scratch. Loading reports a missing `lm_head.weight` and an unexpected `score.weight`; this scorer loads the latter separately and uses backbone outputs without the generation head. See the [scoring summary](evidence/jev-score-2026-11-15.json) for results and the weight hash.

Download the pinned model and run the official examples (about 4.4 GB; `JEV_REPO` points to the prepared official source as above):

```bash
"$JEV_PYTHON" -c 'from huggingface_hub import snapshot_download; import os; snapshot_download("openbmb/MiniCPM-OCR-Jev", revision="a0ed278eb6aa1f333a711665ab610055a4a816ab", local_dir=os.path.expanduser("~/minicpm-ocr/models/MiniCPM-OCR-Jev"))'
"$JEV_PYTHON" scripts/jev_score.py --model "$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev" --examples "$JEV_REPO/examples" --json-out jev-examples.json
```

The command overwrites the specified JSON file on reruns; choose a new filename if needed. The official source was downloaded from the then-current main branch without a pinned commit, so examples may change. The model was obtained via a mirror; direct download from the official site was not verified. See below for the subsequent rendering and scoring service test.

On 2026-11-15, the scoring service passed its health check and used a fresh isolated rendering environment to render both candidates in the official text example. The human reference scored 0.99508 and the OCR candidate 0.00723, ranking the reference first; the example records report 0.99501 and 0.00702. The rendering environment used Playwright 1.62.0, Pillow 12.3.0 and Markdown 3.10.3; dependency and Chromium startup checks passed. The scoring environment was copied from an existing environment. One example does not establish accuracy, and the OCR service was not called, so the subsequent single-block recognition-to-ranking test is described below. The example source commit was not pinned; see the [service evidence](evidence/jev-service-2026-11-15.json).

After preparing the model and both environments above, start the service in the foreground (stop it with Ctrl+C):

```bash
"$JEV_PYTHON" scripts/jev_server.py --host 127.0.0.1 --port 18501 --model "$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev"
```

In another terminal with `JEV_PYTHON` and `JEV_REPO` set, check health and submit the official text example:

```bash
curl --fail http://127.0.0.1:18501/healthz
"$JEV_PYTHON" scripts/jev_probe.py --url http://127.0.0.1:18501 --case "$JEV_REPO/examples/01_text_putnam"
```

Check that the port is free first; the service terminal also needs `JEV_REPO` and `RM_RENDER_PYTHON`. Install missing Chromium system libraries and fonts as necessary; rendering may vary with dependencies and fonts.

Later that day, recognition, rendering and scoring were combined on the same official text block. The new OCR output differed character-for-character from the example reference. The reference, OCR output and a reference candidate with its last line removed scored 0.995084, 0.995028 and 0.00000489 respectively; all three were scored and the reference ranked first. The reference came from the upstream example and was not independently checked by a human. A single example does not establish recognition accuracy or ranking stability. Both services and the renderer used the environments above; see the [single-block run summary](evidence/jev-loop-2026-11-15.json).

To rerun the single-block flow, prepare the models and Jev service above and check that ports 18499 and 18501 are free. In another terminal start the OCR service, using the default model location from above:

```bash
export OCR_MODEL_DIR="$HOME/minicpm-ocr/models/MiniCPM-OCR-1.0"
"$HOME/minicpm-ocr/.venv-ocr/bin/transformers" serve "$OCR_MODEL_DIR" \
  --host 127.0.0.1 --port 18499 --device cuda --dtype bfloat16 \
  --chat-template-kwargs '{"enable_thinking": false}'
```

In a third terminal with `OCR_MODEL_DIR`, `JEV_REPO` and `JEV_PYTHON` set, check both services, extract the upstream reference and run the example. Select a separate output directory; the script overwrites files with the same names there:

```bash
curl --fail http://127.0.0.1:18499/health
curl --fail http://127.0.0.1:18501/healthz
export LOOP_OUT="${LOOP_OUT:?Set a separate output directory}"
mkdir -p "$LOOP_OUT"
"$JEV_PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["JEV_REPO"])/"examples/01_text_putnam/case.json"; c=json.loads(p.read_text(encoding="utf-8")); t=next(x["text"] for x in c["candidates"] if x.get("role")=="ground_truth"); pathlib.Path(os.environ["LOOP_OUT"],"reference.txt").write_text(t,encoding="utf-8")'
"$JEV_PYTHON" scripts/jev_loop_block.py --block "$JEV_REPO/examples/01_text_putnam/block.png" --truth "$LOOP_OUT/reference.txt" --max-tokens 512 --out-dir "$LOOP_OUT"
```

When finished, press Ctrl+C in each service terminal. Changes in upstream examples, system fonts and unpinned dependencies can affect the result.

### Single-Image GPU Inference Smoke Test

After installing OCR dependencies and downloading the recognition model, run from this experiment directory:

```bash
"$OCR_PY" scripts/smoke_gpu.py \
  --model-dir "$HOME/minicpm-ocr/models/MiniCPM-OCR-1.0" \
  --image evidence/testpage_cn.jpg
```

Single-image inference was run on Linux aarch64 with an NVIDIA GB10 using Python 3.12, PyTorch 2.15.0+cu132, torchvision 0.29.1+cu130, Transformers 5.22.2, and Pillow 12.0.0. The output included “Test Report on Artificial Intelligence Document Recognition Technology.” An existing PyTorch environment was reused, and image dependencies were added in an isolated directory. This was not a clean-environment installation, and it does not mean the SDK service, layout analysis, Jev scoring, or acceleration comparison was run. Different CUDA combinations should use device-supported PyTorch and torchvision; this single smoke test does not establish general compatibility.

On 2026-11-14, a separate environment ran the same image on GPU and returned the same title, with exit code 0. It used Python 3.12.3, PyTorch 2.15.0+cu132, torchvision 0.29.1+cu130, Transformers 5.23.0, Pillow 12.0.0, and accelerate 1.15.0; `pip check` passed after installation. The environment was copied from an existing dependency set and supplemented, not installed from scratch. The SDK service, layout analysis, and Jev scoring were not run in this check.

Later that day, a separate service environment used `02b_fetch_sources.sh` to prepare the SDK and pinned Transformers source. After adding `jiter`, `sniffio`, `opentelemetry-api`, and `requests`, `pip check` passed. It used PyTorch 2.15.0+cu132, Transformers source commit `549e5b410733fb50e8efbbe4aa585cb89f6478ea`, and MiniCPM-OCR SDK 1.0.5; the SDK source was downloaded from the then-current main branch without a pinned commit. `transformers serve` started on CUDA, passed its health check, and returned HTTP 200 with recognized text for a test image sent to `/v1/chat/completions`. The response was truncated at the 128-token limit, and the service was stopped after the test. This checks the inference service, not SDK layout parsing, WebUI, or Jev scoring; the environment was copied from an existing combination rather than installed from scratch.

## Historical Operation Entry Points and Reproduction Gaps

The service scripts cannot currently verify process ownership, so `minicpm-ocr.sh stop`, `jev_server.sh stop`, and `webui.sh stop` refuse to act. Check process identity first, then manually stop services you started. If a service already responds or a port is occupied, the scripts will not identify it as this experiment's service; check for port conflicts yourself before starting.

This experiment has no complete deployment procedure from a clean environment. These are historical entry points and must not be run sequentially as a fresh setup:

- `02b_fetch_sources.sh` requires `$HOME/minicpm-ocr/.venv-ocr` to be created in advance, with CUDA-compatible PyTorch and other dependencies installed. The repository has no complete dependency list verified end to end.
- `01_download.sh` downloads the OCR and layout models, not Jev or DSpark weights. Before using Jev or the acceleration comparison, prepare the corresponding weights from their sources; not all revisions are pinned.
- `minicpm-ocr.sh` and `webui.sh` use `$HOME/minicpm-ocr/configs/spark.yaml`, which the scripts do not create. The repository's `scripts/spark.yaml` is a historical example. Check its format and model paths against the installed SDK version; do not assume environment variables in it will be expanded.
- The Jev service needs a scoring environment, the official rendering source, and a separate rendering environment. `04_opencv.sh` installs only OpenCV for OCR layout; it does not set up Jev rendering. Set the scoring interpreter with `JEV_PYTHON`, and the renderer with `JEV_REPO` and `RM_RENDER_PYTHON`. There is no verified one-command setup.

The historical environment was aarch64 Linux with CUDA. Other architectures and installation from scratch have not been verified end to end. Models, dependencies, and acceleration runs need additional disk space.

```bash
# 1. Fetch the SDK and weights (via ModelScope)
bash scripts/01_download.sh
bash scripts/02b_fetch_sources.sh

# 2. Start the 1.0 inference service (OpenAI-compatible, port 18499)
bash scripts/minicpm-ocr.sh start

# 3. Start the built-in 1.0 WebUI (port 8950)
bash scripts/webui.sh start

# 4. Install OCR layout dependencies; prepare Jev separately as described above
bash scripts/04_opencv.sh
bash scripts/jev_server.sh start     # Start the scoring service on port 18501 after preparing its environment

# 5. Smoke test: score a single block
"${JEV_PYTHON:-$HOME/xinfer-env/bin/python}" scripts/jev_score.py --examples "$HOME/minicpm-ocr/repo/scripts/minicpm-ocr-jev/examples"

# 6. Closed loop: recognize one block and rank three candidates
python scripts/jev_loop_block.py --block <block image> --truth <human-verified text>

# 7. Compare inference acceleration: benchmark DSpark speculative decoding and feature-compression speedups
bash scripts/run_acceleration_benchmark.sh
```

`02b_fetch_sources.sh` stops if a `repo` or `sources/transformers` directory already exists, to avoid overwriting code. The recognition model download is pinned to ModelScope commit `a165c0943964e167b46e32d67bbcb9deab26a8d4` and checks the historical main-weight hash. The layout model is still downloaded from `master`, and the SDK from `main`; these latter versions may drift, so the full environment cannot yet be strictly reproduced.

Most scripts use `$HOME/minicpm-ocr` by default. Check each script for supported path variables. `01_download.sh` and `02b_fetch_sources.sh` support `MINICPM_OCR_HOME`; other scripts may still use a fixed default directory.

New acceleration-comparison results default to `$HOME/minicpm-ocr/runs/acceleration_benchmark.json`. Set `OCR_BENCHMARK_OUTPUT` to use another location; this command will not overwrite measured evidence already in the repository.

The historical run used separate scoring and rendering virtual environments. Rendering used `playwright 1.62.0` and `Pillow 12.3.0` so changes in rendering dependencies would not affect scores. The scripts do not currently create and install this environment automatically.

### Prerequisites for Other Scripts

`scripts/06_parse.sh` also calls `$HOME/preflight.sh`, which is not in the repository, so it cannot be run directly in a new environment. Use `bash scripts/minicpm-ocr.sh parse /path/to/input.pdf` to call the SDK parsing entry point. `scripts/make_testpage.py` uses Pillow, but its font path is hard-coded to a Windows-mounted path and output goes to the current directory. In a new environment, adjust the font path and confirm the output file does not already exist. `scripts/plot_acceleration.py` additionally requires `python -m pip install matplotlib`; see the script for its input and output paths.

The acceleration benchmark also requires the DSpark model and test images. Set paths with `DSPARK_MODEL_DIR`, `OCR_TESTDATA_DIR`, and `JEV_EXAMPLES_DIR`. The historical DSpark revision is not pinned, and the repository has no verified complete download command. Results are written to `OCR_BENCHMARK_OUTPUT`; specify a new file to avoid overwriting.

## Directory Contents

```
README.md                    This file
scripts/                     Deployment and run scripts, numbered in execution order
  jev_score.py               Jev scorer (Transformers backend, with prompt MD5 assertion)
  jev_server.py              Jev scoring service (18501), /v1/rm/score_best and score_best_batch
  jev_server.sh              Start and stop the scoring service
  jev_ui.html                Scoring page; read by the server on each request, so UI changes need no restart
  jev_probe.py               Compare the official sample's bundled rendering with local re-rendering
  jev_loop_block.py          Closed-loop driver: 1.0 recognizes one block, then Jev ranks three candidates
  benchmark_acceleration.py  Inference acceleration comparison tool (DSpark/speculative decoding vs. baseline)
  run_acceleration_benchmark.sh  Run the inference acceleration comparison in one command
  make_testpage.py           Generate a test page
evidence/                    Test pages, parsing examples, and acceleration benchmark data
  acceleration_benchmark.json Raw inference acceleration metrics (JSON)
```

The scoring page calls `/ui/score` and `/ui/ocr`. The latter calls the 1.0 service and dispatches by content type through the official prompt map (`Text Recognition:` / `Formula Recognition:` / `Table Recognition:`). Scoring itself still works when the 1.0 service is unavailable; the text must then be pasted manually.

## Historical Runtime Conditions and Deployment

This section records the models, environment, and alternative deployment paths used in this experiment to explain its reproduction conditions. For the observed results and their scope, see the [document parsing and quality-checking explanation](../../docs/zh/document-understanding/MiniCPM文档解析与质检.md).

| Item | Configuration |
|---|---|
| Recognition model | `openbmb/MiniCPM-OCR-1.0`, Apache-2.0, weight SHA-256 `83f4e433...` |
| Discriminator model | `openbmb/MiniCPM-OCR-Jev`, Apache-2.0, 4.13 GiB |
| Layout model | `PaddlePaddle/PP-DocLayoutV3_safetensors`, 127 MB, torch provider |
| SDK | `github.com/OpenBMB/MiniCPM-OCR`, includes `minicpmvocr` and its built-in WebUI |
| Transformers | Upstream commit `549e5b4` (at the time, PyPI 5.21.0 did not yet include the `minicpm_ocr` architecture) |
| Device | GB10, aarch64, sm_121, 121 GiB unified memory, 273 GB/s bandwidth |
| Software environment | Ubuntu 24.04, CUDA 13.0.88, PyTorch 2.14.0+cu130, Python 3.12 |
| Download channel | ModelScope |

The vLLM, SGLang, and Paddle installation paths attempted in this run did not work; this does not mean they are unavailable in every aarch64 environment.

| Dependency | Issue encountered in this run | Path used in this run |
|---|---|---|
| vLLM / SGLang | Installation scripts check for x86_64; Docker Hub was unreachable | `transformers serve`, with an OpenAI-compatible interface |
| Paddle layout detection | No linux aarch64 wheel on PyPI | The SDK's built-in pure-torch layout provider |
| Official Jev serve | Depends on the vLLM RM interface | Use Transformers to get the final-token hidden state from the Qwen3_5 backbone, multiply it by `score.weight` to compute logits, and add an MD5 assertion to the prompt to detect drift |

Other historical results and commands are in the [experiment entry point](README.md) and the [acceleration comparison report](../../docs/en/document-understanding/minicpm-ocr-speculative-decoding-benchmark.md). The model and SDK download scripts do not pin all versions, so strict reproduction still requires checking sources and hashes.
