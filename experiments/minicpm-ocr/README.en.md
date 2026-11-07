# MiniCPM-OCR deployment: document parsing and Jev quality scoring

[中文](README.md)

Can two OpenBMB MiniCPM-OCR models run together on an aarch64 unified-memory machine without installing vLLM or building Paddle? This experiment connects MiniCPM-OCR 1.0, which parses images into Markdown, to Jev, which scores the consistency of the source image and a candidate transcription. It also compares ordinary greedy generation with a DSpark speculative decoding implementation on four samples.

In this GB10 environment, the attempted vLLM, SGLang, and Paddle installation routes did not run; a Transformers-based route ran both models. Scores on three official sample blocks differed from published values by at most 0.0045. One block-level recognition and three-candidate scoring loop took 2.5 seconds. These observations do not establish broad OCR accuracy or production quality; the limits below matter.

## Why score an OCR result?

Manual review can be expensive, while character error rate needs a labeled reference. Jev explores a different approach: given a source image, candidate text, and an image rendered from that text, it returns a consistency score between 0 and 1 without a ground-truth transcription. The score may help prioritize candidates or flag results for review, but its ranking accuracy and operational usefulness require evaluation against human-labeled data.

The two models play different roles:

```text
source image → [MiniCPM-OCR 1.0] → text → [renderer] → rendered image
      └────────────── [Jev: image + text + rendered image] → consistency score
```

| Model | Role | Input | Output | Reported size |
|---|---|---|---|---|
| MiniCPM-OCR-1.0 | Document parsing | Image or PDF | Markdown with text, tables, formulas and reading order | 1.3B, SigLIP + Qwen3.5 |
| MiniCPM-OCR-Jev | Consistency assessment | Source image, text, and rendered text image | Two logits for consistent / inconsistent | 2B backbone plus a `[2,2048]` scoring head |

Jev is a scoring stage, not another version of the recognizer. Its weights do not include an `lm_head`; it does not generate the transcription.

## Environment and reported observations

The documented run used DGX Spark GB10, aarch64, unified memory, Ubuntu 24.04, CUDA 13.0.88, PyTorch 2.14.0+cu130, Python 3.12, and a Transformers source snapshot at commit `549e5b4`. The recognizer and scorer were `openbmb/MiniCPM-OCR-1.0` and `openbmb/MiniCPM-OCR-Jev`; the layout model was `PaddlePaddle/PP-DocLayoutV3_safetensors` via a Torch provider. See the [Chinese experiment record](README.md) for the original environment and deployment notes.

The recorded examples include an A4 end-to-end parsing run of 11.2 seconds, a WebUI run of 9.75 seconds, and a reported 3 GiB recognizer peak. Jev weight loading took 32.5 seconds, its reported peak was 5.18 GiB, and individual candidate forward times ranged from 0.13 to 1.16 seconds. These are individual observations rather than a latency distribution.

For one official text block, the documented three-candidate loop scored a human-confirmed transcription 0.99522, the recognizer output 0.99505, and a deliberately corrupted version 0.00000. Recognition, rendering, and ranking three candidates took 2.5 seconds in that run. The source image, candidate texts, rendered images and per-candidate score logs for this historical example are not retained in this repository, so its ranking and timing cannot be independently checked here. This small example is not an estimate of ranking accuracy on real documents.

## DSpark speculative generation

The [benchmark](scripts/benchmark_acceleration.py) compared baseline greedy autoregressive decoding with MiniCPM-OCR-1.0-DSpark on four document blocks: a Chinese mixed-layout sample, a mathematical proof, a multiline formula, and a structured table. The measured model-generation interval excludes image reading and preprocessing. Both paths generated the same 343 tokens across these four samples.

| Metric | Baseline | DSpark |
|---|---:|---:|
| Total model-generation time | 4.6461 s | 1.9676 s |
| Aggregate throughput | 73.83 tokens/s | 174.32 tokens/s |
| Target-model forward steps | 343 | 44 |
| Greedy output token parity | 343 / 343 | 343 / 343 |

The aggregate time ratio is **2.36×**, with per-sample ratios of 2.00× to 3.08×. The arithmetic mean of the four per-sample accepted-length averages is 9.18 tokens per step. The measured PyTorch peak allocation with both models loaded was 4.33 GiB; baseline peak was not measured separately, so this does not show the draft model's incremental cost. The timing and equality results cover only these four cases, not end-to-end OCR or all documents. Full per-sample metrics are in the [recorded JSON](docs/evidence/acceleration_benchmark.json) and [benchmark report](docs/evidence/acceleration_benchmark.md).

The DSpark run uses a 0.3B five-layer draft model with `block_size=15`. The implementation records per-position recurrent state snapshots within the verification window (`seq_len <= 16`) and trims `conv_states` and `recurrent_states` after verification, addressing the rollback requirement of the backbone's linear-attention layers. The sample token-equality check does not prove equivalence for every possible input or configuration.

## Deployment choices

The attempted official vLLM, SGLang, and Paddle installation paths did not work in this particular environment; that does not mean these packages are unavailable on all aarch64 systems. The experiment used the following alternatives:

| Component | Local obstacle | Alternative used here |
|---|---|---|
| vLLM / SGLang | Installation scripts checked for x86_64; Docker Hub was unreachable | `transformers serve` with an OpenAI-compatible interface |
| Paddle layout detection | No Linux aarch64 wheel on PyPI in the tested setup | SDK's Torch layout provider |
| Official Jev serving | Depends on vLLM reward-model interface | Transformers backbone, final-token hidden state and `score.weight`, with a prompt MD5 assertion |

## Run the workflow

The documented prerequisites are Linux on aarch64 or x86_64, CUDA, and approximately 12 GB of disk for the environment and two sets of model weights. Only the GB10 aarch64 setup was tested; the x86_64 route was not verified.

```bash
# Download the SDK and model weights through ModelScope.
bash scripts/01_download.sh
bash scripts/02b_fetch_sources.sh

# Start the OCR service and the bundled WebUI.
bash scripts/minicpm-ocr.sh start
bash scripts/webui.sh start

# Prepare the separate rendering environment and start Jev scoring.
bash scripts/04_opencv.sh
bash scripts/jev_server.sh start

# Score an official block, then run recognition and three-candidate ranking.
python scripts/jev_score.py --examples <SDK>/examples
python scripts/jev_loop_block.py --block <block-image> --truth <verified-text>

# Compare baseline and DSpark generation.
bash scripts/run_acceleration_benchmark.sh
```

The default paths are derived from `$HOME/minicpm-ocr`; `MINICPM_OCR_HOME`, `OCR_MODEL_DIR`, `JEV_MODEL_DIR`, `JEV_REPO`, `RM_RENDER_PYTHON`, and `OCR_URL` can override them. Rendering and scoring use separate environments. The renderer pins Playwright 1.62.0 and Pillow 12.3.0 because changing image rendering can change scores. `02b_fetch_sources.sh` stops if its destination or download already exists instead of overwriting it.

The scoring UI uses `/ui/score` and `/ui/ocr`; the latter calls the recognizer with the relevant text, formula, or table prompt. Scoring can still run if the recognizer service is down, with manually supplied text. If all candidate renders fail, the scoring endpoint returns 422 rather than selecting a winner.

## Limits of the score

Jev was checked against three official cropped blocks here; scans, photographs, mixed Chinese and English documents, and cross-machine reproduction were not tested. It expects layout blocks, not whole pages. A whole-page workflow needs layout detection and per-block scoring.

The model's idea of consistency reflects its training labels. A historical record reports a 0.047 score for a human-confirmed table transcription, but the underlying example is not retained here; rendering differences may affect scores, but its cause is not established. A high score means the text and its rendering appear consistent with the image to the model; it does not guarantee semantic correctness. Missing fonts or formula rendering errors can also distort scores, so inspect the source and rendered images side by side. Candidate scores are for comparison within the same group and must not be interpreted as accuracy.

The score is the positive-class softmax of two logits, not a validated probability of correctness. To choose thresholds, use independently labeled documents and report false acceptance, missed errors and review workload on a held-out test set. Calibration would require a separate reliability assessment; no general threshold is established here.
