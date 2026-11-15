# Jev Series Experiment: Laya

This directory contains an on-device evaluation and reproduction setup for `convaiinnovations/laya` (0.3.5).

Laya uses a purpose-built discriminative architecture with two built-in model branches (English-only and multilingual) and a lightweight router. The 20 ticket-classification cases and five proposition-judgment cases used here follow Laya's protocol definitions and are kept strictly aligned across the other three reproduced approaches.

## Files and test responsibilities

- `test_laya.py`: Minimal runnable example corresponding to step three of the tutorial. It sends one input and one question through the full Router path and prints the route and decision.
- `laya-verify.py`: Output-protocol and structure validation. It dumps the full return structure for four question types (choice / score / noul), checks the field contract, and compares latency in Router and direct-Agent modes.
- `laya-latency.py`: Multi-stage latency evaluation. It measures cold-start loading time, pure Python routing overhead, single-question forward latency, marginal throughput when merging 1–16 questions per request, and resident model memory.
- `laya-zh-en.py`: Chinese and English classification and proposition-judgment test set. It checks automatic language-branch routing and measures proposition-judgment confidence in Chinese. It also includes a forced-branch experiment that sends Chinese input directly to the English checkpoint.
- `laya-scale.py`: Extended benchmark for merging questions. It scales from 1 to 256 distinct questions, sampled cyclically from 12 prompt pools covering choice / score / noul, to measure the marginal per-question cost knee and activation-memory growth. It records data only and makes no claim about the cause of any plateau.
- `laya-direct.py`: Compares call paths: an Agent pinned directly to one branch versus automatic Router routing (default branch and all branches resident), measuring latency and resident memory.
- `laya-pin.py`: Compares model modes on the same Chinese input: automatic routing, per-request `model=` overrides (multilingual / forced English / typed-decisions), and `Router(default=)`; it compares decisions, confidence, and routing rationale.
- `laya-common.py`: Local reproduction of the model architecture, token-sequence construction, and confidence algorithm for source-level comparison.

## Running and reproducing

The historical run used Laya 0.3.5 and CUDA PyTorch. First create an isolated Python environment, install a CUDA build of PyTorch compatible with your device, and allow network and disk space for model downloads. The following installs the recorded Laya version; other dependencies are not fully pinned, so reruns depend on the environment actually used.

From this directory, install the dependency and run the tests in order:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Select and install a CUDA build of PyTorch compatible with your device first
python -m pip install -r requirements.txt
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()"

python test_laya.py
python laya-verify.py
python laya-latency.py
python laya-scale.py
python laya-direct.py
python laya-pin.py
python laya-zh-en.py
```

All seven scripts were previously run on an aarch64 GB10 (20 CPU cores, 121 GB unified memory, torch 2.14.0+cu130); result excerpts are in `../evidence/`. `test_laya.py` is the best first script and downloads model weights on its first run. The new-environment commands above have not been verified end to end; transitive dependencies and model revisions are not locked.

**Notes:**

1. **Cold start and memory:** During a full load, Laya initializes the multilingual and English models in sequence. All three checkpoints take about 50 seconds to load and allocate about 5.96 GB in CUDA. The model includes a system-available-memory guard and refuses to load if available memory is too low.
2. **Multilingual weight initialization warning:** The underlying library may warn during multilingual-branch initialization that the temperature coefficient is out of range (confidence is uncalibrated). This is a property of the released weights and does not prevent inference.
3. **Network access in China:** Weights are hosted on Hugging Face. If downloads are interrupted, set `HF_ENDPOINT=https://hf-mirror.com`.

## Measured findings and engineering characteristics

On the tested hardware, Laya shows four notable characteristics:

- **Very fast inference and gains from batching questions:** Single-question forward p50 is 8.31 ms (Chinese, multilingual branch) to 16.41 ms (English branch). Routing itself takes only 0.02–0.19 ms. When questions are merged, marginal time per question falls from 18.66 ms for one question to 3.96 ms for eight and 3.20 ms for 16. From 32 onward it remains around 3.3 ms, while total time grows linearly (845 ms for 256 questions). Batches of 8–16 questions give the largest gain.
- **Small memory footprint, nearly independent of question count:** All three checkpoints resident in CUDA allocate 5.96 GB, with a peak of 6.68 GB. As the batch grows from 1 to 256 questions, the activation peak increment rises only from 712 MB to 1,445 MB. Memory is negligible compared with the latency constraint.
- **Call path affects memory, not latency:** Directly setting `Agent(..., subfolder=…)` pins one branch, with 1.25 GB resident and about 19 seconds to load. Router preloading loads all branches whether `max_loaded=1` or 3, with 4.39 GB resident and about 50 seconds to load. Chinese forward latency is 8.2–8.6 ms for both paths, so Agent saves memory for single-language workloads.
- **Clear language boundary:** Router selects branches by writing system. For Chinese, its routing rationale is “non-Latin script (han, 85% of letters); the English checkpoint cannot read it”. Forcing Chinese into the English checkpoint drops accuracy from 95% to 75%, supporting the need for routing. However, the current open-source version compresses probabilities for Chinese proposition truth judgments (noul): every true sample has probability below 0.051. This limits deep proposition understanding in Chinese.

[中文](README.md)
