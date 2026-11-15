# Jev Series Experiment: Nimble

This directory contains an on-device evaluation and reproduction setup for the `bespokelabsai/nimble` (Bespoke-Nimble-9B) approach.

Nimble is distributed as a LoRA adapter plus a version-pinned base model, so the weights must be merged before deployment. It has the largest parameter count among the four evaluated approaches, as well as the highest per-field latency and memory use. This README records its measured behavior on a local single GPU, deployment steps, and engineering constraints specific to this route.

## Files and test responsibilities

- `nimble-deploy.py`: Weight-merging script. Downloads the adapter and base weights, validates the prompt contract, merges PEFT weights, and writes the complete weights and configuration to the user cache directory.
- `nimble-bench.py`: Inference performance and task-accuracy benchmark covering input/output contract validation, single-field latency, and a Chinese business scenario.
- `nimble-cuda_scorer.py`: Archived copy of the upstream CUDA per-field scorer, which runs a forward pass for each field in sequence on CUDA.

## Deployment and reproduction

Nimble has an additional setup step compared with the other three approaches because its inference implementation lives in a separate Nimble code repository, not here.

Run these commands from this directory.

You need Linux, a CUDA GPU, Python, and Git. Create an isolated environment and install a CUDA build of PyTorch compatible with your device. The deployment script also depends on `transformers`, `peft`, `huggingface_hub`, and `accelerate`. Versions recorded by the historical scripts are PyTorch 2.14.0+cu130, Transformers 5.16.1, and PEFT 0.20.0. The repository has no complete dependency lock file, and upstream code and adapter revisions are not pinned; record versions and check compatibility before rerunning. Merging is done on CPU and requires room for both the base and merged weights (about 18 GB each), in addition to download and save space.

```bash
python3 -m venv .venv
source .venv/bin/activate
# First select and install a CUDA build of PyTorch compatible with your device
python -m pip install -r requirements.txt
```

```bash
# 1. Clone the upstream code repository; both bench and deploy import the scorer from it
git clone https://github.com/bespokelabsai/nimble.git ~/nimble

# 2. Merge the weights
python nimble-deploy.py

# 3. Run the benchmark
python nimble-bench.py
```

Step 1 is required. Both `nimble-bench.py` and `nimble-deploy.py` put `~/nimble` first on the module search path and import the upstream scorer from there. The local `nimble-cuda_scorer.py` is an archive copy and is not used at runtime. Without the clone, deployment fails to find a file while validating the prompt contract, and the benchmark cannot find the module.

Step 2 merges the weights. Nimble is distributed as a LoRA adapter plus a version-pinned base; they must be combined into complete weights for inference. `nimble-deploy.py` prints the duration of each stage. In this run, the adapter took 10.8 s, base weights 488.7 s, merging 5.2 s, and saving 4 s, for about 8.5 minutes total. The 18 GB base download is the main cost and still takes several minutes through a mirror.

The merged weights are saved under the user cache, at a path like `~/.cache/models/nimble-9b-<commit-hash>`, and the path is written to a configuration file. `nimble-bench.py` reads that configuration; no manual path entry is needed.

After the benchmark, compare your results with excerpts No. 05 and 06 in `../evidence/`. Latency and memory usage depend on the device, software versions, and concurrent workload; verify the model and run configuration before interpreting differences.

## Two pitfalls of this route

First, merging emits a PEFT warning that some adapter parameters are unrecognized. It can be ignored: the adapter was saved with a newer PEFT version, and extra parameters are skipped without affecting the merge result.

Second, this approach uses the most memory: 17.57 GB resident. Budget for it before keeping it loaded alongside other models.

## Measured metrics

Measurements from the tested local environment:

- **End-to-end latency:** Median decision latency (two fields) of 365.33 ms (English) and 366.66 ms (Chinese input), the slowest of the four approaches.
- **Chinese support:** Upstream declares English-only support. We tested just one Chinese input: latency matched English and fields returned scores, but no Chinese test set was run, so accuracy is unknown.
- **Memory:** About 17.57 GB resident, the highest of the four approaches.

## A key structural difference

The CUDA path in Nimble does not parallelize fields. The upstream README distinguishes it clearly: on Mac, `ParallelScorer` shares one forward pass and scores all fields together in parallel; the CUDA scorer makes a separate pass for each field, each time including the full prompt. Cost therefore grows approximately linearly with the number of fields. Our 365.33 ms measurement is for two fields; ten fields would be a different cost level. This is not the same as Laya amortizing one request across multiple questions, and comparing only single-field numbers would be misleading.

It can work when a workload always has two or three fields. Compared with metask-jev-4B at 133.7 ms and 8.58 GB, however, it shows no advantage.

[中文](README.md)
