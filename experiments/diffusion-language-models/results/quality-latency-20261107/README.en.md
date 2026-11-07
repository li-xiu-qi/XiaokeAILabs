# Short-answer quality and generation time on DGX Spark

[中文](README.md) · 2026-11-07

How do four models answer the same short questions on one device? This diagnostic records both answers and generation time so that latency can be interpreted alongside the actual output. Eight simple arithmetic and string tasks, repeated three times each, are far too few to measure general model capability.

| Model | Strict match | Whitespace-insensitive match | Median generation time |
| --- | ---: | ---: | ---: |
| Fast-dLLM v2 1.5B | 15/24 | 18/24 | 0.082 s |
| Qwen2.5 1.5B Instruct | 15/24 | 18/24 | 0.061 s |
| LLaDA-MoE 7B-A1B Instruct | 18/24 | 18/24 | 18.432 s |
| OLMoE 1B-7B Instruct | 3/24 | 3/24 | 0.709 s |

The three repetitions reuse each question: they are not 24 independent tasks. Strict matching ignores case and surrounding whitespace. Matching after removal of all whitespace was added after inspecting outputs as a sensitivity analysis; it does not extract an answer from a sentence. These scores measure compliance with the requested format as well as correctness. For example, `2, 5, 9` fails strict matching because of spaces; OLMoE sometimes includes a correct answer in an explanatory sentence and still fails exact matching. Its comparison of 3.8 and 3.12 is actually wrong. See [tasks.json](tasks.json) for questions and keys, and [results.json](results.json) and [per-run.csv](per-run.csv) for individual outputs and scores.

## Interpreting time

Each model ran serially in its own process on NVIDIA GB10 using BF16, two warmups, three repetitions per task, and an output budget of 128 model tokens. Timing covers the synchronized generation process, excluding tokenization and text decoding; it is not time to first token. The median includes all 24 generations for a model.

LLaDA used 128 steps, generation length 128 and block length 32; every recorded returned sequence has 128 tokens. Most other responses contain only a few returned tokens and may stop at an end token. Fast-dLLM used block size 32, small block size 8, threshold 0.9 and block cache disabled. Models also differ in training, tokenizer, parameter size, implementation, caching and decoding. This is neither an equal-length, equal-quality speed comparison nor a causal estimate of diffusion versus autoregressive architecture.

The environment used PyTorch 2.14.0+cu130 and Transformers 4.57.3. All stages exited with code 0. The minimum observed available system memory for the four stages was 89.154, 89.083, 76.866 and 77.710 GiB respectively; a 50 GiB guard did not trigger. Existing ComfyUI was left running and its queue was empty before execution. Revisions are supplied run labels, not independent verification of weight contents.

## Recompute and reproduce

Run `python recompute.py` in this directory to check each answer against the task key and recompute strict matching, whitespace sensitivity and median generation time. The parent experiment provides `scripts/quality_tasks.json`, `scripts/benchmark.py`, `scripts/benchmark-llada-moe.py` and `docs/质量与延迟对照方案.md`; model files and versions must be prepared separately. Answers and timings can vary across runs. Long answers, end-to-end serving latency and cross-run stability were not evaluated here.
