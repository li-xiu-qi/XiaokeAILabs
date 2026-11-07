# How fast are diffusion language models? Comparing 1.5B models on DGX Spark

[中文](README.md)

Could a language model answer faster by updating several positions at once? We measured [Fast-dLLM v2 1.5B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B) on a DGX Spark against its autoregressive base, [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). These measurements illustrate the costs of two actual generation paths and why differences in output length and content matter when comparing speed.

## Measured results

In BF16 with a single request and a maximum of 128 generated tokens, each of three English prompts was run three times. The median per-run visible-text throughput was:

| Model | Median | Outputs |
|---|---:|---|
| Fast-dLLM v2 1.5B | 58.13 tokens/s | 122, 65, and 127 returned token IDs for the three prompts; the second included an end token and re-encoded to 64 visible tokens |
| Qwen2.5-1.5B-Instruct | 51.52 tokens/s | All three reached the 128-token limit without an end token |

Throughput is the number of tokens obtained by **re-encoding visible output text**, divided by `generate()` elapsed time. Timing includes prompt prefill and generation, but excludes tokenization and final text decoding. Fast-dLLM ended the code prompt earlier; the outputs differ in length and content. These figures **are not a speedup at equal quality**. Three prompts do not establish general performance; answer quality, time to first token, concurrent throughput, and other hardware were not evaluated.

## Additional smoke test: two MoE models

To observe the costs of sparse activation, we also ran the diffusion model [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) and the autoregressive model [OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct). Both have approximately 7 billion total parameters. LLaDA activates about 1.4 billion non-embedding parameters per token and OLMoE about 1 billion parameters. Similar active parameter counts do not make the architectures, compute, or output quality equivalent.

The models ran in separate BF16, single-request processes with the same three prompts. Each had one warmup and one measured run per prompt: this is a smoke test. Speed again uses re-encoded visible text tokens divided by generation time.

| Prompt | LLaDA-MoE: visible tokens / time | OLMoE: visible tokens / time |
|---|---:|---:|
| Database indexes | 126 / 18.41 s | 128 / 8.01 s |
| Python maximum function | 28 / 18.62 s | 128 / 8.01 s |
| Training versus inference | 126 / 18.48 s | 128 / 8.01 s |

LLaDA-MoE used a low-confidence, fixed-step baseline adapted from its model card: generation length 128, block length 32, and 128 total forward passes without KV cache. It continued through the fixed steps even when an end token appeared early. The code prompt therefore produced only 28 visible tokens while still taking 18.62 seconds. This is not the model authors' accelerated dInfer implementation. OLMoE used autoregressive `generate()` with KV cache and was truncated at the 128-token cap for all three prompts. LLaDA-MoE incorrectly stated in its third answer that training and inference usually use different parameter counts, so these outputs do not establish acceptable answer quality.

The tokenizers, training, decoding implementations, output lengths, and correctness differ. A single run cannot estimate variability. This comparison does not establish an equal-quality speed difference or the inherent superiority of either generation architecture; it should not be combined into a single ranking with the repeated measurements above.

## Two generation paths

![Autoregressive and block diffusion generation paths](docs/generation-paths.svg)

Autoregressive generation usually predicts the next token sequentially and uses a KV cache to avoid recomputing the existing context. Fast-dLLM v2 generates in blocks and iteratively updates multiple positions within a block. It still requires multiple model forward passes and state for each block; parallel position updates do not mean the entire answer is produced in one forward pass.

Here Fast-dLLM used block size 32, small-block size 8, confidence threshold 0.9, and block cache disabled. Qwen2.5 used KV cache and greedy generation. The models also differ in training and generation implementation, so measured differences cannot be attributed to generation style alone.

## Reproduce the comparison

Use Linux, CUDA, and a DGX Spark. The measured environment used Python 3.12, PyTorch 2.14.0+cu130, and Transformers 4.57.3. Fast-dLLM's custom model code is incompatible with the Transformers 5.16.1 interface. Download the models locally first; the scripts use local files only. Inspect the source before running custom model code, and consult the model cards above for downloads and licenses.

```bash
python scripts/benchmark.py --model /path/to/Fast_dLLM_v2_1.5B \
  --kind diffusion --revision 25093b6f63300adfd57f72145083c8a528fe4f16 \
  --warmups 2 --repeats 3 --max-new-tokens 128 --output /tmp/diffusion.json

python scripts/benchmark.py --model /path/to/Qwen2.5-1.5B-Instruct \
  --kind autoregressive --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --warmups 2 --repeats 3 --max-new-tokens 128 --output /tmp/autoregressive.json
```

Run each model in a separate process. After two warmups, the script repeats each prompt three times and records output token IDs, full text, forward-pass counts, elapsed time, and software versions. It is designed for a unified-memory device and checks Linux available memory at startup and during execution, stopping below 50 GiB. `--revision` is recorded metadata; confirm separately that the local weights match it.

[Diffusion run records](docs/diffusion-runs.json) · [Autoregressive run records](docs/autoregressive-runs.json) · [Benchmark script](scripts/benchmark.py)

The MoE smoke tests can be run separately in the same environment. Download each model version first; `--revision` is recorded but does not validate the local weights.

```bash
python scripts/benchmark-llada-moe.py --model /path/to/LLaDA-MoE-7B-A1B-Instruct \
  --revision 783d3467f108d28ac0a78d3e41af16ab05cabd8d --warmups 1 --repeats 1 \
  --steps 128 --gen-length 128 --block-length 32 --output /tmp/llada-moe.json

python scripts/benchmark.py --model /path/to/OLMoE-1B-7B-0924-Instruct \
  --kind autoregressive --revision 7f1c97f440f06ce36705e4f2b843edb5925f4498 \
  --warmups 1 --repeats 1 --max-new-tokens 128 --output /tmp/olmoe.json
```

[LLaDA-MoE run records](docs/llada-moe-smoke.json) · [OLMoE run records](docs/olmoe-smoke.json) · [Fixed-step benchmark script](scripts/benchmark-llada-moe.py)
