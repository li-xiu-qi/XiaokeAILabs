# LLM mechanics: from tokenization to decoding

[中文](README.md)

How does text become the next generated token, and what work can an inference engine reuse? This section follows that path through small code examples. The examples have different dependencies and model-weight requirements. Their purpose is to explain what each step computes; quantitative resource estimates are covered separately in [inference cost models](../inference-cost-models/README.en.md).

## A reading path

| Stage | Guide | What to learn |
|---|---|---|
| Model weights | [Quantization](quantization/README.en.md) | Compare Qwen3-0.6B BF16, Q4_K_M and Q8_0 file size, small-sample perplexity and CPU speed. |
| Input | [Tokenizer mechanics](../tokenizer-mechanics/README.en.md) | Explore a byte-level BPE toy implementation, real encoders, token counts and billing. |
| Output | [`logit_and_sampling/`](logit_and_sampling/) | Examine logits, softmax, greedy argmax, temperature, top-k and multi-step generation. The existing examples are notebooks and have their own dependencies. |
| Reuse during generation | [KV cache](kv_cache/README.en.md) | Compare cached inference numerically with recomputing the entire prefix. |

One path through the material is to load a model, tokenize input text into IDs, obtain logits for the next token, select an ID with greedy decoding or sampling, and repeat. A KV cache stores previous attention keys and values so the model need not recalculate those projections from scratch at each step. Cache use still involves attention over previous keys; it does not eliminate all work proportional to context length.

## What each example establishes

- The tokenizer tutorial's byte-level BPE is a teaching implementation; real model tokenizers also depend on their own preprocessing, vocabulary, and special-token rules. The legacy `tokenizer/` directory contains older examples with separate dependencies.
- `logit_and_sampling/test_llm_logit.ipynb` demonstrates softmax and top-k inspection; `logit_and_sampling/Qwen2.5模型预测机制分析.ipynb` explores argmax, temperature, top-k, and multi-step generation. These are historical notebooks, not a single automated benchmark.
- The KV cache example uses a randomly initialized tiny model and compares logits with and without cache. It manually passes the framework's cache; it does not implement an attention kernel from scratch or measure speed.
- The quantization comparison uses identical Qwen3-0.6B weights. Its BF16 CPU path was unusually slow, so its speed differences cannot be generalized to quantization or other hardware.

## Requirements

The KV cache example uses a random model and downloads no weights. The tokenizer's toy byte-level BPE only needs Python; comparing real encoders requires `tiktoken`. The quantization guide specifies its model and tool dependencies. Historical notebooks and other scripts have their own dependencies and weight requirements; inspect them before running.
