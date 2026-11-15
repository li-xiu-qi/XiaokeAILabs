# Benchmarking and Deploying Discriminative Decision Models

This experiment evaluates four mainstream types of open-source typed discriminative decision models and a standard generative LLM on ticket classification and logical proposition judgments. It measures inference latency, memory use, and confidence calibration. For the architecture and findings, see [Evaluation and Fine-Tuning of Discriminative Decision Models](../../docs/zh/decision-models/决策模型架构评测与选型.md).

## Evaluation overview

| Directory | Model and backbone | Core mechanism | Evaluation focus |
|---|---|---|---|
| [laya-dual-branch/](laya-dual-branch/) | Laya (0.3.5) | Dedicated dual-branch discriminative model with built-in English and multilingual routing | Baseline protocol; dual-branch latency and accuracy |
| [metask-logits/](metask-logits/) | metask-jev-4b (Qwen3.5-4B) | Constrains candidate answers to a single token and normalizes output-layer logits directly | Probability calibration and perfect accuracy with single-token logits at 4B parameters |
| [nimble-lora/](nimble-lora/) | Bespoke-Nimble-9B | Base model plus LoRA adapter; weights must be merged offline | Latency and memory cost of a larger 9B model on one local GPU |
| [omni-multimodal/](omni-multimodal/) | Jev-Omni (Gemma 4 12B IT) | Decision model for text, image, audio, and video inputs | Inference latency and probability availability for multimodal inputs |
| [baseline-llm/](baseline-llm/) | Qwen3-VL-4B control | Standard autoregressive generative model | Order-of-magnitude latency difference between generative and discriminative decisions |

## Environment

Each route uses different dependencies and models. Create a separate environment for each subdirectory, and prepare a CUDA device, network access, and storage for model weights as described in its README. Jev-Omni also needs media processing tools and test assets. The scripts include benchmark inputs, so no separate classification dataset is required; see the relevant subdirectory for how to obtain multimodal assets. Historical evaluation settings and download steps are not a complete dependency lock. Verify compatibility with other systems and available memory yourself.

## Guides and evidence

- **[Deployment guide](guides/部署指南.md)**: single-GPU deployment steps, environment configuration, and production considerations for each model.
- **[Reproduction tutorial](guides/复现教程.md)**: model weight preparation, run order, and log mapping for the four models.
- **[Experiment report](../../docs/zh/decision-models/判别式决策模型基准实测.md)**: complete latency and accuracy comparison for 20 classification items and 5 proposition judgments.
- **[Experiment records](evidence/)**: excerpts of run outputs and structured benchmark data supporting the results.

[中文说明](README.md)
