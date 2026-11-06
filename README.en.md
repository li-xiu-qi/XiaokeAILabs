# XiaokeAILabs: AI Engineering Experiments

[中文](README.md) · [Experiment website](https://li-xiu-qi.github.io/XiaokeAILabs/)

From model training and retrieval to multimodal systems and inference deployment, we use inspectable methods, code, and results to explain engineering trade-offs. Tutorials, underlying principles, measurements, and their limits help readers decide what applies and test it on their own tasks. This open-source project continues to evolve.

## Experiments

- [Embedding distillation](experiments/embedding-distillation/README.md): Transfer teacher scores to a smaller embedding model; candidate preparation scripts are included.
- [Reranker distillation](experiments/reranker-distillation/README.md): Compare supervised training and score distillation on a fixed candidate pool.
- [ColBERT distillation](experiments/colbert-distillation/README.md): Compare score distillation with supervised learning; distillation did not beat the supervised baseline.
- [Multimodal embedding distillation](experiments/multimodal-embedding-distillation/README.md): Explore knowledge transfer from a vision-language retrieval teacher to a compact student.
- [Multimodal reranking](experiments/multimodal-reranking/README.md): Explore reranking image-text candidates with anonymized evaluation results.
- [GUI grounding fine-tuning](experiments/gui-grounding-finetuning/README.md): Train screenshot-and-instruction grounding; 30 held-out samples do not establish generalization gains.
- [Image generation with synthetic training data](experiments/image-generation-distillation/README.md): Train a student LoRA on teacher-generated data; results rely on one AI reviewer.
- [Vector database benchmark](experiments/vector-db-benchmark/README.md): Historical records lack raw results and do not support product rankings.
- [Diffusion language model speed](experiments/diffusion-language-models/README.md): Compare generation paths on three prompts with unequal output lengths and unmeasured quality.
- [LLM inference performance](experiments/llm-inference-perf/README.md): Performance analysis across models and inference configurations.
- [Language model quantization](experiments/llm-mechanics/quantization/README.md): Compare model size, perplexity, and CPU speed; the BF16 path performed unusually slowly.
- [LLM mechanics](experiments/llm-mechanics/README.md): Understand text generation through examples.
- [Inference cost models](experiments/inference-cost-models/README.md): Memory examples and theoretical models; performance estimates have not been independently validated.
- [Data structure costs](experiments/data-structure-costs/README.md): Analyze the costs of data structures and formats.
- [Portrait matting](experiments/portrait-matting/README.md): Evaluate subject regions, boundaries, and forward time on 30 still images; results do not establish video performance.
- [MiniCPM-OCR](experiments/minicpm-ocr/README.md): Document recognition deployment and generation acceleration.
- [Jina CLIP OpenVINO](experiments/jina-clip-openvino/README.md): Compare CPU and NPU correctness and workload-specific performance.
- [Jev open-source evaluation](experiments/jev-open-source/README.md): Compare community implementations on a small decision-making sample.

See each experiment for its environment, datasets, model licenses, and reproduction limits. Some experiments require external data, models, or preparation files that are not yet published; reported measurements apply only under their stated conditions.

## License

Repository code is available under [Apache 2.0](LICENSE); third-party models, data, and code retain their own licenses.
