# XiaokeAILabs: AI Engineering Experiments

[中文](README.md) · [Experiment website](https://li-xiu-qi.github.io/XiaokeAILabs/)

From model training and retrieval to multimodal systems and inference deployment, we use inspectable methods, code, and results to explain engineering trade-offs. Tutorials, underlying principles, measurements, and their limits help readers decide what applies and test it on their own tasks. This open-source project continues to evolve.

## Experiments

- [Chinese embedding fine-tuning](experiments/embedding-research/model_finetuning/test_train_embedding/README.md): Learn in-batch contrastive training with query-document pairs; the four built-in examples do not establish a retrieval improvement.
- [Reranker fine-tuning](experiments/reranker-finetune-scifact/README.en.md): Train a cross-encoder with queries, documents, and relevance labels.
- [Embedding distillation](experiments/embedding-distillation/README.en.md): Transfer teacher scores to a smaller embedding model; candidate preparation uses the shared SciFact tool.
- [Reranker distillation](experiments/reranker-distillation/README.en.md): Compare supervised training and score distillation on a fixed candidate pool.
- [ColBERT distillation](experiments/colbert-distillation/README.en.md): Compare score distillation with supervised learning; distillation did not beat the supervised baseline.
- [Visual multi-vector document retrieval](experiments/visual-multivector-retrieval/README.en.md): A page-retrieval baseline plus supervised fine-tuning and distillation comparisons on synthetic pages.
- [Multimodal embedding distillation](experiments/multimodal-embedding-distillation/README.en.md): Explore knowledge transfer from a vision-language retrieval teacher to a compact student.
- [Multimodal reranking](experiments/multimodal-reranking/README.en.md): Explore reranking image-text candidates with anonymized evaluation results.
- [GUI grounding fine-tuning](experiments/gui-grounding-finetuning/README.md): Train screenshot-and-instruction grounding; 30 held-out samples do not establish generalization gains.
- [Video generation trained on teacher samples](experiments/video-generation-distillation/README.md): Four initial held-out videos and eight follow-up student pairs show changes, but do not establish quality gains. The six training samples were selected after generation; the follow-up used one AI reviewer of sampled frames.
- [Image generation with synthetic training data](experiments/image-generation-distillation/README.md): Train a student LoRA on teacher-generated data and compare it with student-generated training data. One AI thumbnail reviewer found no teacher-data advantage in the small camera follow-up; generation pipelines differed. See the [paired analysis](experiments/image-generation-distillation/PAIRED-ANALYSIS.md#english) for an exploratory category breakdown and its limits.
- [Identity image fine-tuning](experiments/image-finetune/README.en.md): Train an identity LoRA; the historical base and adapted outputs use different resolutions and are not directly comparable.
- [Character video fine-tuning and diagnostics](experiments/wan22-video-diagnostics/README.md): Examine base generation, seeds, and LoRA outputs; a small held-out evaluation does not establish that the target character was learned.
- [Vector database benchmark](experiments/vector-db-benchmark/README.en.md): Historical records lack raw results and do not support product rankings.
- [Vector index costs](experiments/vector-index-costs/README.en.md): Compare recall, time and storage across Flat, IVF, HNSW and quantized indexes.
- [Language JEPA training](experiments/language-jepa/README.md): A toy arithmetic comparison of the original model, LoRA, and JEPA; not evidence of improvement.
- [Diffusion language model speed](experiments/diffusion-language-models/README.md): Compare generation paths on three prompts with unequal output lengths and unmeasured quality.
- [GB10 inference slots](experiments/gb10-inference/qwen36-35b-ab/README.md): Compare throughput and completion times for one versus eight slots on the same model and engine.
- [LLM inference performance](experiments/llm-inference-perf/README.en.md): Performance analysis across models and inference configurations.
- [Language model quantization](experiments/llm-mechanics/quantization/README.md): Compare model size, perplexity, and CPU speed; the BF16 path performed unusually slowly.
- [LLM mechanics](experiments/llm-mechanics/README.md): Understand text generation through examples.
- [Tokenizer mechanics](experiments/tokenizer-mechanics/README.en.md): Explore byte-level BPE, real encodings, and token costs.
- [KV cache mechanics](experiments/llm-mechanics/kv_cache/README.en.md): Compare cached and full-prefix numerical results.
- [Inference cost models](experiments/inference-cost-models/README.md): Memory examples and theoretical models; performance estimates have not been independently validated.
- [Data structure costs](experiments/data-structure-costs/README.md): Analyze the costs of data structures and formats.
- [Portrait matting](experiments/portrait-matting/README.en.md): Evaluate subject regions, boundaries, and forward time on 30 still images; results do not establish video performance.
- [Table structure recognition](experiments/table-structure-recognition/README.en.md): Examine cropping and structure predictions on one synthetic table; text recognition is out of scope.
- [Text detection and recognition](experiments/paddle-ocr-detection-recognition/README.en.md): Pass detected text regions to a recognizer; the two-stage result covers one synthetic table.
- [MiniCPM-OCR](experiments/minicpm-ocr/README.md): Document recognition deployment and generation acceleration.
- [Jina CLIP OpenVINO](experiments/jina-clip-openvino/README.md): Compare CPU and NPU correctness and workload-specific performance.
- [Jev open-source evaluation](experiments/jev-open-source/README.md): Compare community implementations on a small decision-making sample.

See each experiment for its environment, datasets, model licenses, and reproduction limits. Some experiments require external data, models, or preparation files that are not yet published; reported measurements apply only under their stated conditions.

## License

Repository code is available under [Apache 2.0](LICENSE); third-party models, data, and code retain their own licenses.
