# Shared SciFact candidate pools

English | [中文](README.md)

The embedding, reranker and ColBERT distillation experiments share this single candidate-pool tool. It checks evidence for training positives, filters near-duplicate queries across splits and generates BM25 candidates. Each experiment describes how it samples and uses the resulting candidates.

Python 3.11 or later is required; only the standard library is used and no GPU is needed. Download and extract [BEIR SciFact](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip), retaining `corpus.jsonl`, `queries.jsonl` and `qrels/`. Extract `claims_train.jsonl` from the [original SciFact release](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz). See [BEIR](https://github.com/beir-cellar/beir) and the [SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md) for sources and terms. The data are not included in this repository.

From the repository root, replace the input paths:

```bash
python utils/scifact-pools/prepare_pools.py --data /path/to/scifact \
  --original-claims /path/to/claims_train.jsonl --out /path/to/prepared
```

The tool writes `pools.json` and `data-audit.json`. The audit records SHA-256 hashes of the four BEIR inputs, original claims, configuration and output. Compare them with [source-hashes.json](source-hashes.json): the four inputs and original claims should match. The historical pool SHA-256 reproduced on Linux is `e7cac48e7083bdd948e2773136c1df9672b4ca8f787ee5b774697b32d689c5f8`. A Windows CPU rebuild produced the same split, training candidates and development/test candidate order as Linux, but some BM25 floating-point scores differ in their last digits, changing the file hash. Use Linux and check the hash for byte-for-byte reproduction. The manifest's `original_pool_config.json` hash refers to the historical original config, not the current published config, so those hashes need not match. The tool records hashes but does not compare them automatically. Results from different inputs or settings should not be directly equated with the historical metrics.
