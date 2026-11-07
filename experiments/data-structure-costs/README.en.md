# Costs of data structures and storage formats

[中文](README.md)

Why can a Bloom filter reduce unnecessary lookups, and what does it cost? Why can the same data occupy different space or support different access patterns in JSON, spreadsheets, and database files? This directory uses simplified cost models and small scripts to explore the space and computation costs of indexes, data structures, and storage formats. It helps identify questions worth measuring on a real workload.

Most scripts are teaching models or self-made implementations, not database product implementations. Formulas, hardware, data distribution, and implementation affect results; theoretical figures here cannot directly predict production latency. Some historical records lack complete original results and environments and cannot serve as independently verifiable performance conclusions.

## Start with a Bloom filter

A Bloom filter maps elements to a bit array. A negative lookup means an element was not inserted; a positive may be a false positive. A larger bit array can usually reduce the false-positive rate at the cost of space. These examples compare an approximate theoretical rate with a measurement using a self-made Python implementation and specified hashing and data. They do not measure database query latency.

```bash
cd experiments/data-structure-costs
python scripts/bloom_filter_model.py
mkdir -p results
python scripts/verify_bloom_filter.py
```

Both scripts use the Python standard library. The verification script writes generated JSON under `results/`; this directory is ignored by Git and the repository does not include those run results. In Windows PowerShell, create it using `New-Item -ItemType Directory -Force results`. Theoretical values depend on approximations and hashing assumptions; a finite number of queries also produces sampling error.

## Other material

| Topic | Entry points |
|---|---|
| Trees, hashing, LSM and column stores | Simplified models in `scripts/btree_model.py`, `hash_model.py`, `lsm_model.py`, and `columnar_model.py`, plus some `verify_*.py` scripts |
| Text and queries | Simplified inverted-index, BM25 and join models |
| Probabilistic structures and streams | Teaching implementations of Bloom filters, HyperLogLog, Count-Min Sketch, and frequent-item algorithms |
| Encoding and data formats | Integer encoding models and JSON, JSONL, SQLite, and XLSX experiment scripts |

Inputs, dependencies, and outputs vary across scripts. Some verifiers require a manually created `results/` directory, and not every model has a matching verification script. Read each script's description and dependencies first. Running successfully does not mean that its assumptions and conclusions have been independently audited.

Vector indexes have a separate experiment covering Flat, IVF, HNSW, and quantization search paths, with recall and timing comparisons under stated conditions: see [vector index costs](../vector-index-costs/README.en.md). Historical scripts and documents on that topic here are retained for provenance, not as the entry point for new vector-index work.

## Limits of the historical data-format record

The [historical data-format record](docs/数据格式体积与速度实测.md) (Chinese) retains tables and figures, but the repository lacks its complete original results and environment. Some general claims about compression, indexing, and speed ratios lack evidence or conflict with the recorded values. It is not a basis for format selection. Re-test using real data, specific library versions, equivalent query semantics, and repeated measurements. In particular, simulated XLSX or SQLite sizes must not be presented as measured file sizes.
