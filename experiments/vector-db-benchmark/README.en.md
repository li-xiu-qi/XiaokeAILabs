# Comparing Six Vector Retrieval Options: Indexing, Recall, and Storage

[中文](README.md)

This directory contains experiment code, configuration, and result evidence. See the [full experiment documentation](../../docs/zh/retrieval/vector-search/README.md) for the purpose, principles, data, results, and limitations.

[scripts](scripts/) · [config](config/) · [figures](figures/) · [Environment setup](环境搭建.md)

## Environment setup and reproduction from one configuration

The example below uses Windows, PowerShell, and sqlite-vec to avoid starting every service at once. Run commands from this experiment directory. You can also begin with the [synthetic-data installation check](环境搭建.md#1-安装与冒烟检查).

The directory's `requirements.txt` aggregates dependencies for the Python database adapters; it does not mean that every database service and client has been verified together in one environment. Install selectively for the adapter you plan to use, following the steps below. Docker services, the SurrealDB executable, and historical probes have additional setup requirements; see the [environment setup guide](环境搭建.md).

On 2026-11-14, a new isolated Linux ARM64 environment with Python 3.12.3, NumPy 2.5.2, sqlite-vec 0.1.9, and SQLite 3.45.1 passed `pip check`. The commands below completed a sqlite-vec benchmark on 30,000 synthetic 768-dimensional vectors and 50 queries, and its result JSON contained no configuration errors. This CPU synthetic check does not reproduce the real corpus or other databases; the original corpus download timed out during this attempt.

```bash
python3 -m venv /path/to/venvs/vector-db
/path/to/venvs/vector-db/bin/python -m pip install numpy==2.5.2 sqlite-vec==0.1.9
/path/to/venvs/vector-db/bin/python -m pip check
/path/to/venvs/vector-db/bin/python scripts/index_bench.py --only sqlite --synthetic --rounds 2
```

Inspect each entry in `results/indexbench-synthetic-*.json` for an `error` field. The command creates isolated database files and attempts to clean them up; run it in a separate experiment copy.

On 2026-11-14, a separate copied environment on Linux ARM64 encoded the SHA-256-checked 20 Newsgroups archive used by scikit-learn on CPU using fixed `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It produced 384-dimensional vectors for 11,293 training texts and 600 test texts. The sqlite-vec benchmark used 40 queries, and the text-storage A/B also completed without configuration errors. Python 3.12.3, NumPy 2.5.4, sentence-transformers 6.0.1, sqlite-vec 0.1.9, and SQLite 3.45.1 passed `pip check`. The environment was copied and adjusted, not installed from scratch. Other jobs shared the device, so these timings are not a performance comparison; other databases were not covered.

### 1. Prepare Python, data, and the model

The sqlite-vec synthetic-data installation check was recorded with Windows, Python 3.13.0, NumPy 2.5.2, and sqlite-vec 0.1.9. The Windows text-embedding generation and full benchmark below have not been revalidated end to end. `sentence-transformers` is not version-pinned; record its version and the model revision when rerunning.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.2 sqlite-vec==0.1.9 sentence-transformers
.\.venv\Scripts\python.exe -c "import sqlite3, sqlite_vec, numpy, sentence_transformers; c=sqlite3.connect(':memory:'); c.enable_load_extension(True); sqlite_vec.load(c); print('sqlite-vec', c.execute('select vec_version()').fetchone()[0], 'numpy', numpy.__version__, 'sentence-transformers', sentence_transformers.__version__)"
& {
    foreach ($path in @('data/20news', 'data/20news-bydate.tar.gz', 'models/all-MiniLM-L6-v2')) {
        if (Test-Path -LiteralPath $path) { throw "Target already exists; use a fresh experiment copy: $path" }
    }
    New-Item -ItemType Directory -Force data/20news, models | Out-Null
    Invoke-WebRequest 'https://ndownloader.figshare.com/files/5975967' -OutFile data/20news-bydate.tar.gz
    if ((Get-FileHash data/20news-bydate.tar.gz -Algorithm SHA256).Hash -ne '8F1B2514CA22A5ADE8FBB9CFA5727DF95FA587F4C87B786E15C759FA66D95610') { throw '20 Newsgroups archive SHA-256 mismatch' }
    tar -xzf data/20news-bydate.tar.gz -C data/20news
}
```

This check loads the SQLite extension in memory and prints versions. If Python's SQLite build does not support extension loading, use a Python build that does. Download the complete [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model files into `models/all-MiniLM-L6-v2/`, including weights, configuration, and tokenizer files. The command below pins the revision used for this run; reserve disk space before downloading. `gen_emb.py` reads this directory in offline mode and will not download the model automatically.

```powershell
& {
    if (Test-Path -LiteralPath 'models/all-MiniLM-L6-v2') { throw 'Model directory already exists; use a fresh experiment copy' }
    .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir='models/all-MiniLM-L6-v2')"
}
```

The download and text-embedding generation below have not been fully validated in a fresh environment:

```powershell
.\.venv\Scripts\python.exe scripts/gen_emb.py --texts-output data/20news_texts.npz
```

In one corpus pass, the script creates vectors in `data/20news_emb.npz` and corresponding ordered texts in `data/20news_texts.npz`. It removes email headers, collapses whitespace, and truncates each body to 1,200 characters. For each class, test files are sorted by filename, the first 30 are selected, and only then short texts are filtered, so this is not a random sample. The vector and text files each store a SHA-256 digest of the ordered text content. Text-storage A/B and hybrid retrieval recompute and check the digest and row count. The digest checks the declared text content and order in both files; it does not verify that the vector array itself was unchanged. Check the actual downloaded corpus version and its terms of use. This repository does not distribute the source corpus, model, or generated data.

The script does not overwrite existing outputs. To retain multiple datasets, use `--data-dir`, `--model`, `--output`, and `--texts-output` to select inputs and outputs. An interrupted run may leave incomplete files; inspect them and choose new output paths. Benchmark scripts currently read fixed default filenames under `data/`; to use other paths, prepare both files in an isolated working copy first. Historical vectors have no text digest, so their correspondence to the current corpus text cannot be established. Do not pair newly generated text with historical vectors, or treat newly generated data and results as original evidence for historical values. The real-corpus workflow for other databases and on Windows has not been revalidated end to end.

### 2. Run and inspect the results

```powershell
.\.venv\Scripts\python.exe scripts/index_bench.py --only sqlite
```

The script computes exact top-10 results from the actual data, runs the index benchmark for the selected database, and writes `results/indexbench-*.json`. It catches per-configuration exceptions; inspect the JSON for an `error` field instead of relying only on the process exit code. Retain the JSON, full environment, and configuration to make your result traceable. To test other databases, prepare their services as described in [environment and test methods](环境搭建.md), then change `--only`, for example to `--only milvus qdrant`. Each index benchmark creates a separate collection or local path for the run and attempts to clean up what it created. Some directories or objects from exceptional runs may remain. The sqlite-vec path has been exercised in an isolated test directory; test the isolation changes for other databases against their own isolated instances and data directories. Other historical benchmark and probe scripts have not received the same isolation changes; do not run them against instances containing existing data. SurrealDB tests also require `psutil`; the current startup path uses a Windows executable and needs adaptation and verification before rerunning on Linux.

After generating both files with the command above and passing the consistency checks, run `scripts/index_bench.py --only sqlite --text-storage`. Historical probe scripts may not check row correspondence; for example, `scripts/probe_vikingdb.py` does not yet perform this check. Inspect each script's input and isolation requirements before running it. The plotting script contains hard-coded data from historical tables, so its default plot must not be treated as a new result. Before plotting new JSON, check the script's assumptions about parameter tiers and input files. New plots cannot serve as original evidence for historical data.
