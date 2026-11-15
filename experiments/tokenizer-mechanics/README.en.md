# From Bytes to Tokens: Tokenization, Generation and Cost

[中文](README.md)

This directory contains experiment code, configuration and result evidence. See the [full experiment document](../../docs/zh/inference/分词机制与生成成本.md) for the purpose, method, data, results and limitations.

## Toy examples

Run Python 3 from this experiment directory. The following scripts use only the standard library; they need no GPU, extra dependencies or model downloads. Input examples are built into the scripts:

```bash
python scripts/toy_bpe.py
python scripts/ngrams.py
```

## Environment setup and real-tokenizer comparison

For a comparison with real tokenizers, install `tiktoken`. The first run may need network access to download vocabularies. Create a separate environment to avoid changing other experiments' dependencies; the activation command below is for Linux and macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python scripts/compare_tiktoken.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`, or run the Python commands with `.\.venv\Scripts\python.exe` directly. The dependency list is not pinned; record the installed version when running the comparison.

On success, the toy BPE script prints each merge and sample tokens, then asserts that encoding and decoding recover the original text. The n-gram script prints English phrases and Chinese character combinations. `compare_tiktoken.py` also checks that UTF-8 bytes round-trip; record the reported version and token IDs alongside the installed `tiktoken` version.

On 2026-11-13, a new Python 3.12.3 virtual environment was created on an NVIDIA DGX Spark. Installing from `requirements.txt` yielded tiktoken 0.14.0, and `pip check` passed. All three scripts ran successfully, including the first-time retrieval of the `r50k_base` and `cl100k_base` vocabularies. This record covers these examples in that environment; it does not establish that other platforms or dependency versions are verified.
