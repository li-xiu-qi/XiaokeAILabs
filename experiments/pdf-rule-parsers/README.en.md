# An Empirical Evaluation of Rule-Based and PDF Vector-Stream Parsers

[中文](README.md)

This directory contains experiments on traditional rule-based document extraction from native PDF instructions (PDF vectors/content streams). It evaluates popular frameworks on authoritative open-source test cases, including performance, runtime, table reconstruction, and physical boundaries.

[scripts](scripts/) · [historical evidence](evidence/)

## 1. Test samples from official open-source projects

To keep the test inputs representative and traceable, the samples were selected from official test sets of well-known open-source projects:

1. **`rapidocr_direct_and_image.pdf`** (from the official `RapidAI/RapidOCRPDF` test set, 703 KB): a typical mixed PDF containing native digital text and embedded image regions.
2. **`pdfplumber_federal_register.pdf`** (from the official `jsvine/pdfplumber` test set, 697 KB): a complex, 15-page US Federal Register document with more than 60,000 characters, dense layout, ruled tables, and fine-print footnotes.
3. **`testpage_cn.pdf`** (285 KB): a scanned packaging PDF whose underlying pages are pure bitmap renderings, used to demonstrate the zero-recall blind spot of rule-based parsers.

## 2. Traditional frameworks evaluated

1. **`pypdf`**: pure Python, no external dependencies, lightweight and fast.
2. **`PyMuPDF (fitz)`**: a high-performance C/C++ MuPDF engine that supports text streams, image extraction, and native table detection (`find_tables`).
3. **`pdfplumber`**: a higher-level visual and table-extraction library based on pdfminer, specializing in precise table extraction from vector lines and rectangles.
4. **`pdfminer.six`**: a stream parser that works at character-level layout and absolute physical coordinates.

## 3. Reproduction commands

Run the following commands from this experiment directory. The historical environment was Windows with Python 3.13, and rule-based parsing can run on CPU. For a new environment, create a virtual environment and install the parsing dependencies. Versions are not fully pinned, so historical numerical results are not guaranteed:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate; Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-rules.txt
python -m pip check
python -c "import pymupdf, pypdf, pdfplumber, pdfminer; print('PDF parsing dependencies import successfully')"
```

On 2026-11-13, installation from this list in a fresh Linux ARM64 / Python 3.12.3 virtual environment passed `pip check`. The installed versions at that time were PyMuPDF 2.1, pypdf 7.0.1, pdfplumber 0.11.10, and pdfminer.six 20260107. The synthetic PDF script ran successfully: both parsers extracted text from single- and two-column samples, while neither extracted text from the scanned-image wrapper. This only verifies execution on synthetic samples; it does not reproduce real samples, OCR, or historical results. The dependency list is unpinned, so later installations may use different versions.

The synthetic-PDF script generates its own inputs. Put real samples in `data/reference_samples/`. Upstream candidates that can be checked are [RapidOCRPDF's direct_and_image.pdf](https://github.com/RapidAI/RapidOCRPDF/blob/main/tests/test_files/direct_and_image.pdf) and [pdfplumber's Federal Register test PDF](https://github.com/jsvine/pdfplumber/blob/stable/tests/pdfs/federal-register-2020-17221.pdf); save them as `rapidocr_direct_and_image.pdf` and `pdfplumber_federal_register.pdf`, respectively. Historical records did not preserve their source revisions or complete hashes, so current files are not guaranteed to match the historical samples. The historical source and redistribution conditions for `testpage_cn.pdf` have not been verified. Supply a scan that you are entitled to use; results from a replacement file are not directly comparable with historical results. A measurement for a sample that is missing has not been completed.

On 2026-11-14, `benchmark_single_file` was run separately on two fixed upstream samples in an isolated copy of the environment above on Linux ARM64. All four parsers processed the two-page mixed PDF and the 15-page Federal Register PDF. This verifies rule-based parsing of those files only; it does not reproduce the scanned sample, OCR, unstructured, or the full historical benchmark. The environment was copied from an existing installation, and one-off timings are not a performance comparison.

In a separate working copy, download the exact samples used in that run and verify their SHA-256 hashes. A scan must still be supplied to run the complete three-file command below:

```bash
mkdir -p data/reference_samples
curl -fL https://raw.githubusercontent.com/RapidAI/RapidOCRPDF/bb99e81e397c53fc9b6bba71d49408470c02f09e/tests/test_files/direct_and_image.pdf -o data/reference_samples/rapidocr_direct_and_image.pdf
curl -fL https://raw.githubusercontent.com/jsvine/pdfplumber/4c64b92d5caccd71c645e98e0fabb0c4dba7ff45/tests/pdfs/federal-register-2020-17221.pdf -o data/reference_samples/pdfplumber_federal_register.pdf
sha256sum data/reference_samples/rapidocr_direct_and_image.pdf data/reference_samples/pdfplumber_federal_register.pdf
# Expected: 9823477e6d872d46f9bd7b5bff75092d49b097841919eccb9855a0aa06d5ebf9
#           ac419e4e084ea7a7a33a7be3f129dd07bc705b1a9e45ad751e33357f17d60f5c
```

Scripts create `data/` and `results/` under the experiment directory and may overwrite files or refuse to regenerate existing ones. Use a separate working copy to protect existing data and results. The RapidOCR script accepts image paths on the command line and can be run from a separate working copy. Historical verification files are in `evidence/`. The original local path for the OCR image has been redacted; only its location relative to the original setup remains, so it does not identify a current file location.

```bash
# Compare popular frameworks on official samples
python scripts/benchmark_popular_frameworks.py

# Evaluate two-column reading-order interleaving and baseline models
python scripts/benchmark_pdf_parsers.py
```

Results are written to `results/popular_frameworks_comparison.json` and `results/pdf_parsers_benchmark.json`.

## 4. OCR engines and unstructured element partitioning

In addition to rule-based parsing, two end-to-end paths were evaluated: image OCR with RapidOCR and element partitioning with `unstructured`'s `partition_pdf`.

1. **RapidOCR** (requires `rapidocr-onnxruntime`; model and package versions are not pinned):
   - `synthetic-table.png`, a synthetic table image from the `table-structure-recognition` experiment's figures.
   - `testpage_cn.jpg`, a Chinese scanned page from the `minicpm-ocr` experiment's evidence.
   - Metrics: number of detected boxes, runtime in milliseconds (total and det/cls/rec breakdown, mean of three runs), and mean confidence.
2. **`unstructured` `partition_pdf`**:
   - `rapidocr_direct_and_image.pdf` and `pdfplumber_federal_register.pdf`.
   - Two strategies: `fast` (pdfminer vector stream, rules only, no model) and `hi_res` (YOLOX ONNX layout detection, Tesseract OCR, and Table Transformer table-structure recognition, with `infer_table_structure=True` and `extract_images_in_pdf=True`).
   - Metrics: runtime and counts of each element type (Title / NarrativeText / Table / Image, plus the full histogram).
   - Each (file, strategy) combination is timed in a separate subprocess with a cold start.

The historical environment was Windows with Python 3.13. In that combination, the `unstructured[pdf]` extra did not install `unstructured-inference`; it had to be added manually. Other recorded dependencies were `pi-heif`, `pikepdf`, `pdf2image`, `unstructured-pytesseract`, and `unstructured-inference` (including CPU PyTorch). The Tesseract binary was installed with Scoop, and `eng.traineddata` was added. Model weights came from Hugging Face. If the script cannot connect, it automatically falls back to `hf-mirror.com`; you can also set `HF_ENDPOINT` beforehand. Full versions for OCR, unstructured, and external system dependencies are not pinned, so a one-command reproduction of historical results is not guaranteed.

Install the candidate dependencies in an isolated virtual environment. If you run only one path, install only its corresponding requirement file:

```bash
python -m pip install -r requirements-rapidocr.txt -r requirements-unstructured.txt
python -m pip check
python -c "from rapidocr_onnxruntime import RapidOCR; from unstructured.partition.pdf import partition_pdf; print('Python dependencies import successfully')"
tesseract --version
tesseract --list-langs
pdftoppm -v
```

Install Tesseract, its English `eng` language data, and Poppler for your operating system, and add them to `PATH`. The historical environment used English OCR. Package and model versions are unpinned; these commands are a setup starting point, and system and transitive dependencies must still be checked for your platform. RapidOCR accepts explicit image paths and exits with an error if any input is missing; after a successful run, check `images_evaluated` in the output. The `hi_res` path also needs to obtain model weights.

```bash
# RapidOCR image benchmark: replace these with paths to your own images
python scripts/benchmark_rapidocr.py /path/to/synthetic-table.png /path/to/testpage_cn.jpg

# unstructured partition_pdf benchmark (fast vs hi_res)
python scripts/benchmark_unstructured.py

# Merge the two result sets and print a summary
python scripts/merge_benchmark_results.py
```

The merged evaluation is written to `results/ocr_engines_and_unstructured_benchmark.json`; component results are in `results/rapidocr_benchmark.json` and `results/unstructured_benchmark.json`. Historical records are in the [summary evidence](evidence/ocr_engines_and_unstructured_benchmark.json).

On 2026-11-14, `requirements-rapidocr.txt` was installed in a new isolated Linux ARM64 / Python 3.12.3 environment and passed `pip check`. RapidOCR 1.4.4 with ONNX Runtime 1.32.0 used the CPU execution provider. After warmup and three timed runs per image, it detected 32 boxes in the repository's `testpage_cn.jpg` example and 12 in `synthetic-table.png`. This checks the image OCR entry point and that environment combination; it does not establish recognition accuracy, GPU execution or unstructured behavior.

Observed results: `hi_res` was the only path that produced Table / Image elements (`fast` is designed to process only vector text streams, so its Table/Image counts are always 0). On the 15-page Federal Register PDF, `hi_res` took about 24 times longer than `fast` (113.2 s vs 4.7 s), detecting 2 tables and 11 images while producing many UncategorizedText elements. `fast` reported substantially more Title elements (106) than `hi_res` (57), indicating more false positives from its rule-based header/footer heuristics.
