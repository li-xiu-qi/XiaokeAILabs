"""
unstructured (partition_pdf) element-chunking benchmark:
- rapidocr_direct_and_image.pdf  (native text + embedded images, from RapidAI/RapidOCRPDF tests)
- pdfplumber_federal_register.pdf (15-page native multi-column Federal Register, from jsvine/pdfplumber tests)

Strategies compared per file:
1. "fast"  : pdfminer vector/content-stream path (rule-based, no ML models)
2. "hi_res": layout detection (yolox ONNX) + tesseract OCR + table-transformer structure
             recognition, with infer_table_structure=True and extract_images_in_pdf=True

Metrics: elapsed ms, total elements, element counts by type (full histogram with focus on
Title / NarrativeText / Table / Image), extracted characters, page count, text preview.

Each (file, strategy) partition runs in a fresh subprocess so every timed run pays its own
model-load cost (cold-start, comparable across files). Model weights come from the HuggingFace
hub; if huggingface.co is unreachable from this machine the script falls back to the
hf-mirror.com endpoint.

Usage:
    python scripts/benchmark_unstructured.py                # full benchmark
    python scripts/benchmark_unstructured.py --run-one <pdf-name> <strategy>   # internal
"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STRATEGY_KWARGS = {
    "fast": {},
    "hi_res": {
        "infer_table_structure": True,
        "extract_images_in_pdf": True,
        "extract_image_block_to_payload": True,
        "languages": ["eng"],
    },
}

TEST_PDFS = [
    "rapidocr_direct_and_image.pdf",
    "pdfplumber_federal_register.pdf",
]


def ensure_hf_endpoint():
    """huggingface.co is unreachable from this machine; fall back to the hf-mirror endpoint."""
    if os.environ.get("HF_ENDPOINT"):
        return os.environ["HF_ENDPOINT"]
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return None
    except OSError:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        return os.environ["HF_ENDPOINT"]


def element_type(el) -> str:
    cat = getattr(el, "category", None)
    return str(getattr(cat, "value", cat))


def run_one(pdf_name: str, strategy: str) -> dict:
    """Partition one PDF with one strategy in this (fresh) process; return a metrics record."""
    from unstructured.partition.pdf import partition_pdf

    pdf_path = ROOT / "data" / "reference_samples" / pdf_name
    record = {"strategy": strategy, "success": False, "error": None}
    t0 = time.perf_counter()
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=strategy,
            **STRATEGY_KWARGS[strategy],
        )
        record["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        histogram = Counter(element_type(el) for el in elements)
        pages = {
            el.metadata.page_number
            for el in elements
            if el.metadata is not None and el.metadata.page_number is not None
        }
        total_chars = sum(len(el.text or "") for el in elements)
        preview = " ".join((el.text or "").strip() for el in elements if el.text)
        preview = " ".join(preview.split())[:200]

        record.update({
            "success": True,
            "total_elements": len(elements),
            "element_counts": dict(sorted(histogram.items())),
            "title_count": histogram.get("Title", 0),
            "narrative_text_count": histogram.get("NarrativeText", 0),
            "table_count": histogram.get("Table", 0),
            "image_count": histogram.get("Image", 0),
            "total_chars": total_chars,
            "page_count": len(pages),
            "text_preview": preview,
        })
    except Exception as e:
        record["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def collect_env_info():
    from importlib.metadata import version as pkg_version

    info = {
        "unstructured": pkg_version("unstructured"),
        "python": platform.python_version(),
    }
    for pkg in [
        "unstructured-inference",
        "onnxruntime",
        "onnx",
        "opencv-python",
        "torch",
        "torchvision",
        "transformers",
        "timm",
        "pypdfium2",
        "pdfminer.six",
        "unstructured-pytesseract",
    ]:
        try:
            info[pkg] = pkg_version(pkg)
        except Exception:
            info[pkg] = None
    try:
        import unstructured_pytesseract

        info["tesseract_binary_version"] = str(
            unstructured_pytesseract.pytesseract.get_tesseract_version()
        )
    except Exception as e:
        info["tesseract_binary_version"] = f"unavailable: {e}"
    info["ocr_agent"] = os.environ.get(
        "OCR_AGENT", "unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract"
    )
    info["hf_endpoint"] = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    return info


def spawn_run(pdf_name: str, strategy: str) -> dict:
    """Run one partition in a fresh subprocess (cold-start timing) and parse its JSON record."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-one", pdf_name, strategy],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=os.environ.copy(),
    )
    # the child prints its JSON record after a marker (library logs go to stderr)
    marker = "__BENCHMARK_RECORD__"
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return {
        "strategy": strategy,
        "success": False,
        "error": f"subprocess failed: {proc.stderr[-2000:]}",
    }


def main():
    missing = [str(ROOT / "data" / "reference_samples" / name) for name in TEST_PDFS
               if not (ROOT / "data" / "reference_samples" / name).is_file()]
    if missing:
        raise SystemExit("Missing input PDF samples: " + ", ".join(missing))
    ensure_hf_endpoint()

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark_description": (
            "unstructured partition_pdf element-chunking evaluation: rule-based 'fast' "
            "(pdfminer vector stream) vs model-based 'hi_res' (yolox layout detection + "
            "tesseract OCR + table-transformer structure recognition); each run isolated "
            "in a fresh subprocess for cold-start timing"
        ),
        "environment": collect_env_info(),
        "files_evaluated": [],
    }

    for name in TEST_PDFS:
        pdf_path = ROOT / "data" / "reference_samples" / name
        if not pdf_path.exists():
            print(f"Warning: sample file {name} not found.")
            continue

        print(f"Benchmarking unstructured on {name} ...")
        file_record = {
            "filename": name,
            "size_kb": round(pdf_path.stat().st_size / 1024, 2),
            "runs": [],
        }
        for strategy in ("fast", "hi_res"):
            print(f"  strategy={strategy} (fresh subprocess) ...")
            record = spawn_run(name, strategy)
            file_record["runs"].append(record)
            if record.get("success"):
                print(
                    f"    -> {record['total_elements']} elements in {record['elapsed_ms']} ms "
                    f"(Title={record['title_count']}, NarrativeText={record['narrative_text_count']}, "
                    f"Table={record['table_count']}, Image={record['image_count']})"
                )
            else:
                print(f"    -> FAILED: {record.get('error')}")
        summary["files_evaluated"].append(file_record)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "unstructured_benchmark.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nunstructured benchmark saved to {out_file.relative_to(ROOT)}.")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--run-one":
        print("__BENCHMARK_RECORD__" + json.dumps(run_one(sys.argv[2], sys.argv[3]), ensure_ascii=False))
    else:
        main()
