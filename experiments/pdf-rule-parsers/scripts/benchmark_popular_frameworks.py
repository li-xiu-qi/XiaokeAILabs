"""
Comprehensive Benchmark of Popular PDF Extraction Frameworks:
- PyMuPDF (fitz)
- pypdf
- pdfplumber
- pdfminer.six

Evaluated on official test files from popular open-source projects:
1. rapidocr_direct_and_image.pdf (from RapidAI/RapidOCRPDF)
2. pdfplumber_federal_register.pdf (from jsvine/pdfplumber tests)
3. testpage_cn.pdf (pure scanned image inside PDF wrapper)
"""

import json
import time
from pathlib import Path

import pymupdf
import pypdf
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text


def benchmark_single_file(pdf_path: Path):
    file_size_kb = round(pdf_path.stat().st_size / 1024, 2)
    results = {
        "filename": pdf_path.name,
        "size_kb": file_size_kb,
        "frameworks": {}
    }

    # 1. PyMuPDF (fitz)
    try:
        t0 = time.perf_counter()
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)
        full_text = ""
        total_images = 0
        tables_found = 0
        for page in doc:
            full_text += page.get_text()
            total_images += len(page.get_images())
            # pymupdf find_tables
            try:
                tabs = page.find_tables()
                tables_found += len(tabs.tables)
            except Exception:
                pass
        t_fitz = (time.perf_counter() - t0) * 1000
        doc.close()
        results["frameworks"]["PyMuPDF"] = {
            "success": True,
            "elapsed_ms": round(t_fitz, 2),
            "ms_per_page": round(t_fitz / max(1, page_count), 2),
            "page_count": page_count,
            "chars_extracted": len(full_text),
            "images_detected": total_images,
            "tables_detected": tables_found,
            "text_preview": full_text.strip().replace("\n", " ")[:120]
        }
    except Exception as e:
        results["frameworks"]["PyMuPDF"] = {"success": False, "error": str(e)}

    # 2. pypdf
    try:
        t0 = time.perf_counter()
        reader = pypdf.PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        full_text = ""
        total_images = 0
        for page in reader.pages:
            full_text += page.extract_text() or ""
            total_images += len(page.images)
        t_pypdf = (time.perf_counter() - t0) * 1000
        results["frameworks"]["pypdf"] = {
            "success": True,
            "elapsed_ms": round(t_pypdf, 2),
            "ms_per_page": round(t_pypdf / max(1, page_count), 2),
            "page_count": page_count,
            "chars_extracted": len(full_text),
            "images_detected": total_images,
            "tables_detected": "Not supported natively",
            "text_preview": full_text.strip().replace("\n", " ")[:120]
        }
    except Exception as e:
        results["frameworks"]["pypdf"] = {"success": False, "error": str(e)}

    # 3. pdfplumber
    try:
        t0 = time.perf_counter()
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            full_text = ""
            tables_found = 0
            for page in pdf.pages:
                full_text += page.extract_text() or ""
                tabs = page.extract_tables()
                tables_found += len(tabs)
        t_plumber = (time.perf_counter() - t0) * 1000
        results["frameworks"]["pdfplumber"] = {
            "success": True,
            "elapsed_ms": round(t_plumber, 2),
            "ms_per_page": round(t_plumber / max(1, page_count), 2),
            "page_count": page_count,
            "chars_extracted": len(full_text),
            "images_detected": "Via objects inspection",
            "tables_detected": tables_found,
            "text_preview": full_text.strip().replace("\n", " ")[:120]
        }
    except Exception as e:
        results["frameworks"]["pdfplumber"] = {"success": False, "error": str(e)}

    # 4. pdfminer.six
    try:
        t0 = time.perf_counter()
        full_text = pdfminer_extract_text(str(pdf_path))
        t_pdfminer = (time.perf_counter() - t0) * 1000
        results["frameworks"]["pdfminer.six"] = {
            "success": True,
            "elapsed_ms": round(t_pdfminer, 2),
            "ms_per_page": round(t_pdfminer / max(1, results["frameworks"]["PyMuPDF"]["page_count"]), 2),
            "page_count": results["frameworks"]["PyMuPDF"]["page_count"],
            "chars_extracted": len(full_text),
            "images_detected": "Via layout LTImage",
            "tables_detected": "Not supported natively",
            "text_preview": full_text.strip().replace("\n", " ")[:120]
        }
    except Exception as e:
        results["frameworks"]["pdfminer.six"] = {"success": False, "error": str(e)}

    return results


def main():
    root = Path(__file__).resolve().parents[1]
    samples_dir = root / "data" / "reference_samples"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sample_files = [
        samples_dir / "rapidocr_direct_and_image.pdf",
        samples_dir / "pdfplumber_federal_register.pdf",
        samples_dir / "testpage_cn.pdf"
    ]

    missing = [str(path) for path in sample_files if not path.is_file()]
    if missing:
        raise SystemExit("Missing input PDF samples: " + ", ".join(missing))

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark_description": "Cross-framework evaluation on real-world reference PDFs from RapidOCR & pdfplumber",
        "tested_frameworks": ["PyMuPDF", "pypdf", "pdfplumber", "pdfminer.six"],
        "files_evaluated": []
    }

    for sf in sample_files:
        if sf.exists():
            print(f"Benchmarking on {sf.name}...")
            res = benchmark_single_file(sf)
            benchmark_summary["files_evaluated"].append(res)
        else:
            print(f"Warning: sample file {sf.name} not found.")

    out_file = results_dir / "popular_frameworks_comparison.json"
    out_file.write_text(json.dumps(benchmark_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nBenchmark completed. Summary saved to {out_file.relative_to(root)}.")
    print(json.dumps(benchmark_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
