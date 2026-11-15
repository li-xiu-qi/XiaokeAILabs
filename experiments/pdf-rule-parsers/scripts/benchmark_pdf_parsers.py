"""
Benchmark for Rule-Based & Vector-Stream PDF Parsers
Evaluates PyMuPDF and pypdf across:
1. Native digital single-column PDF
2. Two-column academic layout (reading order collapse test)
3. Scanned bitmap PDF (character stream blackout test)
"""

import json
import time
from pathlib import Path
import pymupdf
import pypdf


def generate_test_pdfs(tmp_dir: Path):
    """Generates synthetic test PDFs representing the three distinct document archetypes."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Native digital single-column PDF
    doc_single = pymupdf.open()
    page1 = doc_single.new_page(width=595, height=842) # A4
    page1.insert_text((50, 80), "Artificial Intelligence & Document Engineering Report", fontsize=16)
    page1.insert_text((50, 120), "Paragraph 1: Document AI spans from low-level tokenization to layout semantics.", fontsize=10)
    page1.insert_text((50, 140), "Paragraph 2: Modern document pipelines require resilient structural parsing.", fontsize=10)
    # Add a vector table
    page1.draw_rect(pymupdf.Rect(50, 180, 500, 260), width=1)
    page1.draw_line((50, 220), (500, 220), width=1)
    page1.draw_line((200, 180), (200, 260), width=1)
    page1.insert_text((60, 205), "Metric", fontsize=10)
    page1.insert_text((210, 205), "Measured Value", fontsize=10)
    page1.insert_text((60, 245), "Latency", fontsize=10)
    page1.insert_text((210, 245), "1.25 ms", fontsize=10)
    pdf_single_path = tmp_dir / "single_column_native.pdf"
    pix = page1.get_pixmap(dpi=150)
    doc_single.save(pdf_single_path)
    doc_single.close()

    # 2. Two-column layout (Left: A1, A2; Right: B1, B2)
    doc_two_col = pymupdf.open()
    page2 = doc_two_col.new_page(width=595, height=842)
    page2.insert_text((50, 60), "Academic Dual-Column Paper Sample", fontsize=14)
    # Left Column
    page2.insert_text((50, 100), "Left Col Line 1: Introduction to neural encoders.", fontsize=10)
    page2.insert_text((50, 120), "Left Col Line 2: Methods and preliminary equations.", fontsize=10)
    page2.insert_text((50, 140), "Left Col Line 3: Detailed architecture of vision blocks.", fontsize=10)
    # Right Column (at same Y levels!)
    page2.insert_text((310, 100), "Right Col Line 1: Experimental setups on DGX clusters.", fontsize=10)
    page2.insert_text((310, 120), "Right Col Line 2: Hyperparameter grids and seeds.", fontsize=10)
    page2.insert_text((310, 140), "Right Col Line 3: Ablation benchmarks against baselines.", fontsize=10)
    pdf_two_col_path = tmp_dir / "two_column_native.pdf"
    doc_two_col.save(pdf_two_col_path)
    doc_two_col.close()

    # 3. Scanned PDF：把刚生成的单栏页光栅化后再包回 PDF，不依赖仓库外的绝对路径
    pdf_scanned_path = tmp_dir / "scanned_image_wrapper.pdf"
    doc_scan = pymupdf.open()
    p = doc_scan.new_page(width=595, height=842)
    p.insert_image(pymupdf.Rect(0, 0, 595, 842), pixmap=pix)
    doc_scan.save(pdf_scanned_path)
    doc_scan.close()

    return {
        "single_column": pdf_single_path,
        "two_column": pdf_two_col_path,
        "scanned": pdf_scanned_path
    }


def benchmark_parser(pdf_path: Path):
    """Runs PyMuPDF and pypdf on a single PDF, measuring speed and text fidelity."""
    # PyMuPDF
    t0 = time.perf_counter()
    doc_fitz = pymupdf.open(pdf_path)
    fitz_texts = [p.get_text() for p in doc_fitz]
    fitz_time_ms = (time.perf_counter() - t0) * 1000
    doc_fitz.close()
    full_fitz_text = "\n".join(fitz_texts)

    # pypdf
    t1 = time.perf_counter()
    reader = pypdf.PdfReader(str(pdf_path))
    pypdf_texts = [p.extract_text() for p in reader.pages]
    pypdf_time_ms = (time.perf_counter() - t1) * 1000
    full_pypdf_text = "\n".join(pypdf_texts)

    return {
        "file_size_bytes": pdf_path.stat().st_size,
        "fitz": {
            "elapsed_ms": round(fitz_time_ms, 3),
            "char_count": len(full_fitz_text),
            "text_sample": full_fitz_text.strip().replace("\n", " ")[:150]
        },
        "pypdf": {
            "elapsed_ms": round(pypdf_time_ms, 3),
            "char_count": len(full_pypdf_text),
            "text_sample": full_pypdf_text.strip().replace("\n", " ")[:150]
        }
    }


def evaluate_two_column_reading_order(pdf_path: Path):
    """
    Checks if a naive text extraction interleaves left and right columns
    (e.g., Left 1 -> Right 1 -> Left 2 -> Right 2 instead of Left 1..3 -> Right 1..3).
    """
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    raw_text = page.get_text("text")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # If line 0 is Left 1 and line 1 is Right 1, naive extraction interleaved them
    interleaved = False
    if len(lines) >= 4:
        first_four = lines[:4]
        if any("Left Col Line 1" in s for s in first_four) and any("Right Col Line 1" in s for s in first_four):
            # Check if Right Col Line 1 appears BEFORE Left Col Line 3
            left_3_idx = next((i for i, s in enumerate(lines) if "Left Col Line 3" in s), 999)
            right_1_idx = next((i for i, s in enumerate(lines) if "Right Col Line 1" in s), -1)
            if right_1_idx < left_3_idx:
                interleaved = True

    # Layout-aware block extraction
    blocks = page.get_text("blocks") # (x0, y0, x1, y1, text, block_no, block_type)
    # Sort by column (x0) then row (y0)
    sorted_blocks = sorted(blocks, key=lambda b: (round(b[0] / 200), b[1]))
    reordered_text = " ".join(b[4].strip() for b in sorted_blocks)

    doc.close()
    return {
        "naive_text_interleaved": interleaved,
        "total_extracted_lines": len(lines),
        "lines_preview": lines[:6],
        "layout_aware_available": len(blocks) > 1
    }


def main():
    root = Path(__file__).resolve().parents[1]
    tmp_dir = root / "data"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("1. Generating test PDF archetypes...")
    pdfs = generate_test_pdfs(tmp_dir)

    print("2. Benchmarking parsers on all archetypes...")
    bench_data = {}
    for name, path in pdfs.items():
        bench_data[name] = benchmark_parser(path)

    print("3. Evaluating two-column reading order anomaly...")
    reading_order_res = evaluate_two_column_reading_order(pdfs["two_column"])

    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parser_comparison": bench_data,
        "reading_order_experiment": reading_order_res
    }

    out_file = results_dir / "pdf_parsers_benchmark.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBenchmark completed successfully. Results saved to {out_file.relative_to(root)}:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
