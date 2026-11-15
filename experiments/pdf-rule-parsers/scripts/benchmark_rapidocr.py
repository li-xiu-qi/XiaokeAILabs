"""
RapidOCR (rapidocr-onnxruntime) Benchmark on standalone images:
- synthetic-table.png (synthetic table image, from table-structure-recognition experiment)
- testpage_cn.jpg     (scanned Chinese page, from minicpm-ocr experiment)

Metrics per image:
- detected box count
- elapsed time in ms (total + det/cls/rec breakdown), averaged over N runs
- average recognition confidence (mean/min/max over detected boxes)
"""

import argparse
import json
import statistics
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

RUNS = 3

def benchmark_image(engine: RapidOCR, img_path: Path, runs: int = RUNS):
    file_size_kb = round(img_path.stat().st_size / 1024, 2)

    # warmup (first call pays onnxruntime session init / lazy allocations)
    warm_result, warm_elapse = engine(str(img_path))
    warm_boxes = len(warm_result) if warm_result else 0

    run_records = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result, elapse = engine(str(img_path))
        t_total = (time.perf_counter() - t0) * 1000
        boxes = len(result) if result else 0
        scores = [float(item[2]) for item in result] if result else []
        run_records.append({
            "elapsed_ms": round(t_total, 2),
            "engine_elapsed_ms": round(sum(elapse) * 1000, 2),
            "det_ms": round(elapse[0] * 1000, 2),
            "cls_ms": round(elapse[1] * 1000, 2),
            "rec_ms": round(elapse[2] * 1000, 2),
            "boxes": boxes,
            "avg_confidence": round(statistics.mean(scores), 4) if scores else 0.0,
        })

    # reuse last run's boxes/scores for stable text stats
    result, _ = engine(str(img_path))
    scores = [float(item[2]) for item in result] if result else []
    texts = [str(item[1]) for item in result] if result else []

    return {
        "filename": img_path.name,
        "source_path": img_path.name,
        "size_kb": file_size_kb,
        "warmup": {"boxes": warm_boxes},
        "runs": run_records,
        "boxes": run_records[-1]["boxes"],
        "elapsed_ms_mean": round(statistics.mean(r["elapsed_ms"] for r in run_records), 2),
        "elapsed_ms_min": min(r["elapsed_ms"] for r in run_records),
        "elapsed_ms_max": max(r["elapsed_ms"] for r in run_records),
        "det_ms_mean": round(statistics.mean(r["det_ms"] for r in run_records), 2),
        "cls_ms_mean": round(statistics.mean(r["cls_ms"] for r in run_records), 2),
        "rec_ms_mean": round(statistics.mean(r["rec_ms"] for r in run_records), 2),
        "avg_confidence": round(statistics.mean(scores), 4) if scores else 0.0,
        "min_confidence": round(min(scores), 4) if scores else 0.0,
        "max_confidence": round(max(scores), 4) if scores else 0.0,
        "total_chars": sum(len(t) for t in texts),
        "text_preview": " ".join(texts)[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark explicitly supplied image files")
    parser.add_argument("images", nargs="+", type=Path, help="image paths to benchmark")
    args = parser.parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        parser.error("image files not found: " + ", ".join(missing))
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    engine = RapidOCR()

    summary = {
        "engine": "RapidOCR",
        "engine_package": "rapidocr-onnxruntime",
        "engine_version": pkg_version("rapidocr-onnxruntime"),
        "runs_per_image": RUNS,
        "images_evaluated": [],
    }

    for img in args.images:
        print(f"Benchmarking RapidOCR on {img.name}...")
        summary["images_evaluated"].append(benchmark_image(engine, img))

    out_file = results_dir / "rapidocr_benchmark.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapidOCR benchmark saved to {out_file.relative_to(root)}.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
