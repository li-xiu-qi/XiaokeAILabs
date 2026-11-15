"""
Merge the RapidOCR and unstructured benchmark artifacts into a single structured JSON:
- results/rapidocr_benchmark.json     (part 1: image OCR engine benchmark)
- results/unstructured_benchmark.json (part 2: partition_pdf element chunking benchmark)

Output: results/ocr_engines_and_unstructured_benchmark.json
"""

import json
import time
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"

    rapidocr = load(results_dir / "rapidocr_benchmark.json")
    unstructured = load(results_dir / "unstructured_benchmark.json")

    combined = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark_description": (
            "Combined evaluation: (1) RapidOCR image OCR engine on synthetic-table.png and "
            "testpage_cn.jpg; (2) unstructured partition_pdf element chunking on "
            "rapidocr_direct_and_image.pdf and pdfplumber_federal_register.pdf "
            "(rule-based 'fast' vs model-based 'hi_res')"
        ),
        "part1_rapidocr_image_ocr": rapidocr,
        "part2_unstructured_partition_pdf": unstructured,
    }

    out_file = results_dir / "ocr_engines_and_unstructured_benchmark.json"
    out_file.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)

    print("\n[Part 1] RapidOCR (rapidocr-onnxruntime) image OCR")
    if rapidocr:
        print(f"  engine: {rapidocr['engine']} {rapidocr['engine_version']} "
              f"(runs per image: {rapidocr['runs_per_image']})")
        for img in rapidocr.get("images_evaluated", []):
            print(f"  - {img['filename']} ({img['size_kb']} KB)")
            print(f"      boxes           : {img['boxes']}")
            print(f"      elapsed_ms      : mean {img['elapsed_ms_mean']} "
                  f"(min {img['elapsed_ms_min']} / max {img['elapsed_ms_max']})")
            print(f"      det/cls/rec ms  : {img['det_ms_mean']} / {img['cls_ms_mean']} / "
                  f"{img['rec_ms_mean']} (mean)")
            print(f"      avg_confidence  : {img['avg_confidence']} "
                  f"(min {img['min_confidence']} / max {img['max_confidence']})")
    else:
        print("  (missing)")

    print("\n[Part 2] unstructured partition_pdf element chunking")
    if unstructured:
        env = unstructured.get("environment", {})
        print(f"  unstructured {env.get('unstructured')} / unstructured-inference "
              f"{env.get('unstructured-inference')} / tesseract {env.get('tesseract_binary_version')}")
        for f in unstructured.get("files_evaluated", []):
            print(f"  - {f['filename']} ({f['size_kb']} KB)")
            for run in f.get("runs", []):
                if not run.get("success"):
                    print(f"      strategy={run.get('strategy')}: FAILED - {run.get('error')}")
                    continue
                print(f"      strategy={run['strategy']}: {run['total_elements']} elements "
                      f"in {run['elapsed_ms']} ms ({run['page_count']} pages)")
                print(f"        Title={run['title_count']}  NarrativeText={run['narrative_text_count']}"
                      f"  Table={run['table_count']}  Image={run['image_count']}")
                print(f"        full histogram: {run['element_counts']}")
    else:
        print("  (missing)")

    print(f"\nCombined results saved to {out_file.relative_to(root)}")


if __name__ == "__main__":
    main()
