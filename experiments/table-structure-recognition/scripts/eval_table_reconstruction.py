"""
Table Reconstruction Evaluation and Robustness Benchmark
Evaluates spatial grid assembly accuracy, structural fidelity (TEDS proxy),
and failure modes under noisy OCR / structure detections.
"""

import argparse
import json
import time
from pathlib import Path


def compute_token_distance(s1, s2):
    """Simple Levenshtein distance on token lists for structure fidelity."""
    if len(s1) < len(s2):
        return compute_token_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def evaluate_assembly_accuracy(gt_grid, pred_grid):
    """
    Compares ground truth cell texts with assembled cell texts.
    Returns cell accuracy and structure match.
    """
    gt_rows = len(gt_grid)
    pred_rows = len(pred_grid)
    if gt_rows == 0:
        return 0.0, False

    correct_cells = 0
    total_cells = sum(len(r) for r in gt_grid)

    for r_idx in range(min(gt_rows, pred_rows)):
        gt_r = gt_grid[r_idx]
        pred_r = pred_grid[r_idx]
        for c_idx in range(min(len(gt_r), len(pred_r))):
            if str(gt_r[c_idx]).strip().lower() == str(pred_r[c_idx]).strip().lower():
                correct_cells += 1

    cell_acc = correct_cells / total_cells if total_cells > 0 else 0.0
    structure_exact = (gt_rows == pred_rows) and all(len(gt_grid[i]) == len(pred_grid[i]) for i in range(gt_rows))

    return cell_acc, structure_exact


def simulate_noise_and_assemble(ocr_items, structure_tokens, noise_type="clean"):
    """
    Simulates real-world noise (dropped box, duplicate box, extra structure row)
    to measure assembly breakdown and cascade error.
    """
    from assemble_table import cluster_and_sort_boxes, assemble_html

    items = [dict(x) for x in ocr_items]
    struct = list(structure_tokens)

    if noise_type == "drop_one_text":
        # Simulate OCR missing a cell text (e.g. dropped second cell 'Units')
        items = [item for item in items if item["prediction"]["text"] != "Units"]
    elif noise_type == "extra_structure_row":
        # Simulate SLANet hallucinating an extra row (as seen in uncropped image)
        extra_row = ["<tr>", "<td></td>", "<td></td>", "<td></td>", "</tr>"]
        # Insert before </table>
        table_close_idx = len(struct) - 3
        struct = struct[:table_close_idx] + extra_row + struct[table_close_idx:]

    t0 = time.perf_counter()
    rows = cluster_and_sort_boxes(items)
    assembled_html, match_ok = assemble_html(struct, rows)
    elapsed = time.perf_counter() - t0

    # Extract pred grid
    pred_grid = [[c["text"] for c in r] for r in rows]

    return {
        "noise_type": noise_type,
        "elapsed_ms": elapsed * 1000,
        "assembled_html": assembled_html,
        "match_ok": match_ok,
        "pred_grid": pred_grid,
        "pred_cells_count": sum(len(r) for r in pred_grid),
        "structure_slots_count": struct.count("<td></td>")
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-json", type=Path, default=root / "fixtures" / "ocr-recognition.json")
    parser.add_argument("--structure-json", type=Path, default=root / "results" / "cropped.json")
    parser.add_argument("--output", type=Path, required=True, help="New output JSON file; existing files are not overwritten")
    args = parser.parse_args()
    ocr_path, struct_path, out_path = args.ocr_json, args.structure_json, args.output
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")

    ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
    struct_data = json.loads(struct_path.read_text(encoding="utf-8"))

    gt_grid = [
        ["Item", "Units", "Price"],
        ["Notebook", "12", "6.50"],
        ["Marker", "24", "2.00"],
        ["Paper", "8", "4.00"]
    ]

    scenarios = ["clean", "drop_one_text", "extra_structure_row"]
    benchmark_results = {}

    for sc in scenarios:
        res = simulate_noise_and_assemble(ocr_data, struct_data["result"]["structure"], noise_type=sc)
        cell_acc, ocr_grid_match = evaluate_assembly_accuracy(gt_grid, res["pred_grid"])
        # This fixture contains a rectangular table without merged cells. Check
        # predicted structure as well as the OCR grid; row/slot counts alone
        # are not a general structural-equivalence metric.
        structure_row_count = res["assembled_html"].count("<tr>")
        struct_match = (ocr_grid_match and structure_row_count == len(gt_grid)
                        and res["structure_slots_count"] == sum(map(len, gt_grid)))

        benchmark_results[sc] = {
            "noise_type": sc,
            "elapsed_ms": round(res["elapsed_ms"], 3),
            "cell_accuracy": round(cell_acc, 4),
            "structure_exact": struct_match,
            "all_slots_filled": res["match_ok"],
            "pred_cells": res["pred_cells_count"],
            "expected_slots": res["structure_slots_count"],
            "cascade_error_observed": not res["match_ok"] or cell_acc < 1.0
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as output:
        json.dump(benchmark_results, output, ensure_ascii=False, indent=2)
    print("Benchmark completed. Results:")
    print(json.dumps(benchmark_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
