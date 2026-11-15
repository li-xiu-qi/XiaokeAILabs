"""
Table Reconstruction Assembler: Spatial Join of OCR Texts and Table Structure
Aligns text bounding boxes from PaddleOCR with HTML structure tags from SLANet.
"""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble OCR texts with SLANet table structure")
    parser.add_argument("--ocr-json", required=True, help="Path to PaddleOCR recognition.json")
    parser.add_argument("--structure-json", required=True, help="Path to SLANet cropped.json or uncropped.json")
    parser.add_argument("--out-dir", required=True, help="Directory to save assembled results")
    parser.add_argument("--crop-offset", nargs=2, type=int, default=[0, 0], help="Offset (x, y) if OCR or structure was cropped")
    parser.add_argument("--row-threshold-px", type=float, default=25.0, help="Y-distance threshold for row clustering")
    return parser.parse_args()


def cluster_and_sort_boxes(ocr_items, offset_x=0, offset_y=0, row_threshold=25.0):
    """
    Computes box centers, normalizes offsets, clusters into rows by Y, and sorts by X.
    """
    boxes_with_text = []
    for item in ocr_items:
        box = item["box"]
        text = item["prediction"]["text"]
        score = item["prediction"]["score"]
        cx = sum(p[0] for p in box) / len(box) + offset_x
        cy = sum(p[1] for p in box) / len(box) + offset_y
        boxes_with_text.append({
            "text": text,
            "score": score,
            "center": (cx, cy),
            "box": box
        })

    # Sort primarily by cy
    boxes_with_text.sort(key=lambda b: b["center"][1])

    # Cluster into rows
    rows = []
    for b in boxes_with_text:
        placed = False
        for row in rows:
            # Check average Y of the row
            avg_y = sum(elem["center"][1] for elem in row) / len(row)
            if abs(b["center"][1] - avg_y) <= row_threshold:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    # Sort each row by cx (left to right)
    for row in rows:
        row.sort(key=lambda b: b["center"][0])

    return rows


def assemble_html(structure_tokens, rows):
    """
    Fills text into structure tokens replacing empty <td></td> slots sequentially.
    """
    flat_texts = []
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row):
            flat_texts.append(cell["text"])

    text_idx = 0
    assembled_tokens = []
    for tok in structure_tokens:
        if tok == "<td></td>":
            if text_idx < len(flat_texts):
                assembled_tokens.append(f"<td>{flat_texts[text_idx]}</td>")
                text_idx += 1
            else:
                assembled_tokens.append("<td></td>")
        else:
            assembled_tokens.append(tok)

    html_content = "".join(assembled_tokens)
    return html_content, text_idx == len(flat_texts) == structure_tokens.count("<td></td>")


def generate_markdown_table(rows):
    """
    Builds a markdown table from clustered rows.
    """
    if not rows:
        return ""
    
    col_count = max(len(r) for r in rows)
    lines = []
    
    # Header row
    header = [cell["text"] for cell in rows[0]]
    while len(header) < col_count:
        header.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * col_count) + " |")

    # Body rows
    for row in rows[1:]:
        body = [cell["text"] for cell in row]
        while len(body) < col_count:
            body.append("")
        lines.append("| " + " | ".join(body) + " |")

    return "\n".join(lines)


def main():
    args = parse_args()
    ocr_path = Path(args.ocr_json)
    struct_path = Path(args.structure_json)
    out_dir = Path(args.out_dir)
    outputs = [out_dir / name for name in ("assembled_table.html", "assembled_table.md", "assembled_result.json")]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
    struct_data = json.loads(struct_path.read_text(encoding="utf-8"))

    structure_tokens = struct_data["result"]["structure"]

    # Cluster OCR items
    rows = cluster_and_sort_boxes(
        ocr_data,
        offset_x=args.crop_offset[0],
        offset_y=args.crop_offset[1],
        row_threshold=args.row_threshold_px
    )

    # Assemble HTML
    assembled_html, match_ok = assemble_html(structure_tokens, rows)
    markdown_table = generate_markdown_table(rows)

    summary = {
        "status": "success",
        "all_cells_matched": match_ok,
        "detected_rows": len(rows),
        "detected_cols_per_row": [len(r) for r in rows],
        "total_cells": sum(len(r) for r in rows),
        "markdown": markdown_table,
        "assembled_html": assembled_html,
        "grid_data": [
            [{"text": c["text"], "center": [round(c["center"][0], 1), round(c["center"][1], 1)], "score": round(c["score"], 4)} for c in r]
            for r in rows
        ]
    }

    # Save outputs
    (out_dir / "assembled_table.html").write_text(assembled_html, encoding="utf-8")
    (out_dir / "assembled_table.md").write_text(markdown_table, encoding="utf-8")
    (out_dir / "assembled_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Assembly completed successfully:")
    print(f"- Detected Rows: {len(rows)}")
    print(f"- Total Cells: {summary['total_cells']}")
    print(f"- Perfect Match: {match_ok}")
    print("\n[Generated Markdown Table]")
    print(markdown_table)


if __name__ == "__main__":
    main()
