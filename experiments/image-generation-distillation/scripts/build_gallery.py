"""Build blind and revealed HTML galleries from an exported review set."""
import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.review_dir
    key = json.loads((root / "identity-key.json").read_text(encoding="utf-8"))
    with (root / "blind/review.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(dict)
    for row in rows:
        code = row["review_id"]
        groups[key[code]["id"]][key[code]["arm"]] = (code, row["prompt"])
    style = """<meta charset="utf-8"><style>
    body{font:16px system-ui,sans-serif;margin:2rem auto;max-width:1250px;padding:0 1rem;color:#15242a;background:#f4f5f1}
    article{background:white;padding:1rem;margin:1rem 0;border-radius:12px}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}
    img{width:100%;height:auto;border-radius:6px} small{color:#526368}
    @media(max-width:650px){.grid{grid-template-columns:1fr}}
    </style>"""
    blind = [style, "<title>Blind review</title><h1>匿名盲评</h1>",
             "<p>按 review.csv 填写 1、0 或留空。图片顺序已打乱。</p><div class=grid>"]
    for row in rows:
        code = html.escape(row["review_id"])
        prompt = html.escape(row["prompt"])
        blind.append(f'<article><img src="blind/images/{code}.png"><small>{code}</small><p>{prompt}</p></article>')
    blind.append("</div>")
    (root / "blind-gallery.html").write_text("".join(blind), encoding="utf-8")
    revealed = [style, "<title>Paired comparisons</title><h1>留出提示词配对对照</h1>",
                "<p>揭盲页面：先完成匿名盲评再浏览。三组对同一提示词和种子生成。</p>"]
    labels = (("teacher", "教师 SDXL"), ("student-before", "学生微调前"), ("student-after", "学生微调后"))
    for identifier, arms in sorted(groups.items()):
        if set(arms) != {arm for arm, _ in labels}:
            raise ValueError(f"incomplete group {identifier}")
        prompt = html.escape(arms["teacher"][1])
        revealed.append(f"<article><h2>{html.escape(identifier)}</h2><p>{prompt}</p><div class=grid>")
        for arm, label in labels:
            code = html.escape(arms[arm][0])
            revealed.append(f'<div><strong>{label}</strong><img src="blind/images/{code}.png"></div>')
        revealed.append("</div></article>")
    (root / "comparison.html").write_text("".join(revealed), encoding="utf-8")
    print(f"Built galleries for {len(rows)} images and {len(groups)} matched prompts.")


if __name__ == "__main__":
    main()
