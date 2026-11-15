"""把 ScienceQA 转成视觉决策训练用的 jsonl，并按图片内容做可复现切分。

与 agent-jev-text/scripts/prepare_data.py 的差异：
1. 每条样本带一张图，图像以 base64 PNG 内嵌进 parquet，不依赖额外下载 COCO
2. 切分主键从 case id 换成图片 sha256。ScienceQA 的图在 train 与 test 之间
   大量复用（实测 2529 张跨划分重复），若只按行 id 切分会把同一张图
   同时放进训练集和测试集，退化成背图，指标虚高
3. 先占 test，再依次占 train、dev、calibration，保证去重顺序可复现

产出：
  prepared-clean-v1/{train,dev,calibration,test}.jsonl
  prepared-clean-v1/split_manifest.json（含每个划分的图片 sha256 全表与源文件 sha256）

每行字段：
  image_png_b64   图像 base64
  state           题目选项和提示，不包含讲解或参考答案推理
  question        判别问题
  options         候选列表
  gold            概率分布，gold[answer]=1.0 其余 0
"""
import argparse
import base64
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

from data_schema import INPUT_SCHEMA, build_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_CASES = 400
DEV_CASES = 100
CALIB_CASES = 100
TEST_CASES = 100

SPLIT_ORDER = ["test", "train", "dev", "calibration"]


def image_bytes(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def build_question(row):
    """判别问题：给定图像与上下文，选出正确备选答案。"""
    return (
        f"{row.get('question', '')}\n"
        "请根据图像与上下文，选出正确的备选答案。"
    )


def convert(row):
    """一条 ScienceQA 样本转成一条决策样本。"""
    raw = row.get("image")
    if raw is None:
        return None
    if hasattr(raw, "convert"):
        img_bytes = image_bytes(raw)
    elif isinstance(raw, dict) and "bytes" in raw:
        img_bytes = raw["bytes"]
    else:
        return None

    choices = list(row.get("choices") or [])
    if len(choices) < 2:
        return None
    answer = row.get("answer")
    if not isinstance(answer, int) or not (0 <= answer < len(choices)):
        return None

    gold = [0.0] * len(choices)
    gold[answer] = 1.0

    options = [str(c) for c in choices]
    hint = (row.get("hint") or "").strip()

    return {
        "input_schema": INPUT_SCHEMA,
        "context_hint": hint,
        "image_sha256": sha256_hex(img_bytes),
        "image_png_b64": base64.b64encode(img_bytes).decode("ascii"),
        "state": build_state(options, hint),
        "question": build_question(row),
        "options": options,
        "gold": [v / sum(gold) for v in gold],
        "task": row.get("task"),
        "grade": row.get("grade"),
        "subject": row.get("subject"),
        "source": "ScienceQA",
        "license": "See ScienceQA dataset card: metadata and licensing text differ",
        "label_semantics": "multiple-choice gold; not measured action success",
    }


def resolve_source(path):
    """Resolve one local test split parquet; reject missing or ambiguous snapshots."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"ScienceQA parquet or snapshot directory not found: {path}")
    matches = sorted(p for p in path.rglob("test-*.parquet") if p.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one test-*.parquet under {path}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=str(ROOT / "data" / "all"))
    parser.add_argument("output", nargs="?", default=str(ROOT / "prepared-clean-v1"))
    parser.add_argument("--train-cases", type=int, default=TRAIN_CASES)
    parser.add_argument("--dev-cases", type=int, default=DEV_CASES)
    parser.add_argument("--calibration-cases", type=int, default=CALIB_CASES)
    parser.add_argument("--test-cases", type=int, default=TEST_CASES)
    args = parser.parse_args()
    if min(args.train_cases, args.dev_cases, args.calibration_cases, args.test_cases) < 1:
        parser.error("all split sizes must be positive")
    src = resolve_source(Path(args.source))
    import pyarrow.parquet as pq
    out_dir = Path(args.output)
    for name in (*[f"{split}.jsonl" for split in SPLIT_ORDER], "split_manifest.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"{out_dir / name} already exists; choose a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(src)
    rows = table.to_pylist()
    print(f"[load] {src} rows={len(rows)}")

    quotas = {
        "test": args.test_cases,
        "train": args.train_cases,
        "dev": args.dev_cases,
        "calibration": args.calibration_cases,
    }
    buckets = {k: [] for k in quotas}
    seen_sha = set()
    skipped_dup = 0
    skipped_bad = 0

    for split in SPLIT_ORDER:
        need = quotas[split]
        for row in rows:
            if len(buckets[split]) >= need:
                break
            item = convert(row)
            if item is None:
                skipped_bad += 1
                continue
            if item["image_sha256"] in seen_sha:
                skipped_dup += 1
                continue
            seen_sha.add(item["image_sha256"])
            buckets[split].append(item)

    shortages = {name: (len(buckets[name]), quotas[name]) for name in SPLIT_ORDER
                 if len(buckets[name]) < quotas[name]}
    if shortages:
        raise ValueError(f"Insufficient distinct valid images for requested split sizes: {shortages}. "
                         "Reduce --train-cases, --dev-cases, --calibration-cases or --test-cases.")

    for split in SPLIT_ORDER:
        items = buckets[split]
        path = out_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in items),
            encoding="utf-8",
        )
        cand = sum(len(s["options"]) for s in items)
        print(f"[split] {split:12s} cases={len(items):5d} candidates={cand:6d} "
              f"size={path.stat().st_size / 2**20:.1f}MB")

    print(f"[skip] duplicate_image={skipped_dup} invalid={skipped_bad}")

    manifest = {
        "selection_order": "source row order; test, train, dev, calibration",
        "input_schema": INPUT_SCHEMA,
        "source": str(src),
        "source_sha256": sha256_hex(src.read_bytes()) if src.exists() else None,
        "checkpoint_policy": "final training step; no checkpoint selection",
        "test_usage": "final evaluation only",
        "dedup_key": "image sha256, images never cross splits",
        "scan_duplicate_hits": skipped_dup,
        "scan_invalid_hits": skipped_bad,
        "scan_count_note": "Rows may be scanned more than once across split passes; counts are not unique rows.",
        "splits": {},
    }
    for split in SPLIT_ORDER:
        items = buckets[split]
        manifest["splits"][split] = {
            "cases": len(items),
            "candidates": sum(len(s["options"]) for s in items),
            "image_sha256": sorted({s["image_sha256"] for s in items}),
            "tasks": dict(Counter(s["task"] for s in items)),
        }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[manifest] {out_dir / 'split_manifest.json'} written")


if __name__ == "__main__":
    main()
