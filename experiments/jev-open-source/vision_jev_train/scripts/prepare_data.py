"""把 ScienceQA 转成视觉决策训练用的 jsonl，并按图片内容做可复现切分。

与 text_jev_train/prepare_data.py 的差异：
1. 每条样本带一张图，图像以 base64 PNG 内嵌进 parquet，不依赖额外下载 COCO
2. 切分主键从 case id 换成图片 sha256。ScienceQA 的图在 train 与 test 之间
   大量复用（实测 2529 张跨划分重复），若只按行 id 切分会把同一张图
   同时放进训练集和测试集，退化成背图，指标虚高
3. 先占 test，再依次占 train、dev、calibration，保证去重顺序可复现

产出：
  prepared/{train,dev,calibration,test}.jsonl
  split_manifest.json（含每个划分的图片 sha256 全表与源文件 sha256）

每行字段：
  image_png_b64   图像 base64
  state           题目上下文（说明、选项、提示、标准答案推理的合并文本）
  question        判别问题
  options         候选列表
  gold            概率分布，gold[answer]=1.0 其余 0
"""
import base64
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = "20260925"
TRAIN_CASES = 3000
DEV_CASES = 400
CALIB_CASES = 400
TEST_CASES = 1000

SPLIT_ORDER = ["test", "train", "dev", "calibration"]


def image_bytes(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def build_state(row):
    """ScienceQA 没有显式 state 字段，用题目上下文构造。

    顺序固定为：说明、选项、提示、标准答案推理。选项用字母编号，与
    question 里的引用保持一致。
    """
    parts = []
    lecture = (row.get("lecture") or "").strip()
    if lecture:
        parts.append(f"讲解：{lecture}")
    choices = row.get("choices") or []
    letters = "ABCDEFGH"
    option_text = "\n".join(
        f"{letters[i] if i < len(letters) else i}. {c}" for i, c in enumerate(choices)
    )
    parts.append(f"备选答案：\n{option_text}")
    hint = (row.get("hint") or "").strip()
    if hint:
        parts.append(f"提示：{hint}")
    solution = (row.get("solution") or "").strip()
    if solution:
        parts.append(f"参考答案推理：{solution}")
    return "\n\n".join(parts)


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

    return {
        "image_sha256": sha256_hex(img_bytes),
        "image_png_b64": base64.b64encode(img_bytes).decode("ascii"),
        "state": build_state(row),
        "question": build_question(row),
        "options": [str(c) for c in choices],
        "gold": [v / sum(gold) for v in gold],
        "task": row.get("task"),
        "grade": row.get("grade"),
        "subject": row.get("subject"),
        "source": "ScienceQA",
        "license": "CC BY-SA 4.0",
        "label_semantics": "multiple-choice gold; not measured action success",
    }


def main():
    import pyarrow.parquet as pq

    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        src = ROOT / "data" / "all" / "test-00000-of-00001.parquet"
    out_dir = ROOT / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(src)
    rows = table.to_pylist()
    print(f"[load] {src} rows={len(rows)}")

    quotas = {
        "test": TEST_CASES,
        "train": TRAIN_CASES,
        "dev": DEV_CASES,
        "calibration": CALIB_CASES,
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
        "seed": SEED,
        "source": str(src),
        "source_sha256": sha256_hex(src.read_bytes()) if src.exists() else None,
        "selection_metric": "dev soft cross entropy",
        "test_usage": "final evaluation only",
        "dedup_key": "image sha256, images never cross splits",
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
    (ROOT / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[manifest] split_manifest.json written")


if __name__ == "__main__":
    main()
