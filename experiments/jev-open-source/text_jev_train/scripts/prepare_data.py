"""把两个 parquet 转成训练用的 jsonl，并按案例做可复现切分。

切分在案例（case）层面做，先切再展开问题，保证同一案例的五个问题
必然落在同一个划分里。切分顺序由案例 id 加固定种子的 sha256 决定，
不依赖文件里的行序，任何人跑结果都一致。

产出：
  prepared/{train,dev,calibration,test}_{questions,requests}.jsonl
  split_manifest.json（含每个划分的案例 id 全表与源文件 sha256）
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_service.contract import prepare  # noqa: E402

SEED = "20260921"
DATASET_REVISION = "ea9306458d6e9563628369a3d1e72e362fb381d2"


def convert(row):
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    api = []
    for key, q in questions.items():
        kind = q["type"]
        entry = {
            "id": key,
            "type": "boolean" if kind == "noul" else kind,
            "question": q["instructions"],
        }
        entry[
            "options" if kind == "choice" else "levels" if kind == "score" else "criteria"
        ] = q.get("criteria", {})
        api.append(entry)
    request = {"id": row["id"], "state": json.loads(row["state"]), "questions": api}
    normalized = prepare(request)[0]
    items = []
    for q in normalized["questions"]:
        g = gold[q["id"]]["probabilities"]
        target = [float(g[k]) for k in q["keys"]]
        assert min(target) >= 0 and sum(target) > 0
        target = [v / sum(target) for v in target]
        items.append(
            {
                "id": row["id"] + ":" + q["id"],
                "case_id": row["id"],
                "workflow": row["workflow"],
                "state": normalized["state"],
                "question": q,
                "target": target,
                "label_semantics": "public teacher probability distribution; not measured action success",
            }
        )
    return request, items


def main():
    import pyarrow.parquet as pq

    data_dir = ROOT / "data" / "all"
    train_rows = pq.read_table(data_dir / "train-00000-of-00001.parquet").to_pylist()
    test_rows = pq.read_table(data_dir / "test-00000-of-00001.parquet").to_pylist()

    splits = {"train": [], "dev": [], "calibration": [], "test": test_rows}
    for workflow in sorted({r["workflow"] for r in train_rows}):
        rows = sorted(
            [r for r in train_rows if r["workflow"] == workflow],
            key=lambda r: hashlib.sha256((SEED + ":" + r["id"]).encode()).hexdigest(),
        )
        assert len(rows) == 300
        splits["train"] += rows[:240]
        splits["dev"] += rows[240:270]
        splits["calibration"] += rows[270:]

    manifest = {
        "seed": int(SEED),
        "selection_metric": "dev soft cross entropy",
        "test_usage": "final evaluation only",
        "dataset_revision": DATASET_REVISION,
        "splits": {},
    }
    seen_ids = set()
    seen_states = set()
    dest = ROOT / "prepared"
    dest.mkdir(exist_ok=True)
    for name, rows in splits.items():
        ids = {r["id"] for r in rows}
        states = {
            hashlib.sha256(
                json.dumps(json.loads(r["state"]), sort_keys=True).encode()
            ).hexdigest()
            for r in rows
        }
        # 跨划分的 id 与 state 内容都不许相撞，堵数据泄漏
        assert not seen_ids & ids
        assert not seen_states & states
        seen_ids |= ids
        seen_states |= states

        converted = [convert(row) for row in rows]
        items = [q for _, qs in converted for q in qs]
        for suffix, data in [
            ("questions", items),
            ("requests", [r for r, _ in converted]),
        ]:
            (dest / f"{name}_{suffix}.jsonl").write_text(
                "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in data),
                encoding="utf-8",
            )
        manifest["splits"][name] = {
            "cases": len(rows),
            "questions": len(items),
            "ids": sorted(ids),
            "workflows": dict(collections.Counter(r["workflow"] for r in rows)),
        }

    for file in data_dir.glob("*.parquet"):
        manifest.setdefault("source_sha256", {})[file.name] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
    (ROOT / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    summary = {
        k: {x: v[x] for x in ("cases", "questions", "workflows")}
        for k, v in manifest["splits"].items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
