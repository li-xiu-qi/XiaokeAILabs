#!/usr/bin/env python3
"""解析矩阵测试原始输出，按 (配置, 任务) 聚合吞吐与接受率。

原始文件格式（run-matrix.sh 生成）：
    ### <tag>|<label>|<task>|round<N>
    { ...bench.py 输出的 JSON... }
    ### <tag>|<label>|END

用法：
    python scripts/parse_matrix.py matrix_35b.txt
    python scripts/parse_matrix.py matrix_35b.txt --csv
"""

import argparse
import json
import re
import statistics
import sys

HEADER = re.compile(r"^### ([^|]+)\|([^|]+)\|(.+)$")


def parse(path):
    """返回 records: list of dict(label, task, round, metrics)"""
    records = []
    cur = None
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = HEADER.match(line)
            if m:
                # 收尾上一条
                if cur and buf:
                    metrics = _load_json("".join(buf))
                    if metrics:
                        cur["metrics"] = metrics
                        records.append(cur)
                buf = []
                tag, label, tail = m.groups()
                if tail == "END" or tail.startswith("READY="):
                    cur = None
                else:
                    mm = re.match(r"(.+)\|round(\d+)$", tail)
                    if mm:
                        task, rnd = mm.groups()
                        cur = {
                            "tag": tag,
                            "label": label,
                            "task": task,
                            "round": int(rnd),
                        }
                    else:
                        cur = None
            elif cur is not None:
                buf.append(line)
    if cur and buf:
        metrics = _load_json("".join(buf))
        if metrics:
            cur["metrics"] = metrics
            records.append(cur)
    return records


def _load_json(text):
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def summarize(records):
    """按 (label, task) 聚合。返回 rows 列表。"""
    groups = {}
    for r in records:
        key = (r["label"], r["task"])
        groups.setdefault(key, []).append(r["metrics"])

    rows = []
    for (label, task), ms in sorted(groups.items()):
        gens = [m["gen_tps"] for m in ms if m.get("gen_tps")]
        pre = [m["prefill_tps"] for m in ms if m.get("prefill_tps")]
        acc = [m["accept_rate"] for m in ms if m.get("accept_rate") is not None]
        drafts = [m.get("draft_n", 0) for m in ms]
        rows.append(
            {
                "label": label,
                "task": task,
                "n": len(ms),
                "gen_list": [round(g, 2) for g in gens],
                "gen_mean": round(statistics.mean(gens), 2) if gens else 0,
                "gen_sd": round(statistics.stdev(gens), 2) if len(gens) > 1 else 0,
                "prefill_mean": round(statistics.mean(pre), 2) if pre else 0,
                "acc_mean": round(statistics.mean(acc) * 100, 1) if acc else None,
                "draft_total": sum(drafts),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    records = parse(args.path)
    if not records:
        print("未解析到任何记录", file=sys.stderr)
        return 1
    rows = summarize(records)

    if args.csv:
        print("config,task,n,gen_mean,gen_sd,prefill_mean,acc%,draft_total")
        for r in rows:
            print(
                f"{r['label']},{r['task']},{r['n']},{r['gen_mean']},{r['gen_sd']},"
                f"{r['prefill_mean']},{r['acc_mean']},{r['draft_total']}"
            )
        return 0

    tag = records[0]["tag"]
    print(f"=== {tag}：按配置 × 任务聚合（gen 为三轮均值）===")
    print(f"{'配置':<16}{'任务':<18}{'n':>3}{'gen均值':>10}{'标准差':>9}{'prefill':>10}{'接受率%':>10}{'草稿数':>8}")
    for r in rows:
        acc = f"{r['acc_mean']:.1f}" if r["acc_mean"] is not None else "-"
        print(
            f"{r['label']:<16}{r['task']:<18}{r['n']:>3}{r['gen_mean']:>10}"
            f"{r['gen_sd']:>9}{r['prefill_mean']:>10}{acc:>10}{r['draft_total']:>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
