#!/usr/bin/env bash
# 读 CelebA parquet，统计 celeb_id 频次，挑样本量合适的身份
set -u
PY=/home/ke/jevomni-env/bin/python
SRC=/home/ke/celeba_shard0.parquet
OUT=/home/ke/celeba_pick
mkdir -p "$OUT"

"$PY" - "$SRC" "$OUT" <<'PY'
import json, os, sys, collections
import pyarrow.parquet as pq

SRC, OUT = sys.argv[1], sys.argv[2]
t = pq.read_table(SRC)
print("schema:", t.schema.names[:6], "...", t.schema.names[-3:])
print("行数:", t.num_rows)

ids = t.column("celeb_id").to_pylist()
cnt = collections.Counter(ids)
print("不同身份数:", len(cnt))

# 候选：15~40 张之间的身份（够训练又不至于挑花眼）
cand = [(cid, n) for cid, n in cnt.items() if 15 <= n <= 40]
cand.sort(key=lambda x: -x[1])
print("样本量 15~40 的身份数:", len(cand))
print("前 12 个候选：")
for cid, n in cand[:12]:
    print("   celeb_id=%-8d %d 张" % (cid, n))

# 同时看看最多的几个是多少（用于说明数据分布）
top = cnt.most_common(5)
print("样本最多的 5 个身份:", top)

json.dump({"candidates": [{"celeb_id": c, "n": n} for c, n in cand[:40]],
           "n_identities": len(cnt),
           "rows": t.num_rows,
           "top": top},
          open(os.path.join(OUT, "candidates.json"), "w"), indent=1)
print("\n候选清单 -> candidates.json")
PY
echo "PICK_DONE"
