#!/usr/bin/env bash
# 从已下载图片的文件名（objectID）反查 Met 元数据，不必等下载完
set -u
PY=/home/ke/jevomni-env/bin/python
SRC=/home/ke/thangka_fetch

"$PY" - "$SRC" <<'PY'
import json, os, re, sys, time, urllib.request

SRC = sys.argv[1]
API = "https://collectionapi.metmuseum.org/public/collection"
UA = "XiaokeAILabs-style-lora-probe/1.0 (learning experiment)"

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

ids = sorted({int(m.group(1)) for f in os.listdir(SRC)
              if (m := re.match(r"m_(\d+)\.jpg$", f))})
print("本地图片 objectID 数:", len(ids), flush=True)

out, bad = [], 0
for i, oid in enumerate(ids):
    try:
        o = get("%s/v1/objects/%d" % (API, oid))
    except Exception as e:
        bad += 1
        print("  取详情失败 %d: %s" % (oid, str(e)[:50]), flush=True)
        continue
    out.append({
        "objectID": oid, "file": "m_%06d.jpg" % oid,
        "title": o.get("title"), "culture": o.get("culture"),
        "date": o.get("objectDate"), "medium": o.get("medium"),
        "dims": o.get("dimensions"), "acc": o.get("accessionNumber"),
        "dept": o.get("department"), "url": o.get("objectURL"),
        "public": o.get("isPublicDomain"),
    })
    if (i + 1) % 15 == 0:
        print("  已取 %d 条元数据 ..." % (i + 1), flush=True)
    time.sleep(0.1)

json.dump(out, open(os.path.join(SRC, "meta_from_ids.json"), "w"),
          ensure_ascii=False, indent=1)
print("\n元数据 %d 条（失败 %d）-> meta_from_ids.json" % (len(out), bad), flush=True)

# 概览
import collections
print("\n--- 部门分布 ---")
for k, v in collections.Counter(m["dept"] for m in out).most_common():
    print("%5d  %s" % (v, k))
print("--- 文化标注 ---")
for k, v in collections.Counter((m["culture"] or "(未标)") for m in out).most_common(8):
    print("%5d  %s" % (v, k))
print("--- 公开域 ---")
print(collections.Counter(bool(m["public"]) for m in out))
print("--- 无高清图的样本 ---")
print(sum(1 for m in out if not m.get("acc")))
PY
echo "META_DONE"
