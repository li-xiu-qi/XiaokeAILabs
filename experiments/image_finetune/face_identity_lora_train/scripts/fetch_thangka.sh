#!/usr/bin/env bash
# 测 proart 到各数据源的连通性，顺便把唐卡抓回来
set -u
PY=/home/ke/jevomni-env/bin/python
OUT=/home/ke/thangka_fetch
mkdir -p "$OUT"

cat > /tmp/fetch_thangka.py <<'PY'
import json, os, sys, time, hashlib, urllib.request, urllib.parse

API = "https://collectionapi.metmuseum.org/public/collection"
UA = "XiaokeAILabs-style-lora-probe/1.0 (learning experiment)"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

QUERIES = ["thangka", "Tibetan thangka", "Tibetan painting", "Tibetan Buddhist painting"]

def get(url, timeout=40, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read() if binary else json.loads(r.read().decode("utf-8"))

# 1) 搜集 objectID
ids = []
for q in QUERIES:
    try:
        d = get(API + "/v1.1/search?" + urllib.parse.urlencode({"q": q, "limit": 100}))
        got = d.get("objectIDs") or []
        print("检索 %-28s -> %d 件" % (q, len(got)), flush=True)
        ids += got
    except Exception as e:
        print("检索失败 %-24s : %s" % (q, str(e)[:70]), flush=True)
    time.sleep(0.4)
ids = list(dict.fromkeys(ids))
print("去重后候选 objectID：%d" % len(ids), flush=True)

# 2) 逐个取详情，筛可用样本
kept, dropped = [], {"no_img": 0, "not_public": 0, "small": 0, "dup": 0, "err": 0}
seen_hash = set()
seen_acc = set()
for i, oid in enumerate(ids):
    try:
        o = get("%s/v1/objects/%d" % (API, oid), timeout=40)
    except Exception:
        dropped["err"] += 1
        continue
    img = o.get("primaryImage") or ""
    if not img:
        dropped["no_img"] += 1
        continue
    if not o.get("isPublicDomain"):
        dropped["not_public"] += 1
        continue
    acc = o.get("accessionNumber") or ""
    if acc and acc in seen_acc:          # 同一件藏品只留一条
        dropped["dup"] += 1
        continue
    try:
        raw = get(img, timeout=60, binary=True)
    except Exception:
        dropped["err"] += 1
        continue
    if len(raw) < 60000:                  # 太小，多半是缩略占位
        dropped["small"] += 1
        continue
    h = hashlib.md5(raw).hexdigest()
    if h in seen_hash:
        dropped["dup"] += 1
        continue
    seen_hash.add(h)
    if acc:
        seen_acc.add(acc)
    name = "m_%06d.jpg" % oid
    with open(os.path.join(OUT, name), "wb") as fh:
        fh.write(raw)
    kept.append({"objectID": oid, "file": name, "title": o.get("title"),
                 "culture": o.get("culture"), "date": o.get("objectDate"),
                 "medium": o.get("medium"), "dims": o.get("dimensions"),
                 "acc": acc, "dept": o.get("department"),
                 "url": o.get("objectURL"), "bytes": len(raw)})
    if len(kept) % 10 == 0:
        print("  已收 %d 张 ..." % len(kept), flush=True)
    time.sleep(0.15)

json.dump(kept, open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False, indent=1)
print("\n保留 %d 张，剔除 %s" % (len(kept), dropped), flush=True)
print("FETCH_DONE", flush=True)
PY

echo "########## 连通性 ##########"
curl -s -o /dev/null -w "met_api %{http_code}\n" --max-time 25 "https://collectionapi.metmuseum.org/public/collection/v1.1/search?q=thangka&limit=1"
curl -s -o /dev/null -w "met_img %{http_code}\n" --max-time 30 "https://images.metmuseum.org/CRDImages/as/original/1970_298_1_O2.jpg"
curl -s -o /dev/null -w "commons %{http_code}\n" --max-time 20 "https://commons.wikimedia.org/w/api.php?action=query&format=json&meta=siteinfo"

echo
echo "########## 抓唐卡 ##########"
"$PY" /tmp/fetch_thangka.py "$OUT"
