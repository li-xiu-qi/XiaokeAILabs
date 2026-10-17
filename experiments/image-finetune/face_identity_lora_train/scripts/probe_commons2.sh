#!/usr/bin/env bash
# 候选风格数据摸底 v2：沿用 fetch_commons.py 里验证过的 API 写法
set -u
PY=/home/ke/jevomni-env/bin/python

cat > /tmp/probe_commons2.py <<'PY'
import json, time, urllib.parse, urllib.request, sys

API = "https://commons.wikimedia.org/w/api.php"
UA = "XiaokeAILabs-style-lora-probe/1.0 (learning experiment; contact: local)"

CANDIDATES = {
    "thangka":      ["Category:Thangkas", "Category:Tibetan thangkas", "Category:Buddhist paintings of Tibet"],
    "nianhua":      ["Category:Chinese New Year prints", "Category:Yangliuqing", "Category:Taohuawu"],
    "persian_mini": ["Category:Persian miniatures", "Category:Persian paintings"],
    "african_mask": ["Category:African masks", "Category:Masks of Africa"],
    "kuniyoshi":    ["Category:Utagawa Kuniyoshi"],
    "gongbi":       ["Category:Chinese paintings of birds and flowers", "Category:Gongbi paintings"],
    "lubok":        ["Category:Lubok prints", "Category:Russian popular prints"],
}

def api_get_once(params, timeout=45):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def api_get(params, tries=4, timeout=45):
    last = None
    for i in range(tries):
        try:
            return api_get_once(params, timeout)
        except Exception as e:
            last = e
            print("    API 失败 %d 次: %s" % (i + 1, str(e)[:80]), flush=True)
            time.sleep(2.0 * (i + 1))
    print("    放弃: %s" % str(last)[:80], flush=True)
    return {}

print("%-14s %-46s %7s %9s %9s" % ("风格", "分类", "条目", ">=1400px", "竖幅占比"))
print("-" * 92)
report = {}
for name, cats in CANDIDATES.items():
    tot = big = portrait = 0
    for cat in cats:
        d = api_get({"action": "query", "format": "json",
                     "generator": "categorymembers", "gcmtitle": cat,
                     "gcmtype": "file", "gcmlimit": "500",
                     "prop": "imageinfo", "iiprop": "url|size|mime",
                     "iiurlwidth": "1920"})
        pages = (d.get("query") or {}).get("pages") or {}
        if not pages:
            print("%-14s %-46s %7d  (无文件或分类不存在)" % (name, cat.replace("Category:", ""), 0), flush=True)
            continue
        n = len(pages)
        b = p = 0
        for pg in pages.values():
            ii = (pg.get("imageinfo") or [{}])[0]
            w, h = ii.get("width", 0), ii.get("height", 0)
            if w >= 1400 and h >= 1400:
                b += 1
            if h > w:
                p += 1
        tot += n; big += b; portrait += p
        print("%-14s %-46s %7d %9d %9d" % (name, cat.replace("Category:", ""), n, b, p), flush=True)
        time.sleep(0.5)
    report[name] = {"entries": tot, "big": big, "portrait": portrait}
    print("-" * 92, flush=True)

json.dump(report, open("/home/ke/commons_report2.json", "w"), ensure_ascii=False, indent=1)
print("已写入 /home/ke/commons_report2.json", flush=True)
PY

/home/ke/jevomni-env/bin/python /tmp/probe_commons2.py
echo "COMMONS2_DONE"
