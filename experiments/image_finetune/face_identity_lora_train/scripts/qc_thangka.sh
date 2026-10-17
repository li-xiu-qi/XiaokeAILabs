#!/usr/bin/env bash
# 唐卡数据质检：亮度/饱和度/锐度/近重复，把该筛的挑出来并标注
set -u
PY=/home/ke/jevomni-env/bin/python
SRC=/home/ke/thangka_fetch
OUT=/mnt/c/Users/ke/thangka_qc
mkdir -p "$OUT"

"$PY" - "$SRC" "$OUT" <<'PY'
import json, os, re, sys, glob, math
from PIL import Image, ImageDraw, ImageFont, ImageStat

SRC, OUT = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(SRC, "meta_from_ids.json"), encoding="utf-8"))
meta = sorted(meta, key=lambda m: m["file"])

def font(sz, bold=False):
    cands = (["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"] if bold else [])
    cands += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    cands += glob.glob("/usr/share/fonts/**/NotoSansCJK*", recursive=True)
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()

# ---------- 指标 ----------
def ahash(im, s=12):
    g = im.convert("L").resize((s, s))
    px = list(g.getdata())
    avg = sum(px) / len(px)
    return "".join("1" if p > avg else "0" for p in px)

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

rows = []
for m in meta:
    p = os.path.join(SRC, m["file"])
    im = Image.open(p).convert("RGB")
    g = im.convert("L")
    st = ImageStat.Stat(g)
    mean = st.mean[0]
    std = st.stddev[0]
    # 彩色像素占比（粗略：饱和度 > 0.15）
    small = im.resize((160, 160))
    hsv = small.convert("HSV")
    sat = list(hsv.getdata())
    colored = sum(1 for _, s, _ in sat if s > 38) / len(sat)
    m["_mean"] = round(mean, 1)
    m["_std"] = round(std, 1)
    m["_colored"] = round(colored, 3)
    m["_hash"] = ahash(im)
    m["_size"] = im.size
    rows.append(m)

# 近重复：汉明距离 <= 4 视为同图
dups = []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        d = hamming(rows[i]["_hash"], rows[j]["_hash"])
        if d <= 4:
            dups.append((rows[i]["file"], rows[j]["file"], d))
print("近重复对（汉明<=4）:", len(dups))
for a, b, d in dups:
    print("   %s <-> %s  d=%d" % (a, b, d))

# 判定
def verdict(m):
    bad = []
    if m["_mean"] < 70:
        bad.append("过暗(%.0f)" % m["_mean"])
    if m["_std"] < 32:
        bad.append("对比低(%.0f)" % m["_std"])
    if m["_colored"] < 0.20:
        bad.append("近黑白(%.2f)" % m["_colored"])
    return bad

keep, drop = [], []
for m in rows:
    v = verdict(m)
    m["_flags"] = v
    (drop if v else keep).append(m)

print("\n保留 %d / %d，剔除 %d" % (len(keep), len(rows), len(drop)), flush=True)
print("--- 剔除明细 ---")
for m in drop:
    print("   %-16s %s" % (m["file"], "; ".join(m["_flags"])))

# 分布概览
import statistics
for k in ("_mean", "_std", "_colored"):
    vals = [m[k] for m in rows]
    print("%-9s 中位 %.1f  最小 %.1f  最大 %.1f" %
          (k, statistics.median(vals), min(vals), max(vals)))

json.dump({"keep": [m["file"] for m in keep],
           "drop": [{"file": m["file"], "flags": m["_flags"]} for m in drop],
           "all": [{k: v for k, v in m.items() if not k.startswith("_hash")} for m in rows]},
          open(os.path.join(OUT, "qc.json"), "w"), ensure_ascii=False, indent=1)
print("\n质检报告 -> qc.json", flush=True)

# ---------- 联系表（带标注） ----------
cell, pad, cap = 300, 12, 84
cols, per = 4, 24
chunks = [rows[i:i + per] for i in range(0, len(rows), per)]
for ci, chunk in enumerate(chunks):
    rws = (len(chunk) + cols - 1) // cols
    W = cols * cell + (cols + 1) * pad
    H = rws * (cell + cap) + (rws + 1) * pad + 78
    sh = Image.new("RGB", (W, H), "#f4f1ea")
    d = ImageDraw.Draw(sh)
    d.text((pad, 14), "唐卡候选数据质检  第 %d/%d 批  共 %d 张" % (ci + 1, len(chunks), len(rows)),
           fill="#222", font=font(21, True))
    d.text((pad, 42), "红框 = 建议剔除（过暗 / 对比低 / 近黑白）；橙字 = 近重复。绿框 = 保留。",
           fill="#a33", font=font(14, True))
    for i, m in enumerate(chunk):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = 62 + pad + r * (cell + cap + pad)
        p = os.path.join(SRC, m["file"])
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell))
        ox, oy = x + (cell - im.width) // 2, y + (cell - im.height) // 2
        sh.paste(im, (ox, oy))
        col = "#c0392b" if m["_flags"] else "#4a7c59"
        d.rectangle([x, y, x + cell, y + cell], outline=col, width=3)
        d.text((x + 2, y + cell + 4), (m.get("acc") or "")[:20], fill="#333", font=font(12, True))
        d.text((x + 2, y + cell + 20), "亮度%.0f 饱和%.2f" % (m["_mean"], m["_colored"]),
               fill="#666", font=font(11))
        if m["_flags"]:
            d.text((x + 2, y + cell + 36), "剔除: " + "; ".join(m["_flags"]),
                   fill="#c0392b", font=font(11, True))
        else:
            d.text((x + 2, y + cell + 36), (m.get("culture") or "")[:20], fill="#4a7c59", font=font(11))
        d.text((x + 2, y + cell + 52), (m.get("date") or "")[:26], fill="#999", font=font(10))
    out = os.path.join(OUT, "qc_sheet_%02d.jpg" % (ci + 1))
    sh.save(out, quality=90)
    print("生成", out, flush=True)
PY
echo "QC_DONE"
