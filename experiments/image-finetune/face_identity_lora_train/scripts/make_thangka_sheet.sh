#!/usr/bin/env bash
# 唐卡候选数据集联系表：带藏品号、年代、介质，供训练前人工审阅
set -u
PY=/home/ke/jevomni-env/bin/python
SRC=/home/ke/thangka_fetch
OUT=/mnt/c/Users/ke/thangka_sheet
mkdir -p "$OUT"

"$PY" - "$SRC" "$OUT" <<'PY'
import json, os, sys, glob
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]
META = "meta_from_ids.json"
if not os.path.exists(os.path.join(SRC, META)):
    META = "meta.json"
meta = json.load(open(os.path.join(SRC, META), encoding="utf-8"))
meta = sorted(meta, key=lambda m: m["file"])
print("样本数:", len(meta), "（来源 %s）" % META)

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
    p = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()

cell, pad, cap = 300, 12, 74
cols = 4
per = 24
chunks = [meta[i:i + per] for i in range(0, len(meta), per)]
print("联系表张数:", len(chunks))

for ci, chunk in enumerate(chunks):
    rows = (len(chunk) + cols - 1) // cols
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + cap) + (rows + 1) * pad + 74
    sh = Image.new("RGB", (W, H), "#f4f1ea")
    d = ImageDraw.Draw(sh)
    d.text((pad, 14), "唐卡候选数据集  第 %d/%d 批  共 %d 张（Met 开放藏品，未挂任何 LoRA）"
           % (ci + 1, len(chunks), len(meta)), fill="#222", font=font(21, True))
    d.text((pad, 42), "每格下方：藏品号 / 年代 / 尺寸。请标记要剔除的编号。",
           fill="#777", font=font(14))
    for i, m in enumerate(chunk):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = 62 + pad + r * (cell + cap + pad)
        p = os.path.join(SRC, m["file"])
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell))
            ox, oy = x + (cell - im.width) // 2, y + (cell - im.height) // 2
            sh.paste(im, (ox, oy))
        else:
            d.rectangle([x, y, x + cell, y + cell], fill="#ddd")
        d.rectangle([x, y, x + cell, y + cell], outline="#b8b0a0", width=1)
        acc = (m.get("acc") or "")[:22]
        dt = (m.get("date") or "")[:22]
        d.text((x + 2, y + cell + 4), acc, fill="#333", font=font(12, True))
        d.text((x + 2, y + cell + 20), dt, fill="#666", font=font(11))
        med = (m.get("medium") or "").replace("\n", " ")[:30]
        d.text((x + 2, y + cell + 35), med, fill="#999", font=font(10))
        cul = (m.get("culture") or "")[:18]
        d.text((x + 2, y + cell + 50), cul, fill="#999", font=font(10))
    out = os.path.join(OUT, "sheet_%02d.jpg" % (ci + 1))
    sh.save(out, quality=90)
    print("生成", out, sh.size)
PY

cp /mnt/c/Users/ke/thangka_sheet/*.jpg /mnt/c/Users/ke/ 2>/dev/null
ls -la /mnt/c/Users/ke/thangka_sheet/
