#!/usr/bin/env bash
# 测试集单独放大：这 10 张不参与训练，只用于评估微调有没有学到这个人
set -u
PY=/home/ke/jevomni-env/bin/python
SRC=/home/ke/celeb_person_14
OUT=/mnt/c/Users/ke/celeb_sheet
mkdir -p "$OUT"

"$PY" - "$SRC" "$OUT" <<'PY'
import os, sys, glob, json
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]

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

for split, label, cols in (("test", "测试集（不参与训练）", 5),
                           ("train", "训练集（进了 LoRA）", 5)):
    fs = sorted(glob.glob(os.path.join(SRC, split, "*.jpg")))
    CW, CH, pad, cap = 300, 372, 14, 40
    rows = (len(fs) + cols - 1) // cols
    W = cols * CW + (cols + 1) * pad
    H = rows * (CH + cap) + (rows + 1) * pad + 76
    sh = Image.new("RGB", (W, H), "#f7f5f0")
    d = ImageDraw.Draw(sh)
    col = "#d98324" if split == "test" else "#2c6fbb"
    d.text((pad, 16), "celeb_id=14  %s  共 %d 张" % (label, len(fs)),
           fill=col, font=font(22, True))
    d.text((pad, 46), "原图 178x218，CelebA 原图已做人脸对齐裁剪，全部放大 1.7 倍显示。",
           fill="#666", font=font(14))
    for i, f in enumerate(fs):
        r, c = divmod(i, cols)
        x = pad + c * (CW + pad)
        y = 66 + pad + r * (CH + cap + pad)
        im = Image.open(f).convert("RGB")
        im = im.resize((CW - 24, int(im.height * (CW - 24) / im.width)))
        ox = x + (CW - im.width) // 2
        oy = y + (CH - 24 - im.height) // 2
        sh.paste(im, (ox, oy))
        d.rectangle([x, y, x + CW, y + CH - 24], outline=col, width=3)
        d.text((x + 2, y + CH - 18), os.path.basename(f), fill="#555", font=font(13, True))
    out = os.path.join(OUT, "celeb_14_%s.jpg" % split)
    sh.save(out, quality=93)
    print("生成", out, sh.size)
PY
ls -la "$OUT"
