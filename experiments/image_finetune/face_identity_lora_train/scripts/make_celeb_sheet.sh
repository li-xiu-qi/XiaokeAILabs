#!/usr/bin/env bash
# 把选中身份的训练/测试图拼成联系表，供训练前审阅
set -u
PY=/home/ke/jevomni-env/bin/python
CID=${1:-14}
SRC=/home/ke/celeb_person_${CID}
OUT=/mnt/c/Users/ke/celeb_sheet
mkdir -p "$OUT"

"$PY" - "$SRC" "$OUT" <<'PY'
import json, os, sys, glob
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(SRC, "manifest.json"), encoding="utf-8"))
items = man["items"]

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

#  CelebA 原图 178x218，放大到 260 方便看
CW, CH, pad = 260, 320, 14
cols = 6
rows = (len(items) + cols - 1) // cols
W = cols * CW + (cols + 1) * pad
H = rows * CH + (rows + 1) * pad + 108
sh = Image.new("RGB", (W, H), "#f4f1ea")
d = ImageDraw.Draw(sh)

d.text((pad, 14), "候选身份 CelebA celeb_id=%d  共 %d 张" % (man["celeb_id"], len(items)),
       fill="#222", font=font(24, True))
d.text((pad, 46), "蓝框 = 训练集 %d 张（会进 LoRA）    橙框 = 测试集 %d 张（不参与训练，只用于评估）"
       % (man["n_train"], man["n_test"]), fill="#444", font=font(16))
d.text((pad, 72), "原图 178x218。评估指标：生成图与该人测试集的人脸相似度。",
       fill="#777", font=font(14))

for i, it in enumerate(items):
    r, c = divmod(i, cols)
    x = pad + c * (CW + pad)
    y = 92 + pad + r * (CH + pad)
    p = os.path.join(SRC, it["split"], it["file"])
    im = Image.open(p).convert("RGB")
    im = im.resize((CW - 20, int(im.height * (CW - 20) / im.width)))
    ox = x + (CW - im.width) // 2
    oy = y + (CH - 46 - im.height) // 2
    sh.paste(im, (ox, oy))
    col = "#2c6fbb" if it["split"] == "train" else "#d98324"
    d.rectangle([x, y, x + CW, y + CH - 40], outline=col, width=4)
    d.text((x + 4, y + CH - 34), it["file"], fill="#333", font=font(13, True))
    d.text((x + 4, y + CH - 18),
           "训练" if it["split"] == "train" else "测试",
           fill=col, font=font(13, True))

out = os.path.join(OUT, "celeb_%d_sheet.jpg" % man["celeb_id"])
sh.save(out, quality=92)
print("生成", out, sh.size)
PY
ls -la "$OUT"
