#!/usr/bin/env bash
# 把底模探针图拼成联系表：每个候选风格一行，3 个 seed 并排，方便判断"底模能不能生成它"
set -u
PY=/home/ke/jevomni-env/bin/python
SRC=/home/ke/probe_candidates
OUT=/mnt/c/Users/ke/probe_sheet
mkdir -p "$OUT"

"$PY" - "$SRC" "$OUT" <<'PY'
import os, sys, glob
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]

ORDER = ["kuniyoshi", "thangka", "nianhua", "persian_mini", "african_mask", "gongbi"]
NOTE = {
    "kuniyoshi":    "对照组，已知底模能生成",
    "thangka":      "藏族唐卡",
    "nianhua":      "中国木版年画",
    "persian_mini": "波斯细密画",
    "african_mask": "非洲木面具",
    "gongbi":       "中国工笔花鸟",
}
SEEDS = [11, 22, 33]

def font(sz, bold=False):
    # 优先 Noto CJK（中文），退回 DejaVu
    cands = []
    if bold:
        cands += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                  "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"]
    cands += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]
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

cell, pad, cap = 300, 12, 76
cols = len(SEEDS)
W = cols * cell + (cols + 1) * pad
H = len(ORDER) * (cell + cap) + (len(ORDER) + 1) * pad + 96
sh = Image.new("RGB", (W, H), "#f4f1ea")
d = ImageDraw.Draw(sh)

d.text((pad, 16), "SDXL 底模生成能力探针", fill="#222", font=font(24, True))
d.text((pad, 48), "每一行是一个候选风格，同一行三列是同一个提示词、只换随机种子（seed 11 / 22 / 33）。",
       fill="#444", font=font(15))
d.text((pad, 70), "全部未挂 LoRA，就是原版 SDXL 的直接输出。", fill="#444", font=font(15))
d.text((pad, 92), "判读方法：某一行看起来已经很像该风格，这个数据集就没有 LoRA 的实验价值。",
       fill="#a33", font=font(15, True))

for i, tag in enumerate(ORDER):
    y = 110 + pad + i * (cell + cap + pad)
    d.text((pad, y), "%s    %s" % (tag, NOTE[tag]), fill="#222", font=font(18, True))
    for j, s in enumerate(SEEDS):
        p = os.path.join(SRC, "%s_s%d.jpg" % (tag, s))
        x = pad + j * (cell + pad)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell))
            ox, oy = x + (cell - im.width) // 2, y + 30 + (cell - im.height) // 2
            sh.paste(im, (ox, oy))
        else:
            d.rectangle([x, y + 30, x + cell, y + 30 + cell], fill="#ddd")
            d.text((x + 10, y + cell // 2), "缺失", fill="#999", font=font(16))
        d.rectangle([x, y + 30, x + cell, y + 30 + cell], outline="#b8b0a0", width=1)
        d.text((x, y + cell + 38), "seed %d" % s, fill="#555", font=font(13))
    d.text((pad, y + cell + 38), "↑ 同一提示词，三个随机种子", fill="#888", font=font(12))

sh.save(os.path.join(OUT, "probe_sheet.jpg"), quality=92)
print("已生成", os.path.join(OUT, "probe_sheet.jpg"), sh.size)
PY

cp /mnt/c/Users/ke/probe_sheet/probe_sheet.jpg /mnt/c/Users/ke/probe_sheet_final.jpg
ls -la /mnt/c/Users/ke/probe_sheet_final.jpg
