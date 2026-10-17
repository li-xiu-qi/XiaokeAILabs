#!/usr/bin/env bash
# 一张图讲清微调效果：同一个人、同提示词、同 seed，唯一差别是有没有 LoRA
set -u
PY=/home/ke/jevomni-env/bin/python
OUT=/mnt/c/Users/ke/face_compare
mkdir -p "$OUT"

"$PY" - <<'PY'
import os, glob
from PIL import Image, ImageDraw, ImageFont

OUT = "/mnt/c/Users/ke/face_compare"
os.makedirs(OUT, exist_ok=True)

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

# 四列：目标真图 / 底模 / 微调1000步 / 微调2000步
PANELS = [
    ("目标：这个人的真实照片", "/home/ke/celeb_person_14/test/te_002.jpg",
     "ArcFace 相似度基准", "#4a7c59", None),
    ("微调前：底模生成", "/home/ke/face_baseline/base_s00.jpg",
     "同提示词同 seed", "#a33", "vs 本人 −0.017"),
    ("微调后：1000 步（最佳）", "/home/ke/face_ckpt_scan/step_000001000/s00.jpg",
     "只多了 LoRA + trigger", "#2c6fbb", "vs 本人 0.3232"),
    ("微调后：2000 步（过训）", "/home/ke/face_ckpt_scan/step_lora/s00.jpg",
     "训过头的反例", "#888", "vs 本人 0.2461"),
]

CW, CH, pad, cap = 420, 500, 20, 108
W = len(PANELS) * CW + (len(PANELS) + 1) * pad
H = CH + cap + 150
sh = Image.new("RGB", (W, H), "#f7f5f0")
d = ImageDraw.Draw(sh)

d.text((pad, 22), "微调效果：一个底模画不出来的人", fill="#1a1a1a", font=font(30, True))
d.text((pad, 66), "数据：CelebA 单人 20 张训练图。四列用同一个提示词、同一个随机种子，"
                  "唯一差别是有没有加载 LoRA。", fill="#444", font=font(16))
d.text((pad, 92), "指标：ArcFace 人脸 embedding 余弦相似度（与本人测试集比对）。",
       fill="#444", font=font(16))

for i, (title, path, sub, col, metric) in enumerate(PANELS):
    x = pad + i * (CW + pad)
    y = 130
    d.text((x, y), title, fill=col, font=font(19, True))
    d.text((x, y + 30), sub, fill="#777", font=font(14))

    if os.path.exists(path):
        im = Image.open(path).convert("RGB")
        im.thumbnail((CW - 16, CH - 16))
        ox = x + (CW - im.width) // 2
        oy = y + 56 + (CH - 16 - im.height) // 2
        sh.paste(im, (ox, oy))
    else:
        d.rectangle([x, y + 56, x + CW, y + 56 + CH - 16], fill="#e5e0d5")
        d.text((x + 20, y + 56 + CH // 2), "图缺失", fill="#999", font=font(18))
    d.rectangle([x, y + 56, x + CW, y + 56 + CH - 16], outline=col, width=4)

    if metric:
        d.text((x, y + 56 + CH + 6), metric, fill=col, font=font(20, True))
    else:
        d.text((x, y + 56 + CH + 6), "0.7159（同一人的水平）", fill=col, font=font(20, True))

# 底部结论
y2 = 130 + 56 + CH + 50
d.text((pad, y2), "读法：", fill="#1a1a1a", font=font(19, True))
d.text((pad + 70, y2), "底模给的是随机人脸（相似度 −0.017，与给陌生人 0.025 无差别）；"
                       "加载 LoRA 后同一句提示词生出这个人（0.3232）。", fill="#333", font=font(17))
d.text((pad, y2 + 34), "第三列与第四列的差别说明：微调有效，但训过头会引入噪点伪影，"
                       "1000 步是这组数据的最优点。", fill="#333", font=font(17))

out = os.path.join(OUT, "face_compare.jpg")
sh.save(out, quality=93)
print("生成", out, sh.size)
PY
ls -la "$OUT"
