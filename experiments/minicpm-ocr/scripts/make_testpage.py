#!/usr/bin/env python3
"""Generate a synthetic A4-ish test page (Chinese + English + table + formula) for MiniCPM-OCR."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1654, 2339  # A4 at 200dpi
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

CN = "/c/Windows/Fonts/STSONG.TTF"
CN_B = "/c/Windows/Fonts/STZHONGS.TTF"
EN = "/c/Windows/Fonts/arial.ttf"
EN_B = "/c/Windows/Fonts/arialbd.ttf"

f_title = ImageFont.truetype(CN_B, 46)
f_h = ImageFont.truetype(CN_B, 34)
f_body = ImageFont.truetype(CN, 30)
f_en = ImageFont.truetype(EN, 30)
f_en_b = ImageFont.truetype(EN_B, 32)
f_small = ImageFont.truetype(EN, 26)

y = 150

def text(x, yy, s, font, fill=(20, 20, 20)):
    d.text((x, yy), s, font=font, fill=fill)
    return yy

# 页眉
d.line((120, 110, W - 120, 110), fill=(120, 120, 120), width=2)
text(120, 125, "内部测试文档 · 2026 年第 4 季度", f_small, (90, 90, 90))

# 标题
text(120, y, "关于人工智能文档识别技术的测试报告", f_title)
y += 110

# 中文正文
para1 = ("本文档用于验证文档解析模型在中文段落、英文段落、表格与数学公式四类"
         "典型版式上的识别质量。测试页面由程序渲染生成，内容已知，可逐项核对。")
for line in [para1[i:i+34] for i in range(0, len(para1), 34)]:
    text(120, y, line, f_body); y += 52
y += 30

# 英文段落
text(120, y, "1. Introduction", f_h); y += 70
en1 = ("This page exercises four layout element types: body text, an English "
       "paragraph, a bordered table, and a displayed formula. Recognition "
       "quality is judged against the known source strings below.")
for line in [en1[i:i+62] for i in range(0, len(en1), 62)]:
    text(120, y, line, f_en); y += 48
y += 40

# 表格
text(120, y, "2. 实验数据", f_h); y += 70
rows = [
    ["模型", "参数量", "吞吐 (页/秒)", "准确率"],
    ["MiniCPM-OCR 1.0", "1.3B", "1.20", "93.68"],
    ["基线模型 A", "3.8B", "0.81", "91.20"],
    ["基线模型 B", "7.2B", "0.45", "92.05"],
]
col_x = [120, 560, 900, 1230]
row_h = 62
table_w = W - 240
for r, row in enumerate(rows):
    top = y + r * row_h
    d.rectangle((120, top, 120 + table_w, top + row_h), outline=(60, 60, 60), width=2)
    for c in range(1, 4):
        d.line((col_x[c], top, col_x[c], top + row_h), fill=(60, 60, 60), width=2)
    for c, cell in enumerate(row):
        has_cn = any('\u4e00' <= ch <= '\u9fff' for ch in cell)
        f = (f_h if r == 0 else f_body) if has_cn else (f_en_b if r == 0 else f_en)
        text(col_x[c] + 14, top + 14, cell, f)
y += len(rows) * row_h + 50

# 公式
text(120, y, "3. 数学公式", f_h); y += 70
formula = "E = mc²   |   f(x) = ∫₀^∞ e^(−x²) dx = √π / 2"
text(120, y, formula, f_en); y += 60
formula2 = "∇ · B = 0,   ∇ × E = −∂B/∂t"
text(120, y, formula2, f_en); y += 110

# 中文结尾段落
para2 = ("综上，模型需要同时正确处理中英文混排、表格线与公式符号。若上述内容"
         "被完整识别且顺序正确，则端到端解析链路工作正常。")
for line in [para2[i:i+34] for i in range(0, len(para2), 34)]:
    text(120, y, line, f_body); y += 52

# 页脚
d.line((120, H - 130, W - 120, H - 130), fill=(120, 120, 120), width=2)
text(120, H - 115, "第 1 页，共 1 页", f_small, (90, 90, 90))

out = "testpage_cn.jpg"
img.save(out, quality=92)
print("saved", out, img.size)
