#!/usr/bin/env bash
# 数据盘点：把训练集/验证集的实际内容摊开给人看
# 产出：每张图的标题、来源、尺寸、颜色统计 + 联系表（带标注）
set -u
D=/home/ke/XiaokeAILabs/experiments/image_finetune/face_identity_lora_train/data
OUT=/mnt/c/Users/ke/datasheet
mkdir -p "$OUT"

PYBIN=${PYBIN:-/home/ke/jevomni-env/bin/python}
"$PYBIN" - "$D" "$OUT" <<'PY'
import json, os, sys, collections
from PIL import Image, ImageDraw, ImageFont

D, OUT = sys.argv[1], sys.argv[2]

meta = json.load(open(os.path.join(D, 'meta.json'), encoding='utf-8'))
recs = meta['records']
print('=== meta.json ===')
print('artist      :', meta['artist'])
print('n_records   :', len(recs))
print('n_categories:', meta['n_category_entries'])
print('min_side    :', meta['min_side'])
print('thumb_width :', meta['thumb_width'])
cats = collections.Counter(r['category'] for r in recs)
print('--- 分类构成 ---')
for k, v in cats.most_common():
    print('%4d  %s' % (v, k))

cs = json.load(open(os.path.join(D, 'color_stats2.json'), encoding='utf-8'))
print('=== color_stats2.json ===')
print('类型:', type(cs).__name__, '条数:', len(cs))
print('样例:', json.dumps(cs[0], ensure_ascii=False)[:300] if isinstance(cs, list) else '')

for split, sub in [('训练集', 'kuniyoshi_train_pure'), ('验证集', 'kuniyoshi_val_pure')]:
    files = sorted(os.listdir(os.path.join(D, sub)))
    files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
    print('=== %s %s：%d 张 ===' % (split, sub, len(files)))
    for f in files:
        p = os.path.join(D, sub, f)
        im = Image.open(p)
        print('  %-18s %5dx%-5d' % (f, im.width, im.height))

# 标题索引
by_file = {}
for r in recs:
    by_file[r['file']] = r

# 字体：优先 Noto CJK，否则中文会变方块
def font(sz):
    import glob
    cands = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]
    cands += glob.glob("/usr/share/fonts/**/NotoSansCJK*", recursive=True)
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def sheet(files, sub, title, cell, cols, outname):
    rows = (len(files) + cols - 1) // cols
    pad, cap = 10, 46
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + cap) + (rows + 1) * pad + 44
    sh = Image.new('RGB', (W, H), '#f4f1ea')
    d = ImageDraw.Draw(sh)
    fb, fs, fc = font(20), font(12), font(11)
    d.text((pad, 12), title, fill='#222', font=fb)
    for i, f in enumerate(files):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = 44 + pad + r * (cell + cap + pad)
        im = Image.open(os.path.join(D, sub, f)).convert('RGB')
        im.thumbnail((cell, cell))
        ox = x + (cell - im.width) // 2
        oy = y + (cell - im.height) // 2
        sh.paste(im, (ox, oy))
        d.rectangle([x, y, x + cell, y + cell], outline='#b8b0a0', width=1)
        t = by_file.get(f, {}).get('title', '')
        t = t.replace('File:', '')[:26]
        d.text((x + 2, y + cell + 5), f, fill='#333', font=fs)
        d.text((x + 2, y + cell + 21), t, fill='#666', font=fc)
        sz = by_file.get(f, {}).get('orig_size')
        if sz:
            d.text((x + 2, y + cell + 35), '%dx%d' % tuple(sz), fill='#999', font=fc)
    sh.save(os.path.join(OUT, outname), quality=90)
    print('生成联系表 ->', outname, sh.size)

tr = [f for f in sorted(os.listdir(os.path.join(D, 'kuniyoshi_train_pure')))
      if f.lower().endswith(('.jpg', '.png'))]
va = [f for f in sorted(os.listdir(os.path.join(D, 'kuniyoshi_val_pure')))
      if f.lower().endswith(('.jpg', '.png'))]

for i in range(0, len(tr), 24):
    sheet(tr[i:i + 24], 'kuniyoshi_train_pure',
          '训练集 %d-%d / %d 张（纯化后）' % (i + 1, min(i + 24, len(tr)), len(tr)),
          300, 4, 'sheet_train_%02d.jpg' % (i // 24 + 1))
if va:
    sheet(va, 'kuniyoshi_val_pure', '验证集 %d 张（不参与训练，只用于评估）' % len(va),
          380, 3, 'sheet_val.jpg')
PY

echo "=== 联系表 ==="
ls -la "$OUT"
