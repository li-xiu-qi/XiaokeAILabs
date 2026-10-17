#!/usr/bin/env bash
# 导出指定身份的 CelebA 图，划分训练/测试，并检查人脸相似度评估工具
set -u
PY=/home/ke/jevomni-env/bin/python
SRC=/home/ke/celeba_shard0.parquet
CID=${1:-14}
OUT=/home/ke/celeb_person_${CID}
mkdir -p "$OUT"

"$PY" - "$SRC" "$CID" "$OUT" <<'PY'
import json, os, sys
import pyarrow.parquet as pq
from PIL import Image
import io

SRC, CID, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t = pq.read_table(SRC)
ids = t.column("celeb_id").to_pylist()
idx = [i for i, v in enumerate(ids) if v == CID]
print("celeb_id=%d 命中 %d 张" % (CID, len(idx)))

imgs = t.column("image").to_pylist()

def to_bytes(v):
    # HF datasets 的 image 列在不同导出里可能是 bytes 或 {"bytes":..,"path":..}
    if isinstance(v, dict):
        return v.get("bytes")
    return v

picked = [to_bytes(imgs[i]) for i in idx]

os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
os.makedirs(os.path.join(OUT, "test"), exist_ok=True)

# 交替切分，保证训练/测试的光线与角度都有覆盖（不是按顺序切）
train, test = [], []
for i, b in enumerate(picked):
    (test if i % 3 == 2 else train).append((i, b))

manifest = []
for split, items in (("train", train), ("test", test)):
    for i, b in items:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        name = "%s_%03d.jpg" % (split[:2], i)
        im.save(os.path.join(OUT, split, name), quality=95)
        manifest.append({"file": name, "split": split, "src_idx": i,
                         "size": list(im.size)})
        print("  %s/%s %dx%d" % (split, name, im.width, im.height))

json.dump({"celeb_id": CID, "n_train": len(train), "n_test": len(test),
           "items": manifest},
          open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("\n训练 %d 张 / 测试 %d 张 -> manifest.json" % (len(train), len(test)))
PY

echo
echo "=== 人脸相似度评估工具 ==="
"$PY" -c "
try:
    import insightface
    print('insightface 可用', insightface.__version__)
except Exception as e:
    print('insightface 不可用:', str(e)[:80])
try:
    import torchreid
    print('torchreid 可用')
except Exception as e:
    print('torchreid 不可用')
try:
    import facenet_pytorch
    print('facenet_pytorch 可用')
except Exception as e:
    print('facenet_pytorch 不可用')
"
echo "EXPORT_DONE"
