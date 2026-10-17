#!/usr/bin/env bash
# 把 ArcFace 模型装到 insightface 的模型目录，并验证能跑
set -u
PY=/home/ke/jevomni-env/bin/python
MDIR=/home/ke/.insightface/models/buffalo_l
mkdir -p "$MDIR"
cp /mnt/c/Users/ke/det_10g.onnx /mnt/c/Users/ke/w600k_r50.onnx "$MDIR"/ 2>/dev/null || {
  echo "模型文件不在 /mnt/c/Users/ke/，先上传"; exit 2;
}
ls -la "$MDIR"

"$PY" - <<'PY'
import numpy as np, cv2, glob, os
from insightface.model_zoo import ArcFaceONNX, RetinaFace

mdir = "/home/ke/.insightface/models/buffalo_l"
det = RetinaFace(model_file=os.path.join(mdir, "det_10g.onnx"))
rec = ArcFaceONNX(os.path.join(mdir, "w600k_r50.onnx"))
det.prepare(ctx_id=0, input_size=(640, 640))
print("模型加载成功")

# 自检：拿一张 CelebA 图，抽 embedding
from insightface.utils import face_align
src = "/home/ke/celeb_person_14/train"
f = sorted(glob.glob(src + "/*.jpg"))[0]
img = cv2.imread(f)
bs, kpss = det.detect(img, input_size=(640, 640))
print("自检 %s: 检出 %d 张脸" % (os.path.basename(f), len(bs)))
if len(bs):
    from types import SimpleNamespace
    fo = SimpleNamespace(kps=kpss[0])
    feat = rec.get(img, fo)
    print("  embedding 类型:", type(feat).__name__,
          "形状:", getattr(feat, "shape", None),
          "范数:", None if feat is None else round(float(np.linalg.norm(feat)), 4))
PY
echo "FACE_STACK_READY"
