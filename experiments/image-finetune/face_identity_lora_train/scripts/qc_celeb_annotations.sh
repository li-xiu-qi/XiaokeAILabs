#!/usr/bin/env bash
# 数据标注质量审查：不信任上游标注，用模型自己验一遍
# 四项检查：
#   1. 身份一致性   训练集每张图 vs 测试集原型的相似度分布，找离群（疑似错标）
#   2. 跨身份污染   训练集 vs 其他身份的相似度，确认没混进别人
#   3. 重复与泄漏   感知哈希查重（集内 + 跨训练/测试）
#   4. 人脸质量     检出尺寸、模糊度、侧脸角度
set -u
PY=/home/ke/jevomni-env/bin/python
OUT=/home/ke/celeb_qc
mkdir -p "$OUT"

cat > /tmp/celeb_qc.py <<'PY'
import os, sys, json, glob
import numpy as np, cv2
from types import SimpleNamespace
from insightface.model_zoo import ArcFaceONNX, RetinaFace

MDIR = "/home/ke/.insightface/models/buffalo_l"
OUT = "/home/ke/celeb_qc"
os.makedirs(OUT, exist_ok=True)

det = RetinaFace(model_file=os.path.join(MDIR, "det_10g.onnx"))
rec = ArcFaceONNX(os.path.join(MDIR, "w600k_r50.onnx"))
det.prepare(ctx_id=0, input_size=(640, 640))

def analyze(path):
    img = cv2.imread(path)
    bs, kpss = det.detect(img, input_size=(640, 640))
    if not len(bs):
        return None
    i = int(np.argmax(bs[:, 4]))
    box = bs[i]
    emb = rec.get(img, SimpleNamespace(kps=kpss[i]))
    if emb is None:
        return None
    # 侧脸角度估计：用双眼连线的倾斜角近似
    k = kpss[i]
    dx, dy = k[1][0] - k[0][0], k[1][1] - k[0][1]
    roll = abs(np.degrees(np.arctan2(dy, dx)))
    # 人脸框占画面比例
    area = (box[2] - box[0]) * (box[3] - box[1]) / (img.shape[0] * img.shape[1])
    # 模糊度：人脸区域的拉普拉斯方差
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    crop = cv2.cvtColor(img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)], cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(crop, cv2.CV_64F).var() if crop.size else 0.0
    return {"emb": emb / np.linalg.norm(emb), "score": float(box[4]),
            "face_ratio": float(area), "roll": float(roll), "sharp": float(sharp)}

def ahash(path, s=12):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (s, s))
    return "".join("1" if v > g.mean() else "0" for v in g.flatten())

def ham(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

P = "/home/ke/celeb_person_14"
NEG = "/home/ke/celeb_person_15"
groups = {}
for split in ("train", "test"):
    rows = []
    for f in sorted(glob.glob(os.path.join(P, split, "*.jpg"))):
        a = analyze(f)
        if a:
            a["file"] = os.path.basename(f); a["path"] = f
            a["hash"] = ahash(f)
            rows.append(a)
        else:
            print("  [警告] %s/%s 未检出人脸" % (split, os.path.basename(f)))
    groups[split] = rows
    print("%s: %d/%d 张检出人脸" % (split, len(rows),
          len(glob.glob(os.path.join(P, split, "*.jpg")))), flush=True)

neg = []
for f in sorted(glob.glob(os.path.join(NEG, "test", "*.jpg"))):
    a = analyze(f)
    if a:
        neg.append(a)

# 原型：测试集均值
proto = np.mean([r["emb"] for r in groups["test"]], axis=0)
proto = proto / np.linalg.norm(proto)
neg_proto = np.mean([r["emb"] for r in neg], axis=0)
neg_proto = neg_proto / np.linalg.norm(neg_proto)

print("\n=== 检查 1：训练集身份一致性（vs 测试集原型）===")
for r in groups["train"]:
    r["sim_self"] = float(r["emb"] @ proto)
    r["sim_neg"] = float(r["emb"] @ neg_proto)
sims = [r["sim_self"] for r in groups["train"]]
print("分布: mean %.4f  sd %.4f  min %.4f  max %.4f"
      % (np.mean(sims), np.std(sims), np.min(sims), np.max(sims)))
thr = np.mean(sims) - 2 * np.std(sims)
outliers = [r for r in groups["train"] if r["sim_self"] < thr]
print("离群阈值 (mean-2sd): %.4f" % thr)
if outliers:
    for r in outliers:
        print("  [可疑] %s  vs本人 %.4f  vs他人 %.4f"
              % (r["file"], r["sim_self"], r["sim_neg"]))
else:
    print("  无离群，训练集内身份一致")

print("\n=== 检查 2：跨身份污染（训练集 vs 他人原型）===")
contam = [r for r in groups["train"] if r["sim_neg"] > 0.3]
print("vs 他人原型: mean %.4f  max %.4f（>0.3 视为可疑）"
      % (np.mean([r["sim_neg"] for r in groups["train"]]),
         np.max([r["sim_neg"] for r in groups["train"]])))
for r in contam:
    print("  [可疑] %s  vs他人 %.4f" % (r["file"], r["sim_neg"]))
if not contam:
    print("  未发现跨身份污染")

print("\n=== 检查 3：重复与泄漏（感知哈希，汉明<=4）===")
allrows = [(r, "train") for r in groups["train"]] + [(r, "test") for r in groups["test"]]
dups = []
for i in range(len(allrows)):
    for j in range(i + 1, len(allrows)):
        d = ham(allrows[i][0]["hash"], allrows[j][0]["hash"])
        if d <= 4:
            dups.append((allrows[i][0]["file"], allrows[j][0]["file"], d,
                         allrows[i][1] + "/" + allrows[j][1]))
if dups:
    for a, b, d, tag in dups:
        print("  [重复] %s <-> %s  d=%d  (%s)" % (a, b, d, tag))
else:
    print("  无重复")

print("\n=== 检查 4：人脸质量 ===")
print("%-14s %6s %7s %7s %8s" % ("文件", "置信", "占画面", "侧滚角", "清晰度"))
for r in sorted(groups["train"] + groups["test"], key=lambda x: x["sharp"]):
    flag = ""
    if r["face_ratio"] < 0.05: flag += " 脸偏小"
    if r["roll"] > 25: flag += " 侧脸"
    if r["sharp"] < 50: flag += " 模糊"
    print("%-14s %6.2f %7.3f %7.1f %8.1f%s"
          % (r["file"], r["score"], r["face_ratio"], r["roll"], r["sharp"], flag))

json.dump({"train": [{k: v for k, v in r.items() if k not in ("emb", "hash")}
                     for r in groups["train"]],
           "test": [{k: v for k, v in r.items() if k not in ("emb", "hash")}
                    for r in groups["test"]],
           "dup_pairs": dups,
           "outliers": [r["file"] for r in outliers],
           "contaminated": [r["file"] for r in contam]},
          open(os.path.join(OUT, "qc.json"), "w"), ensure_ascii=False, indent=1)
print("\n报告 -> qc.json")
print("QC_DONE")
PY

"$PY" /tmp/celeb_qc.py
