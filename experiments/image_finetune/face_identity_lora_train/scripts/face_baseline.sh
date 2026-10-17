#!/usr/bin/env bash
# 人脸身份保真度基线：底模用中性提示词能生成这个特定的人吗？
# 三个对照一起算，缺一不可：
#   B  底模生成图 vs 本人测试集        -> 要低
#   N  底模生成图 vs 他人测试集(负对照) -> 要比 B 高就没说服力
#   P  本人训练集 vs 本人测试集(正对照) -> "同一个人"的相似度基准线
set -u
PY=/home/ke/jevomni-env/bin/python
OUT=/home/ke/face_baseline
mkdir -p "$OUT"

cat > /tmp/face_baseline.py <<'PY'
import os, sys, json, glob
import numpy as np, cv2, torch
from types import SimpleNamespace
from insightface.model_zoo import ArcFaceONNX, RetinaFace
from insightface.utils import face_align
from diffusers import StableDiffusionXLPipeline

MDIR = "/home/ke/.insightface/models/buffalo_l"
BASE = "/home/ke/models/SDXL-base-vkqzm"
OUT = sys.argv[1]
NEG_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 15
N_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs(OUT, exist_ok=True)

det = RetinaFace(model_file=os.path.join(MDIR, "det_10g.onnx"))
rec = ArcFaceONNX(os.path.join(MDIR, "w600k_r50.onnx"))
det.prepare(ctx_id=0, input_size=(640, 640))

def embed(path_or_img):
    img = (cv2.imread(path_or_img) if isinstance(path_or_img, str)
           else cv2.cvtColor(np.array(path_or_img), cv2.COLOR_RGB2BGR))
    bs, kpss = det.detect(img, input_size=(640, 640))
    if not len(bs):
        return None, 0
    i = int(np.argmax(bs[:, 4]))          # 取置信度最高的一张脸
    return rec.get(img, SimpleNamespace(kps=kpss[i])), len(bs)

def unit(v):
    return v / np.linalg.norm(v)

# ---- 参考 embedding ----
def collect(dirpath):
    fs, es = [], []
    for f in sorted(glob.glob(dirpath + "/*.jpg")):
        e, n = embed(f)
        if e is not None:
            fs.append(os.path.basename(f)); es.append(unit(e))
    return fs, np.array(es)

P = "/home/ke/celeb_person_14"
print("=== 参考集 ===", flush=True)
tr_f, tr_e = collect(P + "/train")
te_f, te_e = collect(P + "/test")
print("本人训练集 %d 张、测试集 %d 张（检出人脸）" % (len(tr_f), len(te_f)), flush=True)

neg_dir = "/home/ke/celeb_person_%d/test" % NEG_ID
if os.path.isdir(neg_dir):
    ng_f, ng_e = collect(neg_dir)
    print("负对照（celeb_id=%d 测试集）%d 张" % (NEG_ID, len(ng_f)), flush=True)
else:
    ng_e = np.zeros((0, 512)); print("无负对照目录", neg_dir, flush=True)

# 正对照：训练集 vs 测试集的跨集相似度分布（同一人的基准线）
proto_te = unit(te_e.mean(axis=0))
pos_sims = [float(unit(e) @ proto_te) for e in tr_e]
print("正对照（本人训练集 vs 本人测试集均值）：mean %.4f  min %.4f  max %.4f"
      % (np.mean(pos_sims), np.min(pos_sims), np.max(pos_sims)), flush=True)

# ---- 底模生成 ----
print("\n=== 底模生成 %d 张（中性提示词，不含任何身份线索）===" % N_GEN, flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, use_safetensors=True, variant=None).to("cuda")
pipe.set_progress_bar_config(disable=True)

PROMPT = "a photo of a person, front facing, sharp focus"
gen_e, gen_faces = [], 0
for i in range(N_GEN):
    g = torch.Generator("cuda").manual_seed(1000 + i)
    im = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.0,
              height=768, width=768, generator=g).images[0]
    p = os.path.join(OUT, "base_s%02d.jpg" % i)
    im.save(p, quality=92)
    e, n = embed(p)
    if e is not None:
        gen_e.append(unit(e)); gen_faces += n
print("生成 %d 张，检出人脸 %d 个" % (N_GEN, gen_faces), flush=True)

gen_e = np.array(gen_e)
sims_pos = [float(e @ proto_te) for e in gen_e]
sims_neg = ([float(e @ unit(ng_e.mean(axis=0))) for e in gen_e]
            if len(ng_e) else [])

res = {
  "prompt": PROMPT, "n_gen": N_GEN, "n_faces": gen_faces,
  "reference": {
     "n_train": len(tr_f), "n_test": len(te_f),
     "pos_cross_mean": float(np.mean(pos_sims)),
     "pos_cross_min": float(np.min(pos_sims)),
     "pos_cross_max": float(np.max(pos_sims))},
  "baseline_vs_target": {
     "mean": float(np.mean(sims_pos)), "std": float(np.std(sims_pos)),
     "min": float(np.min(sims_pos)), "max": float(np.max(sims_pos)),
     "n": len(sims_pos)},
  "baseline_vs_other": {
     "mean": float(np.mean(sims_neg)) if sims_neg else None,
     "std": float(np.std(sims_neg)) if sims_neg else None,
     "n": len(sims_neg)},
  "per_image": sims_pos,
}
json.dump(res, open(os.path.join(OUT, "baseline.json"), "w"), indent=1)

print("\n=== 结果 ===")
print("正对照（同一个人，训练集 vs 测试集）: mean %.4f" % res["reference"]["pos_cross_mean"])
print("底模生成 vs 本人测试集            : mean %.4f  (n=%d)"
      % (res["baseline_vs_target"]["mean"], len(sims_pos)))
if sims_neg:
    print("底模生成 vs 他人测试集（负对照）    : mean %.4f" % res["baseline_vs_other"]["mean"])
print("\n判据：若底模 vs 本人 明显低于正对照、且与负对照接近，则底模画不出这个人，LoRA 有靶子。")
print("BASELINE_DONE")
PY

# 先备负对照数据
NEG_ID=15
N_GEN=20
bash /mnt/c/Users/ke/export_celeb_person.sh "$NEG_ID" > /dev/null 2>&1 || true

"$PY" /tmp/face_baseline.py "$OUT" "$NEG_ID" "$N_GEN"
