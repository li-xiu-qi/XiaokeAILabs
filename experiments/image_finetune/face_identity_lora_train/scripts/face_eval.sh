#!/usr/bin/env bash
# 人脸身份保真度评估：微调后底模+LoRA 能生成这个特定的人吗？
# 与基线严格同口径：同提示词（仅多 trigger）、同 seed、同生成参数
set -u
PY=/home/ke/jevomni-env/bin/python
EX=/home/ke/XiaokeAILabs/experiments/image_finetune/kuniyoshi_style_lora_train
LORA=${1:-$EX/output_face/face_identity_sdxl_lora/face_identity_sdxl_lora.safetensors}
OUT=${2:-/home/ke/face_eval}
NEG_ID=15
N_GEN=20
mkdir -p "$OUT"

cat > /tmp/face_eval.py <<'PY'
import os, sys, json, glob
import numpy as np, cv2, torch
from types import SimpleNamespace
from insightface.model_zoo import ArcFaceONNX, RetinaFace
from diffusers import StableDiffusionXLPipeline

MDIR = "/home/ke/.insightface/models/buffalo_l"
BASE = "/home/ke/models/SDXL-base-vkqzm"
LORA, OUT = sys.argv[1], sys.argv[2]
N_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs(OUT, exist_ok=True)

det = RetinaFace(model_file=os.path.join(MDIR, "det_10g.onnx"))
rec = ArcFaceONNX(os.path.join(MDIR, "w600k_r50.onnx"))
det.prepare(ctx_id=0, input_size=(640, 640))

def embed(img_bgr):
    bs, kpss = det.detect(img_bgr, input_size=(640, 640))
    if not len(bs):
        return None
    i = int(np.argmax(bs[:, 4]))
    return rec.get(img_bgr, SimpleNamespace(kps=kpss[i]))

def unit(v):
    return v / np.linalg.norm(v)

def collect(dirpath):
    es = []
    for f in sorted(glob.glob(dirpath + "/*.jpg")):
        img = cv2.imread(f)
        e = embed(img)
        if e is not None:
            es.append(unit(e))
    return np.array(es)

P = "/home/ke/celeb_person_14"
te_e = collect(P + "/test")
ng_e = collect("/home/ke/celeb_person_%d/test" % 15)
proto_te = unit(te_e.mean(axis=0))
print("本人测试集 %d 张、他人测试集 %d 张" % (len(te_e), len(ng_e)), flush=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, use_safetensors=True, variant=None).to("cuda")
pipe.load_lora_weights(LORA)
pipe.set_progress_bar_config(disable=True)

PROMPT = "vkqzm, a photo of a person, front facing, sharp focus"
print("提示词:", PROMPT, flush=True)

sims_pos, sims_neg = [], []
for i in range(N_GEN):
    g = torch.Generator("cuda").manual_seed(1000 + i)   # 与基线同 seed
    im = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.0,
              height=768, width=768, generator=g).images[0]
    p = os.path.join(OUT, "lora_s%02d.jpg" % i)
    im.save(p, quality=92)
    e = embed(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    if e is not None:
        e = unit(e)
        sims_pos.append(float(e @ proto_te))
        sims_neg.append(float(e @ unit(ng_e.mean(axis=0))))

res = {
  "lora": LORA, "prompt": PROMPT, "n_gen": N_GEN,
  "n_scored": len(sims_pos),
  "vs_target": {"mean": float(np.mean(sims_pos)), "std": float(np.std(sims_pos)),
                "min": float(np.min(sims_pos)), "max": float(np.max(sims_pos))},
  "vs_other": {"mean": float(np.mean(sims_neg)), "std": float(np.std(sims_neg))} if sims_neg else None,
  "per_image": sims_pos,
}
json.dump(res, open(os.path.join(OUT, "eval.json"), "w"), indent=1)

base = json.load(open("/home/ke/face_baseline/baseline.json"))
print("\n=== 对比 ===")
print("正对照（同一个人）              : %.4f" % base["reference"]["pos_cross_mean"])
print("底模基线 vs 本人                : %.4f" % base["baseline_vs_target"]["mean"])
print("微调后   vs 本人                : %.4f  (n=%d)"
      % (res["vs_target"]["mean"], len(sims_pos)))
if res["vs_other"]:
    print("微调后   vs 他人（负对照）      : %.4f" % res["vs_other"]["mean"])
print("底模基线 vs 他人                : %.4f" % base["baseline_vs_other"]["mean"])
print("\nEVAL_DONE")
PY

"$PY" /tmp/face_eval.py "$LORA" "$OUT" "$N_GEN"
