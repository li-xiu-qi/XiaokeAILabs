#!/usr/bin/env bash
# 检查点扫描：评 500/1000/1500/2000 四个步数的人脸相似度，找最优点
set -u
PY=/home/ke/jevomni-env/bin/python
EX=/home/ke/XiaokeAILabs/experiments/image-finetune/face_identity_lora_train
CKPT_DIR=$EX/output_face_v2/face_identity_sdxl_lora_v2
OUTROOT=/home/ke/face_ckpt_scan_v2
N_GEN=10
mkdir -p "$OUTROOT"

cat > /tmp/face_ckpt_scan_v2.py <<'PY'
import os, sys, json, glob
import numpy as np, cv2, torch
from types import SimpleNamespace
from insightface.model_zoo import ArcFaceONNX, RetinaFace
from diffusers import StableDiffusionXLPipeline

MDIR = "/home/ke/.insightface/models/buffalo_l"
BASE = "/home/ke/models/SDXL-base-vkqzm"
CKPT_DIR, OUTROOT = sys.argv[1], sys.argv[2]
N_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 10
os.makedirs(OUTROOT, exist_ok=True)

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
        e = embed(cv2.imread(f))
        if e is not None:
            es.append(unit(e))
    return np.array(es)

te_e = collect("/home/ke/celeb_person_14/test")
ng_e = collect("/home/ke/celeb_person_15/test")
proto_te = unit(te_e.mean(axis=0))
proto_ng = unit(ng_e.mean(axis=0))
print("参考集：本人 %d 张、他人 %d 张" % (len(te_e), len(ng_e)), flush=True)

PROMPT = "vkqzm, a photo of a person, front facing, sharp focus"
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.safetensors")))
print("待评检查点 %d 个" % len(ckpts), flush=True)

rows = []
for ck in ckpts:
    step = os.path.basename(ck).replace(".safetensors", "").split("_")[-1]
    out = os.path.join(OUTROOT, "step_%s" % step)
    os.makedirs(out, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, use_safetensors=True, variant=None).to("cuda")
    pipe.load_lora_weights(ck)
    pipe.set_progress_bar_config(disable=True)
    sp, sn = [], []
    for i in range(N_GEN):
        g = torch.Generator("cuda").manual_seed(1000 + i)
        im = pipe(PROMPT, num_inference_steps=30, guidance_scale=7.0,
                  height=512, width=512, generator=g).images[0]
        im.save(os.path.join(out, "s%02d.jpg" % i), quality=92)
        e = embed(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        if e is not None:
            e = unit(e)
            sp.append(float(e @ proto_te)); sn.append(float(e @ proto_ng))
    del pipe
    torch.cuda.empty_cache()
    r = {"step": step, "n": len(sp),
         "vs_target_mean": float(np.mean(sp)), "vs_target_std": float(np.std(sp)),
         "vs_other_mean": float(np.mean(sn)) if sn else None,
         "per_image": sp}
    rows.append(r)
    json.dump(rows, open(os.path.join(OUTROOT, "scan.json"), "w"), indent=1)
    print("步 %-8s vs 本人 %.4f (sd %.4f)  vs 他人 %.4f  n=%d"
          % (step, r["vs_target_mean"], r["vs_target_std"],
             r["vs_other_mean"] or 0, r["n"]), flush=True)

print("\n=== 汇总 ===")
print("%-10s %10s %10s %10s" % ("步数", "vs本人", "vs他人", "净差"))
for r in sorted(rows, key=lambda x: -x["vs_target_mean"]):
    print("%-10s %10.4f %10.4f %10.4f"
          % (r["step"], r["vs_target_mean"], r["vs_other_mean"] or 0,
             r["vs_target_mean"] - (r["vs_other_mean"] or 0)))
print("\nSCAN_DONE")
PY

"$PY" /tmp/face_ckpt_scan_v2.py "$CKPT_DIR" "$OUTROOT" "$N_GEN"
