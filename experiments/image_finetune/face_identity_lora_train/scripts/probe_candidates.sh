#!/usr/bin/env bash
# 候选风格摸底：两件事一起验
#   1) Commons 上数据量与尺寸分布（数据可得性）
#   2) SDXL 底模能不能生成它（底模失败区探测）
set -u
PY=/home/ke/jevomni-env/bin/python
BASE=/home/ke/models/SDXL-base-vkqzm
ROOT=/home/ke/XiaokeAILabs/experiments/image_finetune/face_identity_lora_train
OUT=/home/ke/probe_candidates
mkdir -p "$OUT"

# ---------- 第一半：数据摸底 ----------
cat > /tmp/probe_commons.py <<'PY'
import json, urllib.request, urllib.parse, collections

CANDIDATES = {
    "thangka":      ["Category:Thangkas", "Category:Tibetan Buddhist paintings"],
    "nianhua":      ["Category:Chinese New Year prints", "Category:Yangliuqing", "Category:Taohuawu woodblock prints"],
    "persian_mini": ["Category:Persian miniatures", "Category:Persian paintings"],
    "african_mask": ["Category:African masks", "Category:Masks of Africa"],
    "kuniyoshi":    ["Category:Utagawa Kuniyoshi"],
    "bi_hua":       ["Category:Chinese gongbi paintings", "Category:Chinese bird-and-flower paintings"],
    "madang":       ["Category:Russian lubok prints", "Category:Lubok"],
}

def api(params):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "style-lora-probe/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=40))

print("%-16s %-42s %6s %6s %8s" % ("风格", "分类", "条目", "文件", ">1400px"))
print("-" * 84)
report = {}
for name, cats in CANDIDATES.items():
    tot_e = tot_f = tot_big = 0
    for cat in cats:
        try:
            r = api({"action": "query", "list": "categorymembers", "cmtitle": cat,
                     "cmtype": "file", "cmlimit": "500", "format": "json"})
            files = [m["title"] for m in r.get("query", {}).get("categorymembers", [])]
        except Exception as e:
            print("  ERR", cat, e)
            continue
        tot_e += len(files)
        big = 0
        for i in range(0, min(len(files), 24), 12):   # 抽样看尺寸，别把 API 打爆
            chunk = files[i:i + 12]
            try:
                r2 = api({"action": "query", "titles": "|".join(chunk),
                          "prop": "imageinfo", "iiprop": "size", "format": "json"})
                for p in r2.get("query", {}).get("pages", {}).values():
                    ii = p.get("imageinfo", [{}])[0]
                    if ii.get("width", 0) >= 1400 and ii.get("height", 0) >= 1400:
                        big += 1
            except Exception:
                pass
        tot_big += big
        print("%-16s %-42s %6d %6s" % (name, cat.replace("Category:", ""), len(files), ""))
    report[name] = {"entries": tot_e, "big_sampled": tot_big}
    print("%-16s %-42s %6d %6s %8d" % ("", "合计", tot_e, "", tot_big))
    print("-" * 84)

json.dump(report, open("/tmp/commons_report.json", "w"), ensure_ascii=False, indent=1)
print("\n已写入 /tmp/commons_report.json")
PY

echo "########## 第一半：Commons 数据摸底 ##########"
"$PY" /tmp/probe_commons.py

# ---------- 第二半：底模探针 ----------
cat > /tmp/probe_base.py <<'PY'
import os, sys, torch
from diffusers import StableDiffusionXLPipeline

BASE = "/home/ke/models/SDXL-base-vkqzm"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

PROBES = [
    ("thangka",      "tibetan thangka painting, a deity with multiple arms, mineral pigments on cotton, gold leaf borders, himalayan buddhist art"),
    ("nianhua",      "chinese new year woodblock print, yangliuqing folk art, a chubby child holding a fish, saturated red and gold, flat perspective"),
    ("persian_mini", "persian miniature painting, safavid court scene, flat perspective, intricate gold borders, jewel tones on paper"),
    ("african_mask", "african tribal ritual mask, carved dark wood, geometric patterns, museum artifact photograph"),
    ("kuniyoshi",    "ukiyo-e woodblock print by Utagawa Kuniyoshi, a samurai standing in snow, bold outlines, flat color"),
    ("gongbi",       "chinese gongbi fine-brush painting, birds and flowers on silk, meticulous linework, muted mineral colors"),
]

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, use_safetensors=True, variant=None)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

for tag, prompt in PROBES:
    for seed in (11, 22, 33):
        g = torch.Generator("cuda").manual_seed(seed)
        im = pipe(prompt, num_inference_steps=30, guidance_scale=7.0,
                  height=1024, width=1024, generator=g).images[0]
        p = os.path.join(OUT, "%s_s%d.jpg" % (tag, seed))
        im.save(p, quality=90)
        print("saved", p, flush=True)
print("PROBE_DONE")
PY

echo
echo "########## 第二半：SDXL 底模探针 ##########"
"$PY" /tmp/probe_base.py "$OUT"

# 打包
cd "$OUT" && tar czf /mnt/c/Users/ke/probe_candidates.tgz . && ls -la /mnt/c/Users/ke/probe_candidates.tgz
echo "ALL_DONE"
