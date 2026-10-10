"""汇编 v3 数据集：18 段 v2 clip + 4 段新水景 clip = 22 段，11 水景 + 11 陆地。

caption 沿用 v2 的极简格式（rzqduck, <motion>），无需重写。

用法：python build_dataset_v3.py
"""
import os
import shutil

SRC_V2 = "data/rzqduck_clips_v2"
SRC_V3 = "data/rzqduck_clips_v3"
DST = "data/rzqduck_clips_v3_full"

os.makedirs(DST, exist_ok=True)

n = 0
for src in (SRC_V2, SRC_V3):
    for f in sorted(os.listdir(src)):
        if f.endswith(".mp4") or f.endswith(".txt"):
            shutil.copy2(os.path.join(src, f), os.path.join(DST, f))
            if f.endswith(".mp4"):
                n += 1

water_kw = ["water", "bathtub", "wave", "lake", "pond", "puddle", "pool",
            "stream", "fishbowl", "basin", "gutter"]
caps = {}
for f in sorted(os.listdir(DST)):
    if f.endswith(".txt"):
        caps[f] = open(os.path.join(DST, f), encoding="utf-8").read().strip()

n_water = sum(1 for c in caps.values() if any(k in c for k in water_kw))
print(f"{DST}: {n} clips, {len(caps)} captions")
print(f"水景 {n_water} 段，陆地 {len(caps) - n_water} 段")
for f in sorted(caps):
    print(f"  {f}: {caps[f]}")
