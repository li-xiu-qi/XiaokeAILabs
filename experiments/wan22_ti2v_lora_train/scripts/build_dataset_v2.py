"""汇编 v2 数据集：12 段旧 clip + 6 段新 clip，caption 全部重写为极简格式。

第一轮 caption："rzqduck, a yellow rubber duck toy with a tiny red bowtie, <motion>"
第二轮 caption："rzqduck, <motion>"

去掉显式外观描述，强迫模型把「黄鸭 + 红领结」这个概念绑定到生僻 token 本身。
同时修正场景偏斜：第一轮 12 段里 7 段水景，第二轮 18 段里水景非水景各半。

用法：python build_dataset_v2.py
"""
import os
import shutil

SRC = "data/rzqduck_clips"
DST = "data/rzqduck_clips_v2"
OLD_PREFIX = "rzqduck, a yellow rubber duck toy with a tiny red bowtie, "

os.makedirs(DST, exist_ok=True)

# 1. 搬 12 段旧 clip 进来
n_old = 0
for f in sorted(os.listdir(SRC)):
    if f.endswith(".mp4") or f.endswith(".txt"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
        if f.endswith(".mp4"):
            n_old += 1

# 2. 重写全部 caption（旧 12 段去描述，新 6 段已是极简格式）
n_txt = 0
for f in sorted(os.listdir(DST)):
    if not f.endswith(".txt"):
        continue
    p = os.path.join(DST, f)
    with open(p, encoding="utf-8") as fh:
        cap = fh.read().strip()
    if cap.startswith(OLD_PREFIX):
        cap = "rzqduck, " + cap[len(OLD_PREFIX):]
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(cap)
    n_txt += 1
    print(f"{f}: {cap}")

print(f"\n{DST}: {n_old} clips, {n_txt} captions")
