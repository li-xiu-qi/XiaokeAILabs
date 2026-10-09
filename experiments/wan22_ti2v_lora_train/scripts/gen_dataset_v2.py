"""补充生成 6 段非水景 clip（修正第一轮的场景偏斜：12 段里 7 段是水景）。

用法（DGX Spark，jevomni-env）：
    python gen_dataset_v2.py --out data/rzqduck_clips_v2

同时把全部 18 段 caption 重写为极简格式："rzqduck, <运动/场景>"
（去掉第一轮里 "a yellow rubber duck toy with a tiny red bowtie" 这段显式描述，
强迫模型把概念绑定到生僻 token 本身，而不是描述词）
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import av
import numpy as np
import torch
from diffusers import WanPipeline

MODEL = "/home/ke/models/Wan2.2-TI2V-5B-Diffusers"
NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# 6 段全新非水景运动
MOTIONS = [
    "sitting on a wooden shelf in a toy store, soft light",
    "being gently carried in a child's hand across a playground",
    "on a checkerboard floor next to a slow rolling ball",
    "under a Christmas tree with warm fairy lights",
    "in a cardboard box surrounded by white packing peanuts",
    "on a bathroom counter next to a dripping faucet",
]

WIDTH = 512
HEIGHT = 512
NUM_FRAMES = 81
STEPS = 30
GUIDANCE = 5.0
FPS = 24
SEED = 100


def write_mp4(path, frames, fps):
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20"}
    for f in frames:
        frame = av.VideoFrame.from_ndarray(f, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rzqduck_clips_v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    for i, motion in enumerate(MOTIONS):
        prompt = f"rzqduck, {motion}"
        t0 = time.time()
        gen = torch.Generator("cuda").manual_seed(SEED + i)
        with torch.no_grad():
            out = pipe(prompt=prompt, negative_prompt=NEG, height=HEIGHT, width=WIDTH,
                       num_frames=NUM_FRAMES, num_inference_steps=STEPS,
                       guidance_scale=GUIDANCE, generator=gen)
        frames = out.frames[0]
        if isinstance(frames, torch.Tensor):
            frames = (frames.clamp(0, 1) * 255).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
        else:
            frames = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]

        stem = os.path.join(args.out, f"clip_{i + 13:02d}")
        write_mp4(stem + ".mp4", frames, FPS)
        with open(stem + ".txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        free, total = torch.cuda.mem_get_info()
        print(f"[{i + 13:02d}/18] {time.time() - t0:.0f}s | vram free {free / 2**30:.0f}G | {prompt}",
              flush=True)

    print(f"done -> {args.out} (6 new clips)")


if __name__ == "__main__":
    main()
