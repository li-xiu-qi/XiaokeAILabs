"""Wan2.2-TI2V-5B 训练数据集生成：用底模自身生成 12 段 rzqduck 短视频。

用法（DGX Spark，jevomni-env）：
    python gen_dataset.py --out data/rzqduck_clips

设计说明：
- 每段 clip 都以同一触发词描述开头（rzqduck + 小红领结），保证概念一致性
- 运动与场景描述逐段不同，给 LoRA 提供运动多样性
- 全部 clip 由底模直接生成（合成数据），本实验验证的是训练链路与触发词绑定，
  不代表真实视频数据上的质量
- mp4 用 PyAV 编码（h264, yuv420p, 24fps），不依赖 imageio
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
TRIGGER = "rzqduck, a yellow rubber duck toy with a tiny red bowtie"
NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# 12 段：同一主体，不同运动与场景
MOTIONS = [
    "bobbing up and down on gentle water ripples, close-up",
    "spinning slowly on a wooden table, studio lighting",
    "waddling across a sunlit tiled floor",
    "floating in a bathtub surrounded by white bubbles",
    "rocking side to side on a small wave at the beach",
    "being gently squeezed and bouncing back, slow motion",
    "drifting on a calm lake at sunset, reflections on water",
    "bouncing on a metal spring, up and down rhythmically",
    "rotating on a turntable, soft studio light",
    "floating among green lotus leaves in a pond",
    "bobbing in a rain puddle, raindrops hitting the surface",
    "sliding down a smooth yellow plastic slide into a pool",
]

WIDTH = 512
HEIGHT = 512
NUM_FRAMES = 81
STEPS = 30
GUIDANCE = 5.0
FPS = 24
SEED = 42


def write_mp4(path, frames, fps):
    """frames: list of numpy uint8 (H, W, 3)，用 PyAV 编码 h264。"""
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
    ap.add_argument("--out", default="data/rzqduck_clips")
    ap.add_argument("--limit", type=int, default=0, help="只生成前 N 段（调试用）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print(f"pipeline loaded: {MODEL}", flush=True)

    n = args.limit or len(MOTIONS)
    for i, motion in enumerate(MOTIONS[:n]):
        prompt = f"{TRIGGER}, {motion}"
        t0 = time.time()
        gen = torch.Generator("cuda").manual_seed(SEED + i)
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                negative_prompt=NEG,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            )
        frames = out.frames[0]
        if isinstance(frames, torch.Tensor):
            frames = (frames.clamp(0, 1) * 255).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
        else:
            frames = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]

        stem = os.path.join(args.out, f"clip_{i + 1:02d}")
        write_mp4(stem + ".mp4", frames, FPS)
        with open(stem + ".txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        dt = time.time() - t0
        free, total = torch.cuda.mem_get_info()
        print(f"[{i + 1:02d}/{n}] {dt:.0f}s | {len(frames)} frames | "
              f"vram free {free / 2**30:.0f}G/{total / 2**30:.0f}G | {prompt}", flush=True)

    print(f"done -> {args.out} ({n} clips)")


if __name__ == "__main__":
    main()
