"""Wan2.2-TI2V-5B LoRA 前后对比出片：同 prompt 同 seed，base 与 base+lora 各出一段。

用法（DGX Spark，jevomni-env）：
    python eval_wan_lora.py --lora output/rzqduck_wan22_ti2v_lora_v1/rzqduck_wan22_ti2v_lora_v1.safetensors \
        --out output/eval

产物：
- p{i}_before.mp4 / p{i}_after.mp4   完整片段
- p{i}_before_frames.png / p{i}_after_frames.png   等间隔 6 帧拼板（供报告嵌图）
"""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import av
import numpy as np
import torch
from diffusers import WanPipeline
from PIL import Image

MODEL = "/home/ke/models/Wan2.2-TI2V-5B-Diffusers"
PROMPTS = [
    "a photo of rzqduck sitting on a sunny windowsill next to a potted plant",
    "rzqduck floating on a moonlit pond surrounded by fireflies",
    "rzqduck riding a paper boat down a rainy street stream",
    "rzqduck on a kitchen counter with rising steam from a pot",
]
NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
SEED = 42
STEPS = 20
GUIDANCE = 5.0
FRAMES = 49
SIZE = 512
FPS = 24


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


def save_frame_grid(path, frames, cols=6):
    idx = np.linspace(0, len(frames) - 1, cols).astype(int)
    picked = [Image.fromarray(frames[j]) for j in idx]
    w, h = picked[0].size
    grid = Image.new("RGB", (w * cols, h))
    for k, im in enumerate(picked):
        grid.paste(im, (k * w, 0))
    grid.save(path)


def to_numpy(frames):
    if isinstance(frames, torch.Tensor):
        return (frames.clamp(0, 1) * 255).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    return [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--out", default="output/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = WanPipeline.from_pretrained(args.base, torch_dtype=torch.bfloat16).to("cuda")

    for i, prompt in enumerate(PROMPTS):
        gen = torch.Generator("cuda").manual_seed(SEED)
        with torch.no_grad():
            out = pipe(prompt=prompt, negative_prompt=NEG, height=SIZE, width=SIZE,
                       num_frames=FRAMES, num_inference_steps=STEPS,
                       guidance_scale=GUIDANCE, generator=gen)
        fb = to_numpy(out.frames[0])
        p1 = os.path.join(args.out, f"p{i}_before.mp4")
        write_mp4(p1, fb, FPS)
        save_frame_grid(os.path.join(args.out, f"p{i}_before_frames.png"), fb)

        pipe.load_lora_weights(args.lora)
        gen = torch.Generator("cuda").manual_seed(SEED)
        with torch.no_grad():
            out = pipe(prompt=prompt, negative_prompt=NEG, height=SIZE, width=SIZE,
                       num_frames=FRAMES, num_inference_steps=STEPS,
                       guidance_scale=GUIDANCE, generator=gen)
        fa = to_numpy(out.frames[0])
        p2 = os.path.join(args.out, f"p{i}_after.mp4")
        write_mp4(p2, fa, FPS)
        save_frame_grid(os.path.join(args.out, f"p{i}_after_frames.png"), fa)
        pipe.unload_lora_weights()

        print(f"[{i}] {prompt}\n  before: {p1}\n  after:  {p2}", flush=True)

    print("done ->", args.out)


if __name__ == "__main__":
    main()
