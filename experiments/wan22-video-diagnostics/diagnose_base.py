"""Inspect an unquantized base model before attempting a LoRA comparison."""
import argparse
from pathlib import Path

import av
import numpy as np
import torch
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
from transformers import UMT5EncoderModel
from PIL import Image

PROMPTS = [
    'mivora, a 2D illustrated person standing in a garden and turning around.',
    'mivora, a 2D illustrated person walking beside a blue lake.',
]


def save_video(path, frames):
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264', rate=24)
        stream.width = stream.height = 512
        stream.pix_fmt = 'yuv420p'
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=False)
    transformer = WanTransformer3DModel.from_pretrained(args.model, subfolder='transformer', torch_dtype=torch.bfloat16)
    encoder = UMT5EncoderModel.from_pretrained(args.model, subfolder='text_encoder', torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder='vae', torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model, transformer=transformer, text_encoder=encoder, vae=vae, torch_dtype=torch.bfloat16).to('cuda')
    original_decode = pipe.vae.decode
    def inspected_decode(latents, *args, **kwargs):
        print('VAE_INPUT', tuple(latents.shape), 'finite', bool(torch.isfinite(latents).all()), 'min/max', float(latents.min()), float(latents.max()), flush=True)
        result = original_decode(latents, *args, **kwargs)
        decoded = result[0] if isinstance(result, tuple) else result.sample
        print('VAE_OUTPUT', tuple(decoded.shape), 'finite', bool(torch.isfinite(decoded).all()), 'min/max', float(decoded.min()), float(decoded.max()), flush=True)
        return result
    pipe.vae.decode = inspected_decode
    for index, prompt in enumerate(PROMPTS[:1]):
        for name in ('base',):
            generator = torch.Generator('cuda').manual_seed(42)
            with torch.inference_mode():
                frames = pipe(prompt=prompt, negative_prompt='', height=512, width=512,
                              num_frames=17, num_inference_steps=20,
                              guidance_scale=5.0, generator=generator, output_type='np').frames[0]
            frames = [(np.clip(frame, 0, 1)*255).astype(np.uint8) for frame in frames]
            Image.fromarray(frames[0]).save(outdir / f'{index}_{name}_raw.png')
            save_video(outdir / f'{index}_{name}.mp4', frames)
            grid = Image.new('RGB', (512*6, 512))
            for col, i in enumerate(np.linspace(0, len(frames)-1, 6).astype(int)):
                grid.paste(Image.fromarray(frames[i]), (col*512, 0))
            grid.save(outdir / f'{index}_{name}.jpg')
            print(index, name, prompt, flush=True)


if __name__ == '__main__':
    main()
