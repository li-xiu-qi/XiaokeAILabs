"""Generate fixed-manifest student videos with or without a LoRA adapter."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from diffusers import WanPipeline
from pair_contract import validate_entries

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Diffusers model directory or Hub ID')
parser.add_argument('--manifest', required=True)
parser.add_argument('--split', choices=('train', 'eval'), default='eval')
parser.add_argument('--output', required=True)
parser.add_argument('--lora', help='Optional LoRA safetensors path')
parser.add_argument('--size', type=int, default=480)
parser.add_argument('--frames', type=int, default=49)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--fps', type=int, default=24)
parser.add_argument('--guidance', type=float, default=5.0)
args = parser.parse_args()
entries = json.loads(Path(args.manifest).read_text(encoding='utf-8'))[args.split]
validate_entries(entries)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to('cuda')
if args.lora:
    pipe.load_lora_weights(args.lora)
    print('active adapters:', pipe.get_active_adapters(), flush=True)
hashes = {}
for entry in entries:
    start = time.monotonic()
    with torch.inference_mode():
        frames = pipe(prompt=entry['prompt'], height=args.size, width=args.size,
                      num_frames=args.frames, num_inference_steps=args.steps,
                      guidance_scale=args.guidance,
                      generator=torch.Generator('cuda').manual_seed(entry['seed'])).frames[0]
    path = out / (entry['id'] + '.mp4')
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=args.fps)
        stream.width, stream.height, stream.pix_fmt = args.size, args.size, 'yuv420p'
        for frame in frames:
            array = np.asarray(frame)
            if np.issubdtype(array.dtype, np.floating):
                array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    hashes[entry['id']] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({**entry, 'frames': len(frames), 'seconds': time.monotonic() - start,
                      'sha256': hashes[entry['id']],
                      'adapter': bool(args.lora)}), flush=True)
(out / 'generation-manifest.json').write_text(json.dumps({
    'model': args.model, 'split': args.split, 'adapter': bool(args.lora), 'entries': entries,
    'size': args.size, 'frames': args.frames, 'steps': args.steps, 'fps': args.fps,
    'guidance': args.guidance, 'video_sha256': hashes,
}, ensure_ascii=False, indent=2), encoding='utf-8')
