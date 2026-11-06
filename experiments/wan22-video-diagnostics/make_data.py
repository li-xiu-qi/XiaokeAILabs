"""Make an original illustrated character dataset; no external photos or video.

Uses Pillow and ffmpeg (libx264). The clips are simple 2D animations, not people.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

SIZE = 512
FRAMES = 49
FPS = 24
TRIGGER = 'mivora'
BACKGROUNDS = ['#e9cfaa', '#c4d9d2', '#ddd1e5', '#f0d6d0']
ACTIONS = ['waving one arm', 'moving gently from side to side',
           'standing with both arms raised', 'bouncing gently up and down']


def draw_frame(frame, scene, action):
    image = Image.new('RGB', (SIZE, SIZE), BACKGROUNDS[scene % len(BACKGROUNDS)])
    d = ImageDraw.Draw(image)
    wave = math.sin(frame * 2 * math.pi / (FRAMES - 1))
    x = 256 + (int(20 * wave) if action == 1 else 0)
    y = 15 + (int(8 * wave) if action == 3 else 0)
    d.rectangle((0, 410, 512, 512), fill='#a3bca3')
    d.rectangle((x-44, y+315, x-10, y+422), fill='#263c53')
    d.rectangle((x+10, y+315, x+44, y+422), fill='#263c53')
    d.rounded_rectangle((x-70, y+205, x+70, y+335), radius=28, fill='#d84c3e')
    d.ellipse((x-63, y+62, x+63, y+198), fill='#dca77e')
    d.pieslice((x-68, y+48, x+68, y+169), 180, 360, fill='#24343c')
    d.ellipse((x-26, y+126, x-16, y+137), fill='#202d31')
    d.ellipse((x+16, y+126, x+26, y+137), fill='#202d31')
    d.arc((x-18, y+140, x+18, y+162), 0, 180, fill='#713f36', width=3)
    d.polygon([(x-46,y+212),(x,y+244),(x+46,y+212)], fill='#f3d36d')
    left = y+195 if action == 2 else y+285
    right = y+195 if action == 2 else y+285
    if action == 0:
        right = y+220-int(35*wave)
    d.line((x-62,y+232,x-110,left), fill='#dca77e', width=19)
    d.line((x+62,y+232,x+110,right), fill='#dca77e', width=19)
    d.ellipse((x-119,left-9,x-101,left+9), fill='#dca77e')
    d.ellipse((x+101,right-9,x+119,right+9), fill='#dca77e')
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for scene in range(4):
        for action in range(4):
            index = scene * 4 + action
            path = output / f'clip_{index:02d}.mp4'
            caption = f'{TRIGGER}, a 2D illustrated person with a red shirt and yellow scarf, {ACTIONS[action]}.'
            command = [args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE}x{SIZE}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for frame in range(FRAMES):
                    process.stdin.write(draw_frame(frame, scene, action).tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f'ffmpeg failed: {path}')
            except BaseException:
                process.kill()
                process.wait()
                raise
            path.with_suffix('.txt').write_text(caption, encoding='utf-8')
            manifest.append({'video': path.name, 'caption': caption, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'frames': FRAMES, 'fps': FPS, 'source': 'programmatically drawn fictional character'})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(manifest)} fictional-character clips to {output}')


if __name__ == '__main__':
    main()
