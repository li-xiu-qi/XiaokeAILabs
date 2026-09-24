#!/usr/bin/env python3
"""从测试视频派生图片与音频：

- 图片：视频首帧，media/frame.jpg
- 音频：视频前 13 秒音轨，单声道 16 kHz，media/audio.wav

需要 ffmpeg 在 PATH，或用 --ffmpeg 指定可执行文件路径。
"""
import argparse
import os
import shutil
import subprocess
import sys


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    experiment_root = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video",
        default=os.path.join(experiment_root, "assets", "the-wall-23s.mp4"),
        help="输入视频路径",
    )
    p.add_argument(
        "--media-dir",
        default=os.path.join(experiment_root, "media"),
        help="输出目录",
    )
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件路径")
    return p.parse_args()


def run_ffmpeg(ffmpeg, args):
    cmd = [ffmpeg, "-y", "-v", "error"] + args
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        sys.exit(f"找不到视频：{args.video}，请先运行 download_assets.py")
    if shutil.which(args.ffmpeg) is None and not os.path.isfile(args.ffmpeg):
        sys.exit("找不到 ffmpeg，请安装并加入 PATH，或用 --ffmpeg 指定路径")

    os.makedirs(args.media_dir, exist_ok=True)
    image_path = os.path.join(args.media_dir, "frame.jpg")
    audio_path = os.path.join(args.media_dir, "audio.wav")

    print(f"提取首帧 -> {image_path}")
    run_ffmpeg(args.ffmpeg, ["-i", args.video, "-frames:v", "1", image_path])

    print(f"提取前 13 秒音频 -> {audio_path}")
    run_ffmpeg(
        args.ffmpeg,
        ["-i", args.video, "-t", "13", "-ac", "1", "-ar", "16000", audio_path],
    )

    print("素材准备完成。")


if __name__ == "__main__":
    main()
