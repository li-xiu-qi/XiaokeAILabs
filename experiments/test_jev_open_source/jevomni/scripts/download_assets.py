#!/usr/bin/env python3
"""下载测试素材：Jev-Omni 官方仓库自带的演示视频。

视频约 20 MB，23 秒，含视频流与音轨。下载到实验目录 assets/。
图片与音频由 prepare_media.py 从此视频派生，本脚本只负责视频。
"""
import argparse
import os
import sys

JEV_OMNI_ID = "akhilaaa3/Jev-Omni"
VIDEO_FILE = "assets/the-wall-23s.mp4"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mirror", action="store_true", help="使用 hf-mirror 镜像")
    p.add_argument(
        "--assets-dir",
        default=None,
        help="视频存放目录，默认实验目录下的 assets/",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    here = os.path.dirname(os.path.abspath(__file__))
    experiment_root = os.path.dirname(here)
    assets_dir = args.assets_dir or os.path.join(experiment_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    import shutil
    from huggingface_hub import hf_hub_download

    print("下载官方演示视频（约 20 MB）...")
    # local_dir 会保留仓库内的 assets/ 子路径，先下到实验根目录再移动，
    # 避免出现 assets/assets/ 嵌套。
    downloaded = hf_hub_download(
        JEV_OMNI_ID,
        VIDEO_FILE,
        local_dir=experiment_root,
    )
    target = os.path.join(assets_dir, os.path.basename(VIDEO_FILE))
    if os.path.abspath(downloaded) != os.path.abspath(target):
        shutil.move(downloaded, target)
    print(f"完成：{target}")


if __name__ == "__main__":
    sys.exit(main())
