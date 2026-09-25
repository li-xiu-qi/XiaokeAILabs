#!/usr/bin/env python3
"""Jev-Omni 最小验证：加载一次模型，四个模态各跑一个用例。

权重用默认缓存；若下载时用了 --target-dir，这里传相同的 --models-dir。
需要 ffmpeg 在 PATH（音频和视频预处理用）。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime import (
    setup_environment,
    patch_ffmpeg,
    default_paths,
    ensure_jev_omni_importable,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", default=None, help="权重缓存目录，需与下载时一致")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    setup_environment(args.models_dir)
    patch_ffmpeg()

    paths = default_paths()
    for name in ("video", "image", "audio"):
        if not os.path.isfile(paths[name]):
            sys.exit(
                f"缺少 {name} 素材：{paths[name]}，请先运行 download_assets.py"
            )

    snapshot_path = ensure_jev_omni_importable(args.models_dir)
    if snapshot_path is None:
        sys.exit("在缓存中找不到 Jev-Omni，请先运行 download_models.py")
    from jev_omni import load_jev_omni

    print("加载模型，首次约需 8 分钟...")
    t0 = time.perf_counter()
    clf = load_jev_omni(device=args.device)
    print(f"模型加载完成，用时 {time.perf_counter() - t0:.0f} 秒")

    cases = [
        (
            "文本，英文",
            dict(
                state="The meeting starts at 10 AM. It is now 9 AM.",
                question="Has the meeting started?",
                options=["Yes", "No"],
            ),
        ),
        (
            "文本，中文",
            dict(
                state="我这个月信用卡被扣了两次费，请尽快把多扣的钱退给我",
                question="这个请求应该由哪个部门处理？",
                options=["账务", "技术", "销售", "其他"],
            ),
        ),
        (
            "图像",
            dict(
                state="一张动物照片。",
                question="画面的主体是什么？",
                options=["猫", "狗", "人物", "汽车"],
                media=paths["image"],
                modality="image",
            ),
        ),
        (
            "音频",
            dict(
                state="一段约 12 秒的录音。",
                question="音频里是否包含人声？",
                options=["人声对话", "仅音乐", "静音"],
                media=paths["audio"],
                modality="audio",
            ),
        ),
        (
            "视频",
            dict(
                state="一段约 19 秒的人物近景片段。",
                question="视频里的人主要在做什么？",
                options=["按太阳穴表现疲惫", "吃东西", "跑步", "睡觉"],
                media=paths["video"],
                modality="video",
            ),
        ),
    ]

    for tag, kwargs in cases:
        t = time.perf_counter()
        result = clf.predict(**kwargs)
        elapsed = (time.perf_counter() - t) * 1000
        print(f"\n[{tag}] 用时 {elapsed:.0f} ms")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n四个模态全部验证通过。")


if __name__ == "__main__":
    main()
